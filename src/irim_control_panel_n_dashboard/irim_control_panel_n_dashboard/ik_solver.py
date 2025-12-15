#!/usr/bin/env python3
"""
ik_solver.py: Damped Least Squares IK solver for multiple robots.
"""

import numpy as np
from dh_params import DH_CONFIG


def dh_transform(theta, d, a, alpha):
    """
    Compute individual DH transform.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,       sa,       ca,      d],
        [0,        0,        0,      1]
    ])


def compute_hts(angles, dh_params, HT_b20, HT_t2e):
    """
    Compute cumulative homogeneous transforms for each joint frame
    and the end-effector.

    angles: list of joint angles [rad]
    dh_params: list of dicts with keys theta_offset, d, a, alpha
    HT_b20: 4x4 base-to-first-link transform
    HT_t2e: 4x4 last-link-to-end-effector transform

    Returns:
      Ts: list of 4x4 np.array transforms: [T_base, T1, ..., TN, T_ee]
    """
    # Start from the base-to-link0 transform
    Ts = [HT_b20.copy()]
    # Sequentially apply each joint's DH transform
    for q, param in zip(angles, dh_params):
        theta = param['theta_offset'] + q
        d = param['d']
        a = param['a']
        alpha = param['alpha']
        A = dh_transform(theta, d, a, alpha)
        Ts.append(Ts[-1] @ A)
    # Append end-effector transform
    Ts.append(Ts[-1] @ HT_t2e)
    return Ts


def compute_jacobian(hts):
    """
    Build the 6xN Jacobian given list of transforms Ts:
    Ts[0] = T_base, Ts[1]=T1, ..., Ts[N]=TN, Ts[N+1]=T_ee.

    Returns:
      J: np.ndarray shape (6, N)
    """
    N = len(hts) - 2  # number of joints
    J = np.zeros((6, N))
    p_ee = hts[-1][:3, 3]
    for i in range(N):
        Ti = hts[i]
        zi = Ti[:3, 2]
        pi = Ti[:3, 3]
        # Linear velocity component
        Jv = np.cross(zi, (p_ee - pi))
        # Angular velocity component
        Jw = zi
        J[:3, i] = Jv
        J[3:, i] = Jw
    return J


def solve_ik(robot_name, current_angles, delta_x, damping=0.1,
             method='dls', null_motion_scalar=0.0, svd_thresh=1e-6):
    """
    Inverse kinematics update using either Damped Least Squares (DLS) or
    SVD-based pseudoinverse with optional null-space motion.

    robot_name: key in DH_CONFIG
    current_angles: list of N joint angles [rad]
    delta_x: list-like of length 6: [dx, dy, dz, dalpha_x, dalpha_y, dalpha_z]
    damping: damping factor for DLS
    method: 'dls' or 'svd'
    null_motion: optional joint-space vector (length N) for null-space motion
    svd_thresh: singular value threshold for pseudoinverse

    Returns updated joint angles (rad).
    """
    # Load robot-specific DH configuration
    cfg = DH_CONFIG[robot_name]
    HT_b20 = cfg['HT_b20']
    HT_t2e = cfg['HT_t2e']
    dh_params = cfg['dh_parameters']

    # Forward kinematics and Jacobian
    Ts = compute_hts(current_angles, dh_params, HT_b20, HT_t2e)
    J = compute_jacobian(Ts)
    dx = np.array(delta_x).reshape((6,))

    if method == 'dls':
        JT = J.T
        inv = np.linalg.inv(J @ JT + (damping**2) * np.eye(6))
        dq = JT @ inv @ dx
    elif method == 'svd':
        # SVD decomposition
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        # Invert singular values above threshold
        S_inv = np.array([1/s if s > svd_thresh else 0 for s in S])
        J_pinv = Vt.T @ np.diag(S_inv) @ U.T
        # Primary task: Cartesian update
        dq_primary = J_pinv @ dx

        # null_motion_scalar 을 null-space 방향으로 투영
        if null_motion_scalar != 0.0:
            P_null = np.eye(len(current_angles)) - J_pinv @ J
            # 기본 방향: all-ones 벡터 (원하시면 다른 방향으로 바꿔도 좋습니다)
            null_dir = np.ones((len(current_angles),))
            dq = dq_primary + null_motion_scalar * (P_null @ null_dir)
        else:
            dq = dq_primary


        # # Null-space projection if provided
        # if null_motion is not None:
        #     null_motion = np.array(null_motion).reshape((len(current_angles),))
        #     P_null = np.eye(len(current_angles)) - J_pinv @ J
        #     dq = dq_primary + P_null @ null_motion
        # else:
        #     dq = dq_primary
    else:
        raise ValueError(f"Unknown IK method: {method}")

    # Return updated angles
    return (np.array(current_angles) + dq).tolist()





def extract_pose(ht):
    """
    Extract a 6-vector pose [x, y, z, rot_x, rot_y, rot_z] from a homogeneous transform.
    Rotations are represented as a rotation vector (axis * angle).
    """
    p = ht[:3, 3]
    R = ht[:3, :3]
    # Compute angle
    angle = np.arccos(max(min((np.trace(R) - 1) / 2, 1.0), -1.0))
    if abs(angle) < 1e-8:
        return np.concatenate([p, np.zeros(3)])
    # Rotation axis
    axis = np.array([R[2,1] - R[1,2],
                     R[0,2] - R[2,0],
                     R[1,0] - R[0,1]]) / (2 * np.sin(angle))
    return np.concatenate([p, axis * angle])

def numerical_jacobian(robot_name, q, delta=1e-5):
    """
    중앙차분으로 수치 자코비안 계산.
    """
    dh_params = DH_CONFIG[robot_name]['dh_parameters']
    HT_b20    = DH_CONFIG[robot_name]['HT_b20']
    HT_t2e    = DH_CONFIG[robot_name]['HT_t2e']

    n     = len(q)
    J_num = np.zeros((6, n))

    for i in range(n):
        # 1) delta 벡터 생성
        dq = np.zeros(n)
        dq[i] = delta

        # 2) 리스트로 변환
        q_plus  = list(q  + dq)
        q_minus = list(q  - dq)

        # 3) 순전파
        Ts_plus  = compute_hts(q_plus,  dh_params, HT_b20, HT_t2e)
        Ts_minus = compute_hts(q_minus, dh_params, HT_b20, HT_t2e)

        # 4) 위치 차분 → 선속도 성분
        p_plus  = Ts_plus[-1][:3, 3]
        p_minus = Ts_minus[-1][:3, 3]
        J_num[:3, i] = (p_plus - p_minus) / (2 * delta)

        # 5) 회전 차분 → 각속도 성분 (axis-angle 방식)
        R_plus  = Ts_plus[-1][:3, :3]
        R_minus = Ts_minus[-1][:3, :3]
        R_diff  = R_plus @ R_minus.T

        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        if abs(angle) < 1e-8:
            J_num[3:, i] = 0
        else:
            axis = np.array([
                R_diff[2,1] - R_diff[1,2],
                R_diff[0,2] - R_diff[2,0],
                R_diff[1,0] - R_diff[0,1]
            ]) / (2 * np.sin(angle))
            # / (2*delta) 까지 묶어서 계산
            J_num[3:, i] = axis * angle / (2 * delta)

    return J_num



def main():
    # TODO: set your robot key and initial joint angles in DH_CONFIG
    robot = 'Arm_R'
    q0 = np.array(DH_CONFIG[robot].get('initial_angles', [0]*len(DH_CONFIG[robot]['dh_parameters'])))
    
    # Analytic
    Ts = compute_hts(q0,
                     DH_CONFIG[robot]['dh_parameters'],
                     DH_CONFIG[robot]['HT_b20'],
                     DH_CONFIG[robot]['HT_t2e'])
    J_analytic = compute_jacobian(Ts)
    
    # Numerical
    J_numeric = numerical_jacobian(robot, q0)
    
    np.set_printoptions(precision=6, suppress=True)
    print("Analytic Jacobian:\\n", J_analytic)
    print("\\nNumerical Jacobian:\\n", J_numeric)
    print("\\nDifference (analytic - numeric):\\n", J_analytic - J_numeric)

if __name__ == '__main__':
    main()