# ALLEX Testbed

얼굴/허리 제어 테스트베드 패키지 - PID 튜닝 및 알고리즘 테스트

## 개요

이 패키지는 얼굴 추적 및 허리 제어 알고리즘을 테스트하고 튜닝하기 위한 테스트베드입니다. 기존 `allex_ces_idle_interaction` 패키지와 동일한 토픽 구조를 사용하지만, 가장 기본적인 제어 알고리즘부터 시작하여 점진적으로 개선할 수 있습니다.

## 주요 기능

1. **기본 PID 제어**: 가장 간단한 형태의 PID 제어 알고리즘
2. **PID 게인 실시간 튜닝**: GUI를 통한 실시간 PID 게인 조정
3. **목표 명령 직접 설정**: 각도 명령을 직접 설정하여 이동 테스트
4. **상태 모니터링**: 현재 각도, 목표 각도, 오차 실시간 표시
5. **스무딩 파라미터 조정**: 움직임의 부드러움 조정

## 사용 방법

### 빌드

```bash
cd ~/allex_ces_idle_interaction_gui
colcon build --packages-select allex_testbed
source install/setup.bash
```

### 실행

```bash
# 테스트베드 노드 실행 (GUI 포함)
ros2 launch allex_testbed testbed.launch.py
```

### GUI 사용법

1. **PID 튜닝 탭**
   - 목(Yaw/Pitch) 및 허리 PID 게인 조정
   - 스무딩 파라미터 조정
   - "PID 게인 적용" 버튼으로 실시간 적용
   - "PID 상태 리셋" 버튼으로 integral 초기화

2. **목표 명령 탭**
   - 직접 각도 명령 설정
   - "목표 설정" 버튼으로 이동
   - "0도로 이동" 버튼으로 원점 복귀
   - "40도로 이동" 버튼으로 SEARCHING 테스트

3. **상태 모니터링 탭**
   - 현재 각도, 목표 각도, 오차 실시간 표시
   - 로그 확인

## 토픽 구조

기존 `allex_ces_idle_interaction` 패키지와 동일한 토픽을 사용합니다:

- 구독:
  - `/allex_camera/tracking_result` - 추적 결과
  - `/allex_camera/controller_control` - 제어 명령
  - `/robot_outbound_data/theOne_neck/joint_positions_deg` - 목 위치
  - `/robot_outbound_data/theOne_waist/joint_positions_deg` - 허리 위치

- 발행:
  - `/robot_inbound/theOne_neck/joint_command` - 목 명령
  - `/robot_inbound/theOne_waist/joint_command` - 허리 명령
  - `/allex_camera/neck_angle` - 목 각도 정보
  - `/allex_testbed/status` - 상태 정보 (GUI용)

- 테스트베드 전용 토픽:
  - `/allex_testbed/pid_tune` - PID 게인 튜닝 명령
  - `/allex_testbed/target_command` - 직접 목표 명령

## 제어 알고리즘

### 기본 PID 제어

가장 간단한 형태의 PID 제어:

```
output = Kp * error + Ki * integral + Kd * derivative
```

- **P (Proportional)**: 현재 오차에 비례
- **I (Integral)**: 누적 오차 (Steady State Error 제거)
- **D (Derivative)**: 오차 변화율 (진동 억제)

### 스무딩

스무딩 파라미터(0.0 ~ 1.0)로 움직임의 부드러움 조정:
- 1.0: 스무딩 없음 (즉시 반응)
- 0.0: 최대 스무딩 (매우 부드러움)

## 테스트 시나리오

1. **40도 이동 테스트**: GUI에서 "40도로 이동" 버튼 클릭하여 SEARCHING 문제 재현
2. **PID 게인 튜닝**: Kp 값을 조정하여 목표 각도까지 도달하는지 확인
3. **스무딩 테스트**: 스무딩 파라미터를 조정하여 움직임의 부드러움 확인

## 향후 개선 사항

- [ ] 더 정교한 PID 알고리즘 (Adaptive PID 등)
- [ ] 궤적 계획 (Trajectory Planning)
- [ ] 속도 제한 및 가속도 제한
- [ ] 진동 억제 알고리즘
- [ ] 데이터 로깅 및 분석

