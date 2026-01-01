#!/bin/bash

# ============================================
# ALLEX Idle Interaction 실행 스크립트
# PC1 (로컬): GUI + Joystick
# PC2 (mars): 카메라
# PC3 (dgx_allex_one): 제어 노드
# ============================================

# 원격 PC 설정
PC2_IP="192.168.80.8"
PC2_USER="mars"
PC2_PW="0000"

PC3_IP="192.168.80.9"
PC3_USER="dgx_allex_one"
PC3_PW="00000000"

# 프로세스 ID 추적 (종료 시 사용)
PC1_PID=""
PC2_PID=""
PC3_PID=""

cleanup() {
    echo -e "\n[종료] 원격 및 로컬 프로세스 정리 시작..."

    # 1. PC3 (제어기 PC) 원격 노드 강제 종료
    echo "[종료] PC3 (제어기) 노드 종료 중..."
    sshpass -p "$PC3_PW" ssh $PC3_USER@$PC3_IP "pkill -INT -f ros2; sleep 1; pkill -9 -f ros2" 2>/dev/null
    sleep 1

    # 2. PC2 (카메라 PC) 원격 노드 강제 종료
    echo "[종료] PC2 (카메라) 노드 종료 중..."
    sshpass -p "$PC2_PW" ssh $PC2_USER@$PC2_IP "pkill -INT -f ros2; sleep 1; pkill -9 -f ros2; pkill -9 -f orbbec_camera" 2>/dev/null
    sleep 1

    # 3. PC1 (로컬) ROS 2 노드들 정리
    echo "[종료] PC1 (로컬) 노드 종료 중..."
    pkill -INT -f ros2 2>/dev/null
    sleep 2
    pkill -9 -f ros2 2>/dev/null
    pkill -9 -f "idle_interaction_gui" 2>/dev/null
    pkill -9 -f "joystick_control" 2>/dev/null
    
    echo "[종료] 모든 프로세스 정리 완료"
    exit 0
}

# SIGINT(Ctrl+C) 및 SIGTERM 수신 시 cleanup 실행
trap cleanup SIGINT SIGTERM

# ROS 2 환경 확인
if [ -z "$ROS_DOMAIN_ID" ]; then
    echo "[경고] ROS_DOMAIN_ID가 설정되지 않았습니다. 기본값 0을 사용합니다."
    export ROS_DOMAIN_ID=0
fi

# 워크스페이스 소스 확인
if [ -f "$HOME/allex_ces_idle_interaction_gui/install/setup.bash" ]; then
    source "$HOME/allex_ces_idle_interaction_gui/install/setup.bash"
    echo "[설정] 로컬 워크스페이스 소스 완료"
elif [ -f "./install/setup.bash" ]; then
    source "./install/setup.bash"
    echo "[설정] 로컬 워크스페이스 소스 완료 (상대 경로)"
else
    echo "[경고] 워크스페이스 setup.bash를 찾을 수 없습니다."
    echo "      수동으로 source를 실행하거나 절대 경로를 확인하세요."
fi

# PC3 (제어기 PC) 원격 노드 실행
echo "[진행] PC3 (제어기) 노드 실행 중..."
sshpass -p "$PC3_PW" ssh $PC3_USER@$PC3_IP "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID; source ~/allex_ces_idle_interaction_gui/install/setup.bash; ros2 launch allex_ces_idle_interaction allex_idle_interaction.launch.py" &
PC3_PID=$!
echo "[진행] PC3 프로세스 ID: $PC3_PID"

# PC3 안정화 대기
sleep 5

# PC2 (카메라 PC) 원격 노드 실행
echo "[진행] PC2 (카메라) 노드 실행 중..."
sshpass -p "$PC2_PW" ssh $PC2_USER@$PC2_IP "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID; source ~/allex_ces_idle_interaction_gui/install/setup.bash; ros2 launch allex_ces_idle_interaction pc2_camera.launch.py" &
PC2_PID=$!
echo "[진행] PC2 프로세스 ID: $PC2_PID"

# PC2 안정화 대기
sleep 5

# PC1 (로컬) 노드 실행
echo "[진행] PC1 (로컬) GUI 및 Joystick 노드 실행 중..."
ros2 launch allex_ces_idle_interaction pc1_gui_joystick.launch.py &
PC1_PID=$!
echo "[진행] PC1 프로세스 ID: $PC1_PID"

# 모든 자식 프로세스가 종료되거나 신호가 올 때까지 대기
echo ""
echo "============================================"
echo "[대기] 모든 노드가 실행 중입니다."
echo "      Ctrl+C를 누르면 모든 프로세스가 종료됩니다."
echo "============================================"
wait

