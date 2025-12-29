#!/bin/bash

# THOR에서 실행하는 스크립트
# - 로컬(THOR): 카메라 launch, 조이스틱, GUI 실행
# - 원격(SPARK): SSH로 접속해서 메인 제어 시스템 실행

# 원격 SPARK 연결 정보 설정 (필요시 수정)
REMOTE_IP="192.168.80.9"
REMOTE_USER="dgx_allex_one"
REMOTE_PW="00000000"

cleanup() {
    echo -e "\n\033[1;31m[종료] 모든 프로세스를 강제 정지합니다...\033[0m"

    # 원격 SPARK 프로세스 정리
    echo "[1/2] 원격 PC($REMOTE_IP) 프로세스 정리 중..."
    sshpass -p "$REMOTE_PW" ssh $REMOTE_USER@$REMOTE_IP "pkill -9 -f ros2" 2>/dev/null

    # 로컬 프로세스 정리
    echo "[2/2] 로컬 PC 프로세스 정리 중..."
    kill -TERM -$(ps -o pgid= -p $$ | tr -d ' ') 2>/dev/null
    
    exit
}

trap cleanup SIGINT

# 작업 디렉토리로 이동
cd ~/allex_ces_idle_interaction_gui

# ROS2 환경 설정
source /opt/ros/humble/setup.bash
source ~/allex_ces_idle_interaction_gui/install/setup.bash

# 1. 로컬 THOR 노드 실행 (카메라, 조이스틱, GUI)
echo "[진행] 로컬 THOR: 카메라 및 조이스틱/GUI 노드 실행 중..."

# 카메라 launch 실행 (백그라운드)
ros2 launch orbbec_camera femto_bolt.launch.py &

# 조이스틱 및 GUI 노드 실행 (백그라운드)
ros2 launch allex_ces_idle_interaction thor_nodes.launch.py &

# 2. 센서 안정화 대기
sleep 15

# 3. 원격 SPARK에서 메인 제어 시스템 실행
echo "[진행] 원격 SPARK: 메인 제어 시스템 실행 중..."
sshpass -p "$REMOTE_PW" ssh $REMOTE_USER@$REMOTE_IP \
    "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID; \
     source ~/allex_ces_idle_interaction_gui/install/setup.bash; \
     ros2 launch allex_ces_idle_interaction allex_idle_interaction.launch.py"

# 모든 작업이 완료될 때까지 대기
wait
