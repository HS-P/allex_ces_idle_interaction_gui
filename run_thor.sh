#!/bin/bash
# DGX Thor 실행 스크립트
# Orbbec Femto Bolt 카메라와 GUI를 백그라운드로 실행

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 로그 디렉토리 생성
mkdir -p logs

# ROS 2 환경 설정
source /opt/ros/humble/setup.bash  # ROS 2 버전에 맞게 수정
if [ -f install/setup.bash ]; then
    source install/setup.bash
fi

# 기존 프로세스 종료 (선택사항)
# pkill -f "femto_bolt.launch.py"
# pkill -f "idle_interaction_gui"

echo "=========================================="
echo "DGX Thor 실행 스크립트"
echo "=========================================="
echo ""

# 1. Orbbec Femto Bolt 카메라 실행
echo "[1/2] Orbbec Femto Bolt 카메라 실행 중..."
ros2 launch orb_ecc_camera femto_bolt.launch.py > logs/thor_camera.log 2>&1 &
CAMERA_PID=$!
echo "  카메라 프로세스 PID: $CAMERA_PID"
echo "  로그 파일: logs/thor_camera.log"
sleep 2

# 2. GUI 실행 (있는 경우)
if command -v idle_interaction_gui &> /dev/null; then
    echo "[2/2] GUI 애플리케이션 실행 중..."
    idle_interaction_gui > logs/thor_gui.log 2>&1 &
    GUI_PID=$!
    echo "  GUI 프로세스 PID: $GUI_PID"
    echo "  로그 파일: logs/thor_gui.log"
else
    echo "[2/2] GUI 애플리케이션을 찾을 수 없습니다. 건너뜁니다."
    GUI_PID=""
fi

echo ""
echo "=========================================="
echo "실행 완료!"
echo "=========================================="
echo ""
echo "실행 중인 프로세스:"
echo "  - 카메라: PID $CAMERA_PID"
if [ ! -z "$GUI_PID" ]; then
    echo "  - GUI: PID $GUI_PID"
fi
echo ""
echo "프로세스 종료 방법:"
echo "  kill $CAMERA_PID"
if [ ! -z "$GUI_PID" ]; then
    echo "  kill $GUI_PID"
fi
echo ""
echo "또는 모든 프로세스 종료:"
echo "  pkill -f 'femto_bolt.launch.py'"
if [ ! -z "$GUI_PID" ]; then
    echo "  pkill -f 'idle_interaction_gui'"
fi
echo ""
echo "로그 확인:"
echo "  tail -f logs/thor_camera.log"
if [ ! -z "$GUI_PID" ]; then
    echo "  tail -f logs/thor_gui.log"
fi
echo ""

