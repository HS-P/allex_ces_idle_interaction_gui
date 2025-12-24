#!/bin/bash
# DGX SPARK 실행 스크립트
# 메인 추적 및 제어 시스템 실행

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
# pkill -f "allex_idle_interaction.launch.py"

echo "=========================================="
echo "DGX SPARK 실행 스크립트"
echo "=========================================="
echo ""

# 메인 시스템 실행
echo "ALLEX Idle Interaction 시스템 실행 중..."
ros2 launch allex_ces_idle_interaction allex_idle_interaction.launch.py > logs/spark_main.log 2>&1 &
MAIN_PID=$!

echo "  메인 시스템 프로세스 PID: $MAIN_PID"
echo "  로그 파일: logs/spark_main.log"
echo ""

echo "=========================================="
echo "실행 완료!"
echo "=========================================="
echo ""
echo "실행 중인 프로세스:"
echo "  - 메인 시스템: PID $MAIN_PID"
echo ""
echo "프로세스 종료 방법:"
echo "  kill $MAIN_PID"
echo ""
echo "또는:"
echo "  pkill -f 'allex_idle_interaction.launch.py'"
echo ""
echo "로그 확인:"
echo "  tail -f logs/spark_main.log"
echo ""

