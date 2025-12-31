#!/bin/bash

# THOR에서 실행하는 스크립트
# - 로컬(THOR): 카메라 launch, 조이스틱, GUI 실행
# - 원격(SPARK): SSH로 접속해서 메인 제어 시스템 실행

# 원격 SPARK 연결 정보 설정 (필요시 수정)
REMOTE_IP="192.168.80.9"
REMOTE_USER="dgx_allex_one"
REMOTE_PW="00000000"

# cleanup 중복 실행 방지 플래그
CLEANUP_DONE=false

# PID 변수 초기화
SPARK_PID=""
THOR_PID=""
GUI_PID=""

cleanup() {
    # 이미 cleanup이 실행 중이면 무시
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    
    echo -e "\n\033[1;31m[종료] 모든 프로세스를 강제 정지합니다...\033[0m"
    
    # 원격 SPARK 프로세스 정리
    echo "[1/3] 원격 PC($REMOTE_IP) 프로세스 정리 중..."
    sshpass -p "$REMOTE_PW" ssh $REMOTE_USER@$REMOTE_IP \
        "pkill -9 -f 'ros2 launch.*allex_idle_interaction' 2>/dev/null; \
         pkill -9 -f 'yolo_detection_node' 2>/dev/null; \
         pkill -9 -f 'tracking_fsm_node' 2>/dev/null; \
         pkill -9 -f 'gaze_controller_neck_waist_node' 2>/dev/null; \
         pkill -9 -f 'allex_idle_interaction_node' 2>/dev/null; \
         pkill -9 -f 'component_container' 2>/dev/null; \
         pkill -9 -f 'camera.camera' 2>/dev/null" 2>/dev/null
    
    # 로컬 프로세스 그룹 단위로 종료 (PID 기반)
    echo "[2/3] 로컬 PC 프로세스 그룹 정리 중..."
    if [ -n "$SPARK_PID" ] && kill -0 $SPARK_PID 2>/dev/null; then
        echo "  - SPARK 프로세스 그룹 종료 중 (PID: $SPARK_PID)..."
        kill -TERM -$SPARK_PID 2>/dev/null
    fi
    if [ -n "$THOR_PID" ] && kill -0 $THOR_PID 2>/dev/null; then
        echo "  - THOR 프로세스 그룹 종료 중 (PID: $THOR_PID)..."
        kill -TERM -$THOR_PID 2>/dev/null
    fi
    if [ -n "$GUI_PID" ] && kill -0 $GUI_PID 2>/dev/null; then
        echo "  - GUI 프로세스 그룹 종료 중 (PID: $GUI_PID)..."
        kill -TERM -$GUI_PID 2>/dev/null
    fi
    
    sleep 1
    
    # 강제 종료
    if [ -n "$SPARK_PID" ] && kill -0 $SPARK_PID 2>/dev/null; then
        kill -KILL -$SPARK_PID 2>/dev/null
    fi
    if [ -n "$THOR_PID" ] && kill -0 $THOR_PID 2>/dev/null; then
        kill -KILL -$THOR_PID 2>/dev/null
    fi
    if [ -n "$GUI_PID" ] && kill -0 $GUI_PID 2>/dev/null; then
        kill -KILL -$GUI_PID 2>/dev/null
    fi
    
    # 개별 노드 프로세스 강제 종료 (ros2 node list로 확인된 노드들)
    echo "[3/3] 개별 노드 프로세스 강제 종료 중..."
    
    # 로컬 노드 프로세스 종료
    pkill -9 -f "yolo_detection_node" 2>/dev/null
    pkill -9 -f "tracking_fsm_node" 2>/dev/null
    pkill -9 -f "gaze_controller_neck_waist_node" 2>/dev/null
    pkill -9 -f "allex_idle_interaction_node" 2>/dev/null
    pkill -9 -f "idle_interaction_gui_node" 2>/dev/null
    pkill -9 -f "joystick_control_node" 2>/dev/null
    pkill -9 -f "orbbec_camera" 2>/dev/null
    pkill -9 -f "camera.camera" 2>/dev/null
    pkill -9 -f "component_container" 2>/dev/null
    pkill -9 -f "ros2 launch" 2>/dev/null
    
    # ros2 node list로 확인된 노드들의 PID 직접 찾아서 종료
    if command -v ros2 &> /dev/null; then
        # ROS2 노드 목록 가져오기
        NODE_LIST=$(ros2 node list 2>/dev/null | grep -v "^/" | sed 's|^/||' || true)
        
        if [ -n "$NODE_LIST" ]; then
            echo "  - 남아있는 노드 발견, 강제 종료 중..."
            for node_name in $NODE_LIST; do
                # 노드 이름으로 프로세스 찾기
                PIDS=$(pgrep -f "$node_name" 2>/dev/null || true)
                if [ -n "$PIDS" ]; then
                    echo "    - $node_name 종료 중 (PIDs: $PIDS)..."
                    echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
                fi
            done
        fi
    fi
    
    sleep 1
    
    # 원격 SPARK 노드 프로세스도 강제 종료
    echo "[4/4] 원격 SPARK 개별 노드 프로세스 강제 종료 중..."
    sshpass -p "$REMOTE_PW" ssh $REMOTE_USER@$REMOTE_IP \
        "pkill -9 -f 'yolo_detection_node' 2>/dev/null; \
         pkill -9 -f 'tracking_fsm_node' 2>/dev/null; \
         pkill -9 -f 'gaze_controller_neck_waist_node' 2>/dev/null; \
         pkill -9 -f 'allex_idle_interaction_node' 2>/dev/null; \
         pkill -9 -f 'component_container' 2>/dev/null; \
         pkill -9 -f 'camera.camera' 2>/dev/null; \
         pkill -9 -f 'ros2 launch' 2>/dev/null" 2>/dev/null
    
    echo "[완료] 모든 프로세스 정리 완료"
    exit
}

trap cleanup SIGINT SIGTERM EXIT

# 스크립트가 위치한 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 실행 전 기존 프로세스 정리 함수 (cleanup과 달리 exit 없음)
pre_cleanup() {
    echo "[정리] 기존 프로세스 확인 및 정리 중..."
    
    # 원격 SPARK 프로세스 정리
    echo "  [1/2] 원격 SPARK 프로세스 정리 중..."
    sshpass -p "$REMOTE_PW" ssh $REMOTE_USER@$REMOTE_IP \
        "pkill -9 -f 'ros2 launch.*allex_idle_interaction' 2>/dev/null; \
         pkill -9 -f 'yolo_detection_node' 2>/dev/null; \
         pkill -9 -f 'tracking_fsm_node' 2>/dev/null; \
         pkill -9 -f 'gaze_controller_neck_waist_node' 2>/dev/null; \
         pkill -9 -f 'allex_idle_interaction_node' 2>/dev/null" 2>/dev/null
    
    # 로컬 프로세스 정리 (패턴 기반)
    echo "  [2/2] 로컬 프로세스 정리 중..."
    pkill -9 -f "ros2 launch.*thor_camera_joystick" 2>/dev/null
    pkill -9 -f "ros2 launch.*thor_gui" 2>/dev/null
    pkill -9 -f "idle_interaction_gui_node" 2>/dev/null
    pkill -9 -f "joystick_control_node" 2>/dev/null
    pkill -9 -f "orbbec_camera" 2>/dev/null
    pkill -9 -f "yolo_detection_node" 2>/dev/null
    pkill -9 -f "tracking_fsm_node" 2>/dev/null
    pkill -9 -f "gaze_controller_neck_waist_node" 2>/dev/null
    pkill -9 -f "allex_idle_interaction_node" 2>/dev/null
    pkill -9 -f "component_container" 2>/dev/null
    pkill -9 -f "camera.camera" 2>/dev/null
    
    # ros2 node list로 확인된 노드들의 PID 직접 찾아서 종료
    if command -v ros2 &> /dev/null; then
        NODE_LIST=$(ros2 node list 2>/dev/null | grep -v "^/" | sed 's|^/||' || true)
        if [ -n "$NODE_LIST" ]; then
            for node_name in $NODE_LIST; do
                PIDS=$(pgrep -f "$node_name" 2>/dev/null || true)
                if [ -n "$PIDS" ]; then
                    echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
                fi
            done
        fi
    fi
    
    sleep 2
    echo "[완료] 기존 프로세스 정리 완료"
}

source /opt/ros/jazzy/setup.bash

# 패키지 설치 디렉토리 source
if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    source "$SCRIPT_DIR/install/setup.bash"
fi

# 실행 전 기존 프로세스 정리
pre_cleanup

# 실행 전 기존 프로세스 확인 및 정리 (선택적)
echo "[확인] 기존 프로세스 확인 중..."
EXISTING_NODES=$(ros2 node list 2>/dev/null | grep -E "(allex_idle_interaction_node|gaze_controller_neck_waist_node|tracking_fsm_node|yolo_detection_node|idle_interaction_gui_node|joystick_control_node)" 2>/dev/null | wc -l)
# EXISTING_NODES가 빈 문자열이거나 숫자가 아닌 경우 0으로 처리
if [ -z "$EXISTING_NODES" ] || ! [[ "$EXISTING_NODES" =~ ^[0-9]+$ ]]; then
    EXISTING_NODES=0
fi

if [ "$EXISTING_NODES" -gt 0 ]; then
    echo "[경고] 기존 노드가 실행 중입니다. 정리합니다..."
    # cleanup 대신 pre_cleanup을 다시 호출 (exit 없이 정리만 수행)
    pre_cleanup
    sleep 2
    echo "[완료] 기존 프로세스 정리 완료"
else
    echo "[확인] 기존 노드 없음. 정상 진행합니다."
fi

# 1. 원격 SPARK에서 메인 제어 시스템 실행 (먼저 실행, 새 프로세스 그룹으로)
echo "[진행] 원격 SPARK: 메인 제어 시스템 실행 중..."
setsid sshpass -p "$REMOTE_PW" ssh $REMOTE_USER@$REMOTE_IP \
    "export ROS_DOMAIN_ID=$ROS_DOMAIN_ID; \
     cd ~/allex_ces_idle_interaction_gui && \
     if [ -f /opt/ros/jazzy/setup.bash ]; then source /opt/ros/jazzy/setup.bash; fi && \
     source ~/allex_ces_idle_interaction_gui/install/setup.bash && \
     ros2 launch allex_ces_idle_interaction allex_idle_interaction.launch.py" & SPARK_PID=$!

# 2. 센서 안정화 대기
sleep 5

# 3. 로컬 THOR 노드 실행 (카메라, 조이스틱 - GUI 제외, 새 프로세스 그룹으로)
echo "[진행] 로컬 THOR: 카메라 및 조이스틱 노드 실행 중..."
setsid ros2 launch allex_ces_idle_interaction thor_camera_joystick.launch.py & THOR_PID=$!

# 4. 센서 안정화 대기
sleep 10

# 5. GUI 노드 실행 (가장 마지막에 실행, 새 프로세스 그룹으로)
echo "[진행] 로컬 THOR: GUI 노드 실행 중..."
setsid ros2 launch allex_ces_idle_interaction thor_gui.launch.py & GUI_PID=$!

# 모든 작업이 완료될 때까지 대기
wait
