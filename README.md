# ALLEX CES Idle Interaction System

ALLEX 로봇의 Idle Interaction 시스템입니다. 사람 추적, 시선 제어, 제스처 인식 등의 기능을 제공합니다.

## 시스템 구성

이 시스템은 두 대의 로봇에서 실행됩니다:

- **DGX SPARK**: 메인 추적 및 제어 시스템 실행
- **DGX Thor**: 카메라 스트리밍 실행

## 시스템 아키텍처

### 노드 구조

시스템은 다음과 같은 ROS2 노드들로 구성됩니다:

1. **yolo_detection_node** (`yolo_detection_node.py`)
   - YOLO 기반 사람 감지
   - 카메라 이미지를 받아 사람 바운딩 박스 검출
   - BotSort 추적 알고리즘으로 객체 추적

2. **tracking_fsm_node** (`tracking_fsm_node.py`)
   - 추적 상태 머신(FSM) 관리
   - 상태 전환 로직 처리 (IDLE, WAITING, TRACKING, LOST, SEARCHING, HELLO, HANDSHAKE)
   - 타겟 선택 및 추적 관리

3. **gaze_controller_neck_waist_node** (`gaze_controller_neck_waist_node.py`)
   - 목과 허리 제어
   - PID 제어를 통한 부드러운 움직임
   - 상태별 제어 전략 적용

4. **allex_idle_interaction_node** (`allex_idle_interaction_node.py`)
   - 전체 상호작용 관리
   - 루틴 제어 (HELLO, HANDSHAKE 등)
   - GUI 명령 처리

### 상태 머신 (TrackingState)

- **IDLE**: 초기 상태, 영자세 상태
- **WAITING**: 타겟 찾기 대기
- **TRACKING**: 타겟 추적 중
- **LOST**: 추적 대상 놓침 (Exponential smoothing으로 감속)
- **SEARCHING**: 주변 탐색 (목과 허리 독립 제어)
- **HELLO**: 인사 제스처 실행
- **HANDSHAKE**: 악수 제스처 실행

### 데이터 흐름

```
카메라 이미지
    ↓
yolo_detection_node (사람 감지)
    ↓
tracking_fsm_node (상태 관리, 타겟 선택)
    ↓
gaze_controller_neck_waist_node (목/허리 제어 명령 생성)
    ↓
로봇 하드웨어
```

### 주요 토픽

- `/camera/color/image_raw/compressed`: 카메라 이미지 (CompressedImage)
- `/allex_camera/detections`: YOLO 감지 결과 (std_msgs/String, JSON)
- `/allex_camera/tracking_result`: 추적 결과 (std_msgs/String, JSON)
- `/allex_camera/tracking_data`: 추적 데이터 (GUI용, std_msgs/String, JSON)
- `/robot_inbound/theOne_neck/joint_command`: 목 관절 명령 (std_msgs/Float64MultiArray)
- `/robot_inbound/theOne_waist/joint_command`: 허리 관절 명령 (std_msgs/Float64MultiArray)
- `/debug/routine`: 루틴 상태 피드백 (std_msgs/String, JSON)

### 제어 전략

#### 초기 추적 (Initial Tracking)
- 타겟 감지 후 처음 0.85초 동안 목만 움직이며 추적
- 허리는 고정 상태 유지
- 부드러운 움직임을 위한 최적화된 PID 게인 사용

#### 일반 추적 (Normal Tracking)
- 목과 허리 협조 제어 (neck-lead / waist-follow 구조)
- 목이 먼저 빠르게 추종하고, 허리가 천천히 따라옴
- 12% 속도 증가 적용

#### LOST 상태 감속
- 타겟을 놓쳤을 때 Exponential smoothing으로 부드럽게 감속
- 목 Yaw는 영자세(0도)로 복귀, Pitch는 현재 각도 유지

#### SEARCHING 모드
- 목과 허리를 독립적으로 제어하여 스캔
- 목: 20~50도(우측) 또는 -50~-20도(좌측)
- 허리: 0~35도(우측) 또는 -35~0도(좌측)
- Exponential smoothing으로 천천히 이동

## 실행 방법

### DGX Thor에서 실행

카메라 스트리밍을 실행합니다:

```bash
cd ~/allex_ces_idle_interaction_gui
source /opt/ros/humble/setup.bash
source install/setup.bash

# 카메라 실행
ros2 launch orb_ecc_camera femto_bolt.launch.py
```

또는 자동 실행 스크립트 사용:

```bash
cd ~/allex_ces_idle_interaction_gui
./run_thor.sh
```

### DGX SPARK에서 실행

메인 추적 및 제어 시스템을 실행합니다:

```bash
cd ~/allex_ces_idle_interaction_gui
source /opt/ros/humble/setup.bash
source install/setup.bash

# 메인 시스템 실행
ros2 launch allex_ces_idle_interaction allex_idle_interaction.launch.py
```

또는 자동 실행 스크립트 사용:

```bash
cd ~/allex_ces_idle_interaction_gui
./run_spark.sh
```

## 빌드

```bash
cd ~/allex_ces_idle_interaction_gui
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 설정 파일

- `config/topics.json`: 토픽 설정
- `config/botsort.yaml`: BotSort 추적 알고리즘 설정
