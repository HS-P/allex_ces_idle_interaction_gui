# ALLEX CES Idle Interaction System

ALLEX 로봇의 Idle Interaction 시스템입니다. 사람 추적, 시선 제어, 제스처 인식 등의 기능을 제공합니다.

## 시스템 구성

이 시스템은 두 대의 로봇에서 실행됩니다:

- **DGX SPARK**: 메인 추적 및 제어 시스템 실행
- **DGX Thor**: 카메라 스트리밍 실행

## 사전 요구사항

- ROS 2 (Humble 또는 Iron 권장)
- Python 3.12+
- CUDA 지원 GPU (YOLO 추론용)
- Orbbec Femto Bolt 카메라 (Thor에서 사용)

## 빌드 및 설치

```bash
cd ~/allex_ces_idle_interaction_gui
colcon build --symlink-install
source install/setup.bash
```

## 실행 방법

### 방법 1: 수동 실행

#### DGX Thor (카메라 실행)
```bash
ros2 launch orb_ecc_camera femto_bolt.launch.py
```

#### DGX SPARK (메인 시스템 실행)
```bash
ros2 launch allex_ces_idle_interaction allex_idle_interaction.launch.py
```

### 방법 2: 자동 실행 스크립트 사용

#### DGX Thor에서 실행
```bash
cd ~/allex_ces_idle_interaction_gui
./run_thor.sh
```

이 스크립트는 다음을 자동으로 실행합니다:
- Orbbec Femto Bolt 카메라 launch 파일
- GUI 애플리케이션

각 프로세스는 별도의 백그라운드 프로세스로 실행되며, 로그는 `logs/` 디렉토리에 저장됩니다.

#### DGX SPARK에서 실행
```bash
cd ~/allex_ces_idle_interaction_gui
./run_spark.sh
```

## 시스템 아키텍처

### 주요 노드

1. **yolo_detection_node**: YOLO 기반 사람 감지
2. **tracking_fsm_node**: 추적 상태 머신 관리
3. **gaze_controller_neck_waist_node**: 목과 허리 제어
4. **allex_idle_interaction_node**: 전체 상호작용 관리

### 상태 머신

- **IDLE**: 영자세 상태
- **WAITING**: 타겟 찾기
- **TRACKING**: 추적 중
- **LOST**: 추적 대상 놓침
- **SEARCHING**: 주변 탐색
- **HELLO**: 인사 제스처
- **HANDSHAKE**: 악수 제스처

## 주요 기능

### 초기 추적 (Initial Tracking)
- 타겟 감지 후 처음 0.85초 동안 목만 움직이며 추적
- 허리는 고정 상태 유지
- 부드러운 움직임을 위한 최적화된 PID 게인 사용

### 일반 추적 (Normal Tracking)
- 목과 허리 협조 제어
- 목이 먼저 빠르게 추종하고, 허리가 천천히 따라옴
- 12% 속도 증가 적용

### LOST 상태 감속
- 타겟을 놓쳤을 때 현재 위치에서 부드럽게 멈춤
- 1초 동안 선형 감속 적용

### SEARCHING 모드
- 목 pitch 26도로 기울여 주변 탐색
- 목 속도 감소 (부드러운 움직임)

## 토픽 구조

### 주요 토픽
- `/camera/color/image_raw/compressed`: 카메라 이미지 (CompressedImage)
- `/allex_camera/detections`: YOLO 감지 결과
- `/allex_camera/tracking_result`: 추적 결과
- `/allex_camera/neck_angle`: 목 각도 명령

## 문제 해결

### 카메라 연결 문제
- Thor에서 카메라가 인식되지 않으면 USB 연결 확인
- `lsusb` 명령으로 Orbbec 장치 확인

### 네트워크 통신 문제
- SPARK와 Thor 간 ROS 2 통신 확인
- `ROS_DOMAIN_ID` 환경 변수 확인 (기본값: 0)

### 로그 확인
- Thor: `logs/thor_camera.log`, `logs/thor_gui.log`
- SPARK: `logs/spark_main.log`

## 개발자 정보

- 초기 추적 시간: 0.85초
- LOST 감속 시간: 1.0초
- SEARCHING 목 pitch: 26도

