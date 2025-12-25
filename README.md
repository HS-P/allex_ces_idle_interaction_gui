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

### 사전 준비

**⚠️ 중요: ROS 2 Humble 환경이 올바르게 설정되어 있어야 합니다!**

```bash
# ROS 2 Humble 환경 설정 (Thor에서 실행)
source /opt/ros/humble/setup.bash

# 현재 ROS 2 버전 확인
echo $ROS_DISTRO  # "humble"이 출력되어야 함
```

### 빌드

```bash
cd ~/allex_ces_idle_interaction_gui

# ROS 2 Humble 환경 확인 (반드시 실행!)
source /opt/ros/humble/setup.bash

# 빌드 실행
colcon build --symlink-install

# 설치 후 환경 설정
source install/setup.bash
```

### 빌드 오류 해결

**⚠️ 중요: 모든 빌드 명령은 DGX Thor에서 실행해야 합니다!**

#### `pkgutil.ImpImporter` 오류 (Python 3.12 환경)
만약 `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` 오류가 발생하면:

**해결 방법 1: setuptools 업그레이드 (Python 3.12 권장)**
```bash
# Thor에서 실행
pip install --upgrade setuptools
colcon build --symlink-install
```

**해결 방법 2: setuptools 특정 버전 설치 (Python 3.10 이하 환경)**
```bash
# Thor에서 실행 (Python 3.10 이하인 경우에만)
pip install setuptools==58.1.0
colcon build --symlink-install
```

#### `--editable` 옵션 오류
만약 `error: option --editable not recognized` 오류가 발생하면:

**해결 방법: 환경 변수 정리 후 재빌드**
```bash
# Thor에서 실행
# 환경 변수 정리
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH

# 빌드 캐시 정리
rm -rf build/ install/ log/

# 재빌드
colcon build --symlink-install
```

#### `rosidl_typesupport_c` 찾을 수 없음 오류
만약 `No 'rosidl_typesupport_c' found` 오류가 발생하면:

**원인: 잘못된 ROS 2 버전이 소스되어 있거나 ROS 2가 제대로 설치되지 않음**

**해결 방법:**
```bash
# Thor에서 실행

# 1. ROS 2 Humble 환경을 명시적으로 소스
source /opt/ros/humble/setup.bash

# 2. ROS_DISTRO 환경 변수 확인
echo $ROS_DISTRO  # "humble"이어야 함

# 3. 만약 "jazzy" 또는 다른 값이면, .bashrc 또는 .bash_profile 확인
# 잘못된 ROS 2 소스 명령이 있는지 확인하고 제거

# 4. rosidl 관련 패키지 설치 확인 (필요시)
sudo apt update
sudo apt install ros-humble-rosidl-typesupport-c

# 5. 재빌드
cd ~/allex_ces_idle_interaction_gui
colcon build --symlink-install
```

#### 병렬 작업자 수 제한
메모리 부족 등의 문제가 발생하면:
```bash
# Thor에서 실행
colcon build --symlink-install --parallel-workers 2
```

#### 경로 경고 해결
`AMENT_PREFIX_PATH` 또는 `CMAKE_PREFIX_PATH`에 존재하지 않는 경로가 포함되어 있으면:
```bash
# 환경 변수 확인
echo $AMENT_PREFIX_PATH
echo $CMAKE_PREFIX_PATH

# 존재하지 않는 경로 제거 (예시)
export AMENT_PREFIX_PATH=$(echo $AMENT_PREFIX_PATH | tr ':' '\n' | grep -v '/home/mars/allex_ces_idle_interaction' | tr '\n' ':' | sed 's/:$//')
export CMAKE_PREFIX_PATH=$(echo $CMAKE_PREFIX_PATH | tr ':' '\n' | grep -v '/home/mars/allex_ces_idle_interaction' | tr '\n' ':' | sed 's/:$//')
```

**참고:** 경로 경고는 빌드를 막지 않지만, 정리하면 더 깔끔한 빌드가 가능합니다.

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

