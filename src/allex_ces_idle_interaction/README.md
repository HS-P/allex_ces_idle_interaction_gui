# Allex CES Idle Interaction

## 버전 정보
- 현재 버전: v1.8.1
- 마지막 업데이트: 2025-01-22

## 개요
Allex 로봇의 Idle 상태 상호작용 시스템으로, 사람 추적, 인사 제스처, 악수 등과 같은 상호작용을 제공합니다.

## 해결해야 할 문제 (TODO)

### 1. 얼굴 추정 시 Steady State Error 및 진동 발생
- **문제 설명**: 얼굴 추정할 때 steady state error와 진동이 발생합니다. (문제 3번과 유사한 현상)
- **영향**: 로봇의 목 움직임이 불안정하고 부자연스러움
- **관련 파일**: 
  - `gaze_controller_neck_waist_node.py`
  - PID 게인 조정이 필요할 수 있음
- **예상 원인**:
  - PID 게인 불균형
  - 목 관절의 하드웨어 지연
  - 목표 각도와 실제 피드백 간의 오차 누적

### 2. 낮은 확률로 악수 진행 중 SEARCHING 상태로 전환
- **문제 설명**: HANDSHAKE 상태에서 루틴이 진행 중인데도 SEARCHING 상태로 넘어가는 경우가 발생합니다.
- **영향**: 악수 루틴이 완료되기 전에 다른 상태로 전환되어 부자연스러운 동작
- **관련 파일**:
  - `tracking_fsm_node.py` - HANDSHAKE 상태 처리 로직
  - `allex_idle_interaction_node.py` - 루틴 제어 로직
- **예상 원인**:
  - `/debug/routine` 토픽 피드백의 타이밍 문제
  - 루틴 상태 판단 로직의 경쟁 조건(race condition)
  - 루틴 시작과 상태 확인 사이의 지연 시간 부족

### 3. 타겟 위치 추종 부정확 (목 1.2도 부족)
- **문제 설명**: 타겟 위치를 추종할 때 목이 약 1.2도 정도 부족하게 추종하고, 나머지는 허리가 추종하는 느낌입니다. 이 1도 차이가 시각적으로 매우 문제가 됩니다.
- **영향**: 로봇이 사람을 정확히 응시하지 못함
- **관련 파일**:
  - `gaze_controller_neck_waist_node.py` - 목/허리 제어 로직
- **예상 원인**:
  - 목과 허리 간의 각도 분배 로직 문제
  - 목 관절의 각도 제한 또는 데드존(deadzone)
  - 목표 각도 계산 시 오프셋 누락
  - PID 제어의 적분 항 누적 오차

## 시스템 구조

### 주요 노드
- `tracking_fsm_node.py`: 추적 상태 머신 (FSM) 관리
- `gaze_controller_neck_waist_node.py`: 목/허리 제어
- `yolo_detection_node.py`: YOLO 기반 사람 감지
- `allex_idle_interaction_node.py`: 루틴 제어 및 상태 관리
- `idle_interaction_gui_node.py`: GUI 인터페이스

### 상태 (TrackingState)
- `IDLE`: 초기 상태, 대상 선택 대기
- `TRACKING`: 타겟 추적 중
- `LOST`: 타겟 놓침
- `SEARCHING`: 주변 탐색
- `HELLO`: 인사 제스처 실행 중
- `HANDSHAKE`: 악수 제스처 실행 중

## 실행 방법

```bash
# 빌드
cd /home/dgx_allex_one/allex_ces_idle_interaction_gui
colcon build --packages-select allex_ces_idle_interaction

# 소스
source install/setup.bash

# 실행
ros2 launch allex_ces_idle_interaction allex_idle_interaction.launch.py
```

## 토픽 정보
주요 토픽은 `config/topics.json`에 정의되어 있습니다.

### 주요 토픽
- `/debug/routine`: 루틴 상태 피드백 (std_msgs/String, JSON)
- `/allex_camera/tracking_data`: 추적 데이터 (std_msgs/String, JSON)
- `/robot_inbound/theOne_neck/joint_command`: 목 관절 명령
- `/robot_inbound/theOne_waist/joint_command`: 허리 관절 명령

## 설정 파일
- `config/topics.json`: 토픽 설정
- `config/botsort.yaml`: BotSort 추적 알고리즘 설정

## 변경 이력
- v1.8.1: `/debug/routine` 토픽 기반 루틴 상태 확인으로 변경
- v1.8.0: HANDSHAKE 상태 추가, 키보드 제어 추가

