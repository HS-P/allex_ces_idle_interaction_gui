# 현재 개발 상태 및 주요 변경사항 정리

> **작성일**: 2025-12-31  
> **목적**: CPP 파일과 연동하기 전 시대로 되돌리기 전, 현재까지의 주요 개선사항을 상세히 기록  
> **참고**: 마지막 develop branch의 commit을 참고하여 작성

---

## 📋 목차

1. [시스템 아키텍처 개요](#시스템-아키텍처-개요)
2. [주요 노드별 상세 기능](#주요-노드별-상세-기능)
3. [Neck-Waist Movement 구현](#neck-waist-movement-구현)
4. [Joystick Control 구현](#joystick-control-구현)
5. [GUI 개선사항](#gui-개선사항)
6. [Tracking FSM 개선사항](#tracking-fsm-개선사항)
7. [알려진 이슈 및 To-Do List](#알려진-이슈-및-to-do-list)

---

## 시스템 아키텍처 개요

### 노드 구조

```
allex_ces_idle_interaction_gui/
├── allex_idle_interaction_node.py      # 메인 제어 노드 (루틴 제어, 상태 관리)
├── tracking_fsm_node.py                 # 추적 FSM 노드 (상태 머신, 타겟 선택)
├── gaze_controller_neck_waist_node.py  # 목/허리 제어 노드 (Neck-Waist Movement)
├── joystick_control_node.py            # 조이스틱 제어 노드 (키보드 입력 처리)
├── idle_interaction_gui_node.py        # GUI 노드 (PySide6 기반)
└── yolo_detection_node.py               # YOLO 검출 노드
```

### 토픽 구조

**주요 Publisher:**
- `/allex_camera/tracking_result` - 추적 결과 (tracking_fsm_node → allex_idle_interaction_node, GUI)
- `/allex_camera/tracking_data` - GUI용 추적 데이터 (allex_idle_interaction_node → GUI)
- `/allex_camera/neck_angle` - 목 각도 정보 (gaze_controller → tracking_fsm_node)
- `/allex_camera/manual_control` - 수동 제어 명령 (GUI, Joystick → tracking_fsm_node)
- `/robot_inbound/theOne_neck/joint_command` - 목 명령 (gaze_controller → 로봇)
- `/robot_inbound/theOne_waist/joint_command` - 허리 명령 (gaze_controller → 로봇)

**주요 Subscriber:**
- `/allex_camera/detections` - YOLO 검출 결과
- `/camera/color/image_raw/compressed` - 카메라 이미지
- `/camera/depth/image_raw` - Depth 이미지
- `/robot_outbound_data/theOne_neck/joint_positions_deg` - 목 현재 위치
- `/robot_outbound_data/theOne_waist/joint_positions_deg` - 허리 현재 위치
- `/debug/routine` - 루틴 상태 피드백

---

## 주요 노드별 상세 기능

### 1. `allex_idle_interaction_node.py` - 메인 제어 노드

**주요 클래스:**
- `RoutineController`: 루틴 제어 클래스
- `AllexIdleInteractionNode`: 메인 노드 클래스

**주요 기능:**
- GUI 명령 처리 (`/allex_camera/manual_control` 구독)
- 상태 관리 및 루틴 제어
- 추적 결과를 받아서 처리 및 발행
- `/debug/routine` 토픽 구독하여 루틴 상태 피드백 확인
- Handshake/Hello 완료 확인 (부모 노드 status 기반)

**루틴 제어 방식:**
- PAUSE -> RESET 방식으로 할 것


---

### 2. `tracking_fsm_node.py` - 추적 FSM 노드

**주요 클래스:**
- `TrackingState`: 추적 상태 Enum
  - `IDLE`: 영자세로 돌아가기 (7.5초 후 WAITING 전이)
  - `WAITING`: 타겟 찾기
  - `TRACKING`: 추적 중
  - `LOST`: 추적 대상 놓침
  - `SEARCHING`: 주변 두리번대기
  - `HELLO`: 인사 제스처
  - `HANDSHAKE`: 악수 제스처

**주요 기능:**
- Detection 결과를 받아서 FSM 처리
- Manual/Auto 모드 지원
- Depth 기반 HELLO/HANDSHAKE 분기 판단 (1.5m 기준)
- HELLO 완료 ID 추적 (중복 HELLO 방지)
- ROI 영역 기반 타겟 선택 (좌우 15% 마진)

**타겟 선택 로직:**
- `_find_closest_person()`: 프레임 중심에 가장 가까운 사람 찾기
- 최소 지속 시간 (`min_target_duration = 1.4초`) 이상인 객체만 후보
- SEARCHING 상태에서는 8초 쿨다운 후 타겟 선택 시작
- HELLO 완료 ID는 자동으로 제외

**상태 전이 로직:**
- IDLE → WAITING: 타겟 발견 후 7.5초 경과
- WAITING → TRACKING: 타겟 자동 선택
- TRACKING → LOST: 타겟 손실 (최대 4초 대기)
- LOST → SEARCHING: 4초 경과 후 자동 전이
- TRACKING → HELLO: 위치 안정성 체크 완료 후 Depth 기반 분기
- HELLO/HANDSHAKE → SEARCHING: 루틴 완료 후 자동 전이

---

### 3. `gaze_controller_neck_waist_node.py` - 목/허리 제어 노드

**주요 클래스:**
- `GazeControllerNode`: 목/허리 제어 노드

**핵심 기능: Neck-Waist Movement 구현**

#### 3.1 목(Neck) 제어

**PID 제어:**
- 일반 추적용 PID 파라미터:
  - `kp_yaw = 1.05`, `ki_yaw = 0.01`
  - `kp_pitch = 1.15`, `ki_pitch = 0.1344`
- SEARCHING 상태용 게인 (속도 감소):
  - `kp_yaw_searching = 0.30`, `kp_pitch_searching = 0.30`

**스무딩:**
- PID 제어 결과 스무딩: `smoothing_alpha = 0.3`
- 전체 시선각 스무딩: `total_yaw_smoothing_alpha = 0.6`
- 목 목표 각도 스무딩: `neck_target_alpha = 0.85`

**각도 제한:**
- Yaw: `-65° ~ +65°` (SEARCHING용)
- Pitch: `-5° ~ 215°`

**상태별 동작:**
- **IDLE**: 영자세 복귀 (7초 동안 부드럽게 복귀)
- **WAITING**: 현재 위치 유지
- **TRACKING**: 타겟 추종 (PID 제어)
- **LOST**: 지수 감쇠로 목표 위치로 이동 (`tau_neck_lost = 1.5초`)
- **SEARCHING**: 좌우 스캔 (Phase 0: 우측 +40도, Phase 1: 좌측 -40도)
- **HELLO/HANDSHAKE**: 현재 위치 유지

#### 3.2 허리(Waist) 제어

**추종 제어:**
- TRACKING 상태: 목 전체 시선각을 따라감 (`kp_waist_tracking = 1.8`)
- Exponential smoothing: `tau_waist = 0.29초` (TRACKING), `tau_waist_searching = 1.5초` (SEARCHING)
- 최대 변화량: `max_delta_waist_tracking = 0.8도/프레임`

**상시 Pitch 움직임 (Breathing):**
- 모든 상태에서 sin 파형으로 Pitch 움직임
- 진폭: `7도`, 오프셋: `-4도`, 주기: `7초`
- `_get_waist_breathe_pitch()` 함수로 계산

**SEARCHING 상태:**
- 목과 허리 독립 제어
- Phase 기반 좌우 스캔
- 초반 1.5초 동안 느린 속도 (`waist_initial_smoothing_factor = 0.6`)
- Phase 타임아웃: `15초`

**각도 제한:**
- Waist Yaw: `-85° ~ +85°`

#### 3.3 HELLO 전환 조건

**위치 안정성 체크:**
- 현재 위치에서 ±2.5도 이내로 1.75초 유지
- 조건 만족 시 `hello_transition_ready` 요청 발송
- Depth 기반으로 HELLO 또는 HANDSHAKE 결정 (1.5m 기준)

---

### 4. `joystick_control_node.py` - 조이스틱 제어 노드

**주요 클래스:**
- `JoystickControlNode`: 조이스틱 제어 노드

**키보드 입력 처리:**
- `pynput` 라이브러리 사용
- j 키와 함께 눌러야 작동하는 안전장치
- 키 입력 디바운싱 (0.3초)

**지원 명령:**
- `j+c`: Auto Run
- `j+d`: Manual Run
- `j+f`: Stop
- `j+k`: 타겟 선택 (왼쪽으로 이동)
- `j+m`: 타겟 선택 (오른쪽으로 이동)
- `j+i`: Handshake State
- `j+h`: Idle State
- `j+g`: Hello State

**타겟 선택 로직:**
- 초기(타겟이 없을 때): 화면 중심 기준으로 좌우 분리 후 거리 순 정렬
- 타겟이 있을 때: x 좌표 기준으로 좌측부터 정렬
- `tracked_objects_sorted` 리스트를 사용하여 순차적 선택

**현재 구현 상태:**
- 타겟이 없을 때: 중심 기준 좌우 분리 후 거리 순 정렬 ✅
- 타겟이 있을 때: x 좌표 순서대로 정렬 ✅
- **이슈**: j+k가 좌측 사람을 바라보지 않음 (키 인식 문제일 수도 있음) ⚠️

---

### 5. `idle_interaction_gui_node.py` - GUI 노드

**주요 클래스:**
- `GuiNode`: GUI 노드 (PySide6 기반)
- `TargetButton`: 타겟 변경 버튼

**주요 기능:**
- PySide6 기반 GUI (v1.10.2)
- 실시간 상태 표시 (State, FPS, 처리 시간 등)
- 목/허리 각도 정보 표시
- 타겟 변경 버튼 (최대 10개)
- Manual/Auto 모드 전환
- 파라미터 제어 패널

**타겟 버튼 정렬 로직:**
- 화면 중심 기준으로 좌우 분리
- 각 방향에서 중심에 가까운 순으로 정렬
- 좌측 객체 먼저, 그 다음 우측 객체 순서

**현재 구현 상태:**
- 화면 중심 기준 좌우 분리 ✅
- 거리 순 정렬 ✅
- **이슈**: 타겟이 정해진 이후 타겟 중심 기준으로 좌우 선택이 안 됨 ⚠️

---

## Neck-Waist Movement 구현

### 구현 배경
기존에는 목만 제어했으나, 허리도 함께 제어하여 더 자연스러운 움직임 구현

### 주요 특징

1. **목과 허리 독립 제어**
   - 목: 빠른 반응 (PID 제어)
   - 허리: 느린 추종 (Exponential smoothing)

2. **상태별 동작**
   - **TRACKING**: 허리가 목 전체 시선각을 따라감
   - **SEARCHING**: 목과 허리 독립적으로 좌우 스캔
   - **LOST**: 지수 감쇠로 목표 위치로 이동
   - **IDLE**: 영자세 복귀 (목과 허리 모두)

3. **상시 Breathing 움직임**
   - 허리 Pitch가 sin 파형으로 움직임
   - 모든 상태에서 적용되어 생동감 부여

4. **스무딩 및 제한**
   - 다단계 스무딩 (PID 결과 → 목표 각도 → 전체 시선각)
   - Rate limiting으로 부드러운 움직임 보장

---

## Joystick Control 구현

### 구현 배경
GUI 없이도 키보드로 시스템을 제어할 수 있도록 구현

### 주요 특징

1. **안전장치**
   - j 키와 함께 눌러야 작동 (실수 방지)
   - 키 입력 디바운싱 (0.3초)

2. **타겟 선택**
   - 초기: 화면 중심 기준 좌우 분리 후 거리 순
   - 타겟 있음: x 좌표 순서대로

3. **상태 제어**
   - RUN/STOP 제어
   - Manual/Auto 모드 전환
   - 상태 직접 변경 (IDLE, HELLO, HANDSHAKE)

---

## GUI 개선사항

### 주요 개선사항

1. **PySide6 기반 GUI (v1.10.2)**
   - Qt 플러그인 경로 자동 설정
   - 멀티스레드 안전한 시그널/슬롯 사용

2. **실시간 정보 표시**
   - State, FPS, 처리 시간
   - 목/허리 각도 (현재/목표)
   - 추적 객체 수

3. **타겟 변경 버튼**
   - 최대 10개 타겟 버튼
   - 화면 중심 기준 좌우 분리 후 거리 순 정렬
   - 현재 타겟 하이라이트

4. **파라미터 제어 패널**
   - PID 파라미터 실시간 조정
   - 스무딩 파라미터 조정
   - 허리 파라미터 조정

5. **외부 명령 수신 지원**
   - 조이스틱 명령 수신 시 GUI 자동 업데이트
   - RUN/STOP 상태 동기화
   - Manual/Auto 모드 동기화

---

## Tracking FSM 개선사항

### 주요 개선사항

1. **상태 머신 개선**
   - IDLE 상태: 타겟 발견 후 7.5초 후 WAITING 자동 전이
   - SEARCHING 상태: 8초 쿨다운 후 타겟 선택 시작
   - HELLO 완료 ID 추적 (중복 방지)

2. **타겟 선택 로직**
   - 최소 지속 시간 체크 (1.4초)
   - ROI 영역 기반 필터링 (좌우 15% 마진)
   - 마지막 타겟 위치 우선 고려 (같은 사람 재선택)

3. **Depth 기반 분기**
   - Depth 이미지에서 타겟의 거리 추출
   - 1.5m 기준으로 HELLO/HANDSHAKE 결정

4. **Manual 모드 지원**
   - 상태 수동 설정
   - 타겟 수동 선택
   - 자동 전이 비활성화

---

## 알려진 이슈 및 To-Do List

### 🔴 긴급 수정 필요

#### 1. Joystick j+k 키 인식 문제
**문제**: j+k를 눌러도 좌측 사람을 바라보지 않음

**가능한 원인:**
- 키 인식 문제 (j 키와 k 키 동시 입력 감지 실패)
- 타겟 선택 로직 문제 (인덱스 계산 오류)

**확인 필요:**
- `_handle_target_left()` 함수의 키 인식 로직
- `tracked_objects_sorted` 리스트의 정렬 순서
- 로그에서 키 입력이 제대로 감지되는지 확인

**예상 수정 방향:**
```python
# joystick_control_node.py의 _handle_target_left() 함수 확인
# 키 인식 로직 점검 및 디버깅 로그 추가
```

#### 2. 타겟 선택 로직 개선 필요

**현재 문제:**
- "처음 타겟"은 중심 기준으로 가장 가까운 좌측/우측 선택 ✅ (구현됨)
- "타겟이 정해진 이후"는 타겟 중심 기준으로 가장 가까운 좌측/우측을 선택해야 함 ❌ (미구현)
- 현재는 우측 사람을 누르고 좌측 버튼을 누르면 "가장 먼 좌측"을 찾음 (픽셀 중심점 기준)

**요구사항:**
1. **처음 타겟 선택:**
   - 화면 중심 기준으로 가장 가까운 좌측/우측 선택 ✅

2. **타겟이 정해진 이후:**
   - 현재 타겟의 centroid를 기준으로
   - 좌측 버튼: 현재 타겟보다 좌측에 있는 사람 중 가장 가까운 사람
   - 우측 버튼: 현재 타겟보다 우측에 있는 사람 중 가장 가까운 사람
   - 순차적으로 한 명씩 선택 가능해야 함

**현재 구현 (GUI):**
```python
# idle_interaction_gui_node.py의 _update_target_buttons()
# 항상 화면 중심 기준으로 정렬 (타겟이 있어도)
```

**수정 필요:**
```python
# 타겟이 있을 때는 현재 타겟의 centroid를 기준으로 좌우 분리
# 각 방향에서 타겟에 가장 가까운 순으로 정렬
```

**수정 필요 (Joystick):**
```python
# joystick_control_node.py의 _tracking_result_callback()
# 타겟이 있을 때는 현재 타겟 중심 기준으로 정렬
```

### 🟡 개선 권장

#### 3. 타겟 버튼 정렬 로직 개선
- 타겟이 정해진 이후 타겟 중심 기준으로 좌우 분리
- 각 방향에서 타겟에 가장 가까운 순으로 정렬

#### 4. 로그 개선
- Joystick 키 입력 감지 로그 추가
- 타겟 선택 시 상세 로그 출력

---

## 코드 구조 상세

### 주요 함수 및 메서드

#### `allex_idle_interaction_node.py`

**RoutineController 클래스:**
- `publish_command()`: 루틴 명령 발행
- `send_external_command()`: External Topic으로 서브 루틴 명령 전송
- `transition_to_routine()`: 루틴 전환 (Reset 후 새 루틴 시작)
- `start_breathing()`: 숨쉬기 루틴 시작
- `stop_routine()`: 현재 루틴 Reset
- `wait_for_idle()`: Idle 상태 대기
- `is_reset_complete()`: RESET 완료 여부 확인

**AllexIdleInteractionNode 클래스:**
- `_routine_status_callback()`: 루틴 상태 피드백 처리
- `_check_handshake_markers()`: Handshake 마커 확인
- `_find_children_range()`: 자식 노드 범위 찾기
- `_check_all_descendants_completed()`: 모든 자식 노드 완료 확인
- `_handle_state_change()`: 상태 변경 시 루틴 전환 처리
- `_manual_control_callback()`: Manual 제어 명령 처리

#### `tracking_fsm_node.py`

**TrackingFSMNode 클래스:**
- `_process_fsm()`: FSM 처리 메인 로직
- `_find_closest_person()`: 가장 가까운 사람 찾기
- `set_state()`: 상태 수동 설정
- `set_target()`: 타겟 수동 설정
- `_extract_depth_from_detection()`: Depth 값 추출
- `_state_request_callback()`: 상태 변경 요청 처리 (HELLO/HANDSHAKE 전환)

#### `gaze_controller_neck_waist_node.py`

**GazeControllerNode 클래스:**
- `_update_control()`: 제어 업데이트 메인 로직
- `_pid_control()`: PID 제어
- `_send_neck_command()`: 목 명령 전송
- `_send_waist_command()`: 허리 명령 전송
- `_waist_follow_total()`: 허리 전체 시선각 추종
- `_searching_behavior()`: SEARCHING 상태 동작
- `_check_hello_transition()`: HELLO 전환 조건 체크

#### `joystick_control_node.py`

**JoystickControlNode 클래스:**
- `_start_keyboard_listener()`: 키보드 리스너 시작
- `_handle_key_command()`: 키 명령 처리
- `_handle_target_left()`: 타겟 왼쪽으로 이동
- `_handle_target_right()`: 타겟 오른쪽으로 이동
- `_tracking_result_callback()`: 추적 결과 업데이트 및 정렬

#### `idle_interaction_gui_node.py`

**GuiNode 클래스:**
- `_update_target_buttons()`: 타겟 버튼 업데이트
- `_on_target_button_clicked()`: 타겟 버튼 클릭 처리
- `_camera_data_callback()`: 카메라 데이터 콜백
- `update_info()`: 정보 업데이트 (주기적 호출)

---

## 주요 파라미터 값

### 목(Neck) 제어 파라미터

```python
# PID 제어 (일반 추적)
kp_yaw = 1.05
ki_yaw = 0.01
kp_pitch = 1.15
ki_pitch = 0.1344

# PID 제어 (SEARCHING)
kp_yaw_searching = 0.30
kp_pitch_searching = 0.30

# 스무딩
smoothing_alpha = 0.3
total_yaw_smoothing_alpha = 0.6
neck_target_alpha = 0.85

# 각도 제한
yaw_min = -65°, yaw_max = +65°
pitch_min = -5°, pitch_max = 215°
```

### 허리(Waist) 제어 파라미터

```python
# 추종 제어
kp_waist_tracking = 1.8
tau_waist = 0.29초 (TRACKING)
tau_waist_searching = 1.5초 (SEARCHING)
max_delta_waist_tracking = 0.8도/프레임

# Breathing
waist_breathe_amplitude = 7도
waist_breathe_offset = -4도
waist_breathe_period = 7초

# 각도 제한
waist_yaw_min = -85°, waist_yaw_max = +85°
```

### Tracking FSM 파라미터

```python
# 타겟 선택
min_target_duration = 1.4초
tracking_roi_left_margin = 0.15 (15%)
tracking_roi_right_margin = 0.15 (15%)

# 상태 전이
idle_duration = 7.5초 (IDLE → WAITING)
max_lost_frames = 120 (약 4초, LOST → SEARCHING)
searching_cooldown_duration = 8.0초

# HELLO 전환
hello_position_threshold_deg = 2.5도
hello_stable_duration = 1.75초
```

---

## 파일별 주요 변경사항 요약

### `allex_idle_interaction_node.py`
- ✅ RoutineController 클래스 추가 (루틴 제어 로직 분리)
- ✅ External Topic 기반 루틴 제어
- ✅ Handshake/Hello 완료 확인 (부모 노드 status 기반)
- ✅ `/debug/routine` 토픽 구독하여 루틴 상태 피드백 확인

### `tracking_fsm_node.py`
- ✅ Depth 기반 HELLO/HANDSHAKE 분기 판단
- ✅ HELLO 완료 ID 추적 (중복 방지)
- ✅ ROI 영역 기반 타겟 선택
- ✅ SEARCHING 상태 쿨다운 (8초)
- ✅ IDLE 상태 자동 전이 (7.5초)

### `gaze_controller_neck_waist_node.py`
- ✅ **Neck-Waist Movement 구현** (핵심 기능)
- ✅ 허리 추종 제어 (목 전체 시선각 추종)
- ✅ 상시 Breathing 움직임 (허리 Pitch sin 파형)
- ✅ SEARCHING 상태 목/허리 독립 제어
- ✅ LOST 상태 지수 감쇠
- ✅ HELLO 전환 조건 체크 (위치 안정성)

### `joystick_control_node.py`
- ✅ **Joystick Control 구현** (핵심 기능)
- ✅ j 키 기반 안전장치
- ✅ 키 입력 디바운싱
- ✅ 타겟 선택 (좌우 이동)
- ✅ 상태 직접 제어

### `idle_interaction_gui_node.py`
- ✅ PySide6 기반 GUI (v1.10.2)
- ✅ 화면 중심 기준 타겟 버튼 정렬
- ✅ 목/허리 각도 정보 표시
- ✅ 파라미터 제어 패널
- ✅ 외부 명령 수신 지원

---

## To-Do List (우선순위 순)

### 🔴 긴급 (수정 필요)

1. **Joystick j+k 키 인식 문제 해결**
   - 키 입력 감지 로직 점검
   - 디버깅 로그 추가
   - 좌측 사람을 제대로 바라보는지 확인

2. **타겟 선택 로직 개선**
   - 타겟이 정해진 이후 타겟 중심 기준으로 좌우 분리
   - 각 방향에서 타겟에 가장 가까운 순으로 정렬
   - 순차적으로 한 명씩 선택 가능하도록 수정

### 🟡 개선 권장

3. **GUI 타겟 버튼 정렬 개선**
   - 타겟이 있을 때 타겟 중심 기준으로 정렬

4. **로깅 개선**
   - Joystick 키 입력 상세 로그
   - 타겟 선택 시 상세 로그

---

## 참고사항

### Git 커밋 히스토리
- 최신 커밋: `3f25b27 a`
- 주요 커밋: `d62668f 일단 완성`, `cb4811e Tested`
- 안정화 버전: `49e4cbf 안정화, 잘 작동 버전`

### 되돌릴 커밋
CPP 파일과 연동하기 전 시대로 되돌릴 예정이므로, 위의 모든 개선사항을 MD 파일로 기록하여 추후 참고 가능하도록 함.

### 주요 개선사항 요약
1. **Neck-Waist Movement**: 목과 허리 독립 제어, 자연스러운 움직임
2. **Joystick Control**: 키보드로 시스템 제어 가능
3. **GUI 개선**: PySide6 기반, 실시간 정보 표시
4. **Tracking FSM 개선**: Depth 기반 분기, HELLO 완료 ID 추적
5. **루틴 제어**: External Topic 기반, 부모 노드 status 확인

---

## 결론

현재 코드베이스는 CPP 파일과 연동하기 전까지 많은 개선이 이루어졌습니다. 특히 **Neck-Waist Movement**와 **Joystick Control**은 핵심 기능으로, 되돌린 후에도 빠르게 재구현할 수 있도록 상세히 기록했습니다.

**주의사항**: 
- Joystick j+k 키 인식 문제는 수정 필요
- 타겟 선택 로직은 타겟 중심 기준으로 개선 필요

