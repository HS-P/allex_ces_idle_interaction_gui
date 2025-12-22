# 초기 추적 중 목 움직임 문제 분석

## 문제 현상
- 새로운 타겟을 인식한 후 초기 1.5초 동안 목(neck)이 전혀 움직이지 않음
- 1.5초 이후 일반 추적 모드로 전환되면 정상적으로 동작함
- 예: 0도에서 45도 타겟을 보면 1.5초 동안 가만히 있다가 1.5초 후 발작적으로 움직임

## 요구사항
- 초기 1.5초 동안 목이 초당 36도 (1.2도/프레임, 30Hz 기준)로 움직여야 함
- 초기 1.5초 동안 허리(waist)는 정지 상태 유지
- 초기 추적과 일반 추적의 차이는 **rate limit만 다르게** 설정하면 됨
  - 초기 추적: 초당 36도 (1.2도/프레임)
  - 일반 추적: 초당 2430도 (81도/프레임)

## 현재 코드 구조

### 1. 초기 추적 플래그 설정
```python
# track_id 변경 감지
if self.current_track_id != target_info.track_id:
    self.current_track_id = target_info.track_id
    self.tracking_start_time = current_time_check
    # 상태 초기화
    self.last_neck_target_yaw = None
    self.last_neck_delta_yaw = 0.0
    self.last_neck_delta_pitch = 0.0
    self.last_sent_neck_delta_yaw = 0.0
    self.last_sent_neck_delta_pitch = 0.0
    self.integral_yaw = 0.0
    self.integral_pitch = 0.0
    self.last_error_yaw = 0.0
    self.last_error_pitch = 0.0

# 초기 추적 여부 확인
is_initial_tracking = (
    self.tracking_start_time is not None and
    current_time_check - self.tracking_start_time < self.initial_tracking_duration  # 1.5초
)
```

### 2. 목 타겟 계산
```python
# 전체 시선각 계산
current_total_yaw = self.current_waist_yaw_rad + self.current_yaw_rad
raw_desired_total_yaw = current_total_yaw + relative_yaw_rad

# 목 타겟: raw_total_target_yaw - current_waist_yaw
neck_target_yaw = raw_desired_total_yaw - self.current_waist_yaw_rad
neck_target_yaw = max(self.yaw_min, min(self.yaw_max, neck_target_yaw))

# LOST -> TRACKING 전환 시 rate limit + 스무딩
if self.last_neck_target_yaw is not None:
    if is_initial_tracking:
        max_neck_target_delta = self.initial_tracking_max_delta  # 1.2도/프레임
        # 초기 추적: rate limit만 적용, 스무딩 없음
        neck_target_delta = neck_target_yaw - self.last_neck_target_yaw
        neck_target_delta = max(-max_neck_target_delta, min(max_neck_target_delta, neck_target_delta))
        neck_target_yaw = self.last_neck_target_yaw + neck_target_delta
    else:
        max_neck_target_delta = math.radians(12.0)  # 일반 추적: 12도/프레임
        smoothing_factor = 0.95  # 5% 스무딩
        neck_target_delta = neck_target_yaw - self.last_neck_target_yaw
        neck_target_delta = max(-max_neck_target_delta, min(max_neck_target_delta, neck_target_delta))
        neck_target_yaw = self.last_neck_target_yaw + smoothing_factor * neck_target_delta
else:
    # 초기화: 계산된 목 타겟으로 바로 설정
    self.last_neck_target_yaw = neck_target_yaw
```

### 3. 목 명령 전송 (_send_neck_command)
```python
def _send_neck_command(self, target_yaw_rad, target_pitch_rad, use_pid=True, use_searching_gain=False, is_initial_tracking=False):
    # PID 제어 또는 직접 계산
    if use_pid:
        delta_yaw_rad, delta_pitch_rad = self._pid_control(target_yaw_rad, target_pitch_rad, use_searching_gain=use_searching_gain)
    else:
        delta_yaw_rad = target_yaw_rad - self.current_yaw_rad
        delta_pitch_rad = target_pitch_rad - self.current_pitch_rad
    
    # Rate limit 적용: 초기 추적 중에는 초당 36도, 일반 추적에서는 81도/프레임
    if is_initial_tracking:
        max_delta_angle = self.initial_tracking_max_delta  # 초당 36도 (1.2도/프레임)
    else:
        max_delta_angle = math.radians(81.0)  # 일반 추적: 81도/프레임
    
    # 기본 rate limit
    delta_yaw_rad = max(-max_delta_angle, min(max_delta_angle, delta_yaw_rad))
    delta_pitch_rad = max(-max_delta_angle, min(max_delta_angle, delta_pitch_rad))
    
    # PID 제어 결과 스무딩
    if use_searching_gain:
        smoothing_alpha = 0.75
    else:
        smoothing_alpha = 0.9
    
    smoothed_delta_yaw = self.last_neck_delta_yaw + smoothing_alpha * (delta_yaw_rad - self.last_neck_delta_yaw)
    smoothed_delta_pitch = self.last_neck_delta_pitch + smoothing_alpha * (delta_pitch_rad - self.last_neck_delta_pitch)
    
    # 스무딩된 값도 rate limit 적용
    smoothed_delta_yaw = max(-max_delta_angle, min(max_delta_angle, smoothed_delta_yaw))
    smoothed_delta_pitch = max(-max_delta_angle, min(max_delta_angle, smoothed_delta_pitch))
    
    # 추가 스무딩
    delta_yaw_magnitude = abs(smoothed_delta_yaw - self.last_sent_neck_delta_yaw)
    delta_pitch_magnitude = abs(smoothed_delta_pitch - self.last_sent_neck_delta_pitch)
    
    small_change_threshold = math.radians(0.3)
    
    if delta_yaw_magnitude < small_change_threshold:
        final_smoothing_alpha = 0.7
    else:
        final_smoothing_alpha = 0.95
    
    if delta_pitch_magnitude < small_change_threshold:
        final_pitch_smoothing_alpha = 0.7
    else:
        final_pitch_smoothing_alpha = 0.95
    
    final_delta_yaw = self.last_sent_neck_delta_yaw + final_smoothing_alpha * (smoothed_delta_yaw - self.last_sent_neck_delta_yaw)
    final_delta_pitch = self.last_sent_neck_delta_pitch + final_pitch_smoothing_alpha * (smoothed_delta_pitch - self.last_sent_neck_delta_pitch)
    
    # Rate limit 재적용
    final_delta_yaw = max(-max_delta_angle, min(max_delta_angle, final_delta_yaw))
    final_delta_pitch = max(-max_delta_angle, min(max_delta_angle, final_delta_pitch))
    
    # 명령 전송
    msg = Float64MultiArray()
    msg.data = [float(final_delta_pitch), float(final_delta_yaw)]
    self.neck_publisher.publish(msg)
```

### 4. 허리 정지 처리
```python
def _waist_follow_target(self, desired_waist_yaw, searching_mode=False):
    current_time = time.monotonic()
    
    # 초기 추적 중에는 허리를 움직이지 않음 (얼굴만 움직임)
    if (not searching_mode and 
        self.tracking_start_time is not None and 
        current_time - self.tracking_start_time < self.initial_tracking_duration):
        # 초기 추적 중: 현재 허리 위치 유지
        if self.last_waist_command is None:
            self.last_waist_command = self.current_waist_yaw_rad
        return self.last_waist_command
    
    # 일반 허리 제어 로직...
```

## 문제 분석

### 가능한 원인들
1. **초기화 문제**: `last_neck_delta_yaw = 0.0`으로 초기화되어 스무딩 계산 시 문제 발생 가능
2. **PID 제어 문제**: 초기 추적 중 PID 출력이 0에 가까운 값이 나올 수 있음
3. **목 타겟 계산 문제**: `last_neck_target_yaw`가 None인 경우와 아닌 경우의 처리 차이
4. **스무딩 문제**: 여러 단계의 스무딩이 누적되어 초기 움직임이 억제될 수 있음
5. **Rate limit 문제**: `max_delta_angle`이 너무 작아서 움직임이 거의 없을 수 있음

### 확인 사항
- `is_initial_tracking` 플래그가 제대로 전달되는지 확인
- `tracking_start_time`이 제대로 설정되는지 확인
- PID 제어 출력값이 실제로 계산되는지 확인
- `last_neck_delta_yaw`가 0.0으로 초기화되어 스무딩에 영향을 주는지 확인
- 여러 단계의 rate limit 적용이 중복되어 문제를 일으키는지 확인

## 해결 방안

### 핵심 원칙
- 초기 추적 중에도 일반 추적과 **동일한 로직** 사용
- **오직 rate limit만 다르게** 설정 (초기: 1.2도/프레임, 일반: 81도/프레임)
- PID 제어, 스무딩 등은 일반 추적과 동일하게 적용

### 검증 포인트
1. `is_initial_tracking` 플래그가 `_send_neck_command`에 제대로 전달되는가?
2. 초기 추적 중 PID 제어 출력이 0이 아닌 값이 나오는가?
3. `last_neck_delta_yaw` 초기화가 문제를 일으키는가?
4. Rate limit 적용이 중복되어 문제를 일으키는가?

## 파일 위치
- `/home/dgx_allex_one/allex_ces_idle_interaction_gui/src/allex_ces_idle_interaction/allex_ces_idle_interaction/gaze_controller_neck_waist_node.py`
- 주요 함수:
  - `_update_control()`: TRACKING 상태 처리 (약 715줄~)
  - `_send_neck_command()`: 목 명령 전송 (약 611줄~)
  - `_waist_follow_target()`: 허리 제어 (약 505줄~)

## 파라미터
- `self.initial_tracking_duration = 1.5`  # 초
- `self.initial_tracking_max_delta = math.radians(1.2)`  # 초당 36도 (1.2도/프레임, 30Hz 기준)

