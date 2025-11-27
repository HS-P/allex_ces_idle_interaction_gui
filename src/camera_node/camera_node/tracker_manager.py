#!/usr/bin/env python3
"""
Ultralytics YOLO + BoT-SORT(+ReID) 기반 TrackerManager
"""
from typing import List, Optional, Dict
from collections import namedtuple
from pathlib import Path
from enum import Enum
import time
import math

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from huggingface_hub import hf_hub_download

TrackedObject = namedtuple('TrackedObject', [
    'track_id', 'bbox', 'centroid', 'state', 'confidence', 'age'
])

# 타겟 정보 출력 구조
TargetInfo = namedtuple('TargetInfo', [
    'point',      # 타겟 중심점 (x, y) 또는 None
    'state',      # 현재 추적 상태 (TrackingState)
    'track_id',   # 타겟 track_id 또는 None
])

PERSON_CLASS_ID = 0  # COCO 사람 클래스 ID

class TrackingState(Enum):
    """추적 상태"""
    IDLE = "idle"           # 초기 대상 선택
    TRACKING = "tracking"   # 추적 중
    LOST = "lost"          # 추적 대상 놓침 (잠시 대기)
    SEARCHING = "searching" # 주변 두리번대기 (대상 선택)
    WAIST_FOLLOWER = "waist_follower"   # 허리 따라가기 (0도 유지)
    INTERACTION = "interaction" # 인터렉션

class TrackerManager:
    """Ultralytics 내장 Bot-SORT TrackerManager"""
    def __init__(
        self,
        model_path: Optional[str] = None,
        tracker_cfg: Optional[str] = None,
        conf_threshold: float = 0.7,
    ):
        self.conf_threshold = conf_threshold

        model_path = "yolo11n.pt"
        face_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")

        self.model_path = str(model_path)
        
        # GPU 디바이스 설정 (PyTorch가 지원하는 경우)
        device = 'cpu'
        if torch.cuda.is_available():
            # CUDA capability 확인
            try:
                device_props = torch.cuda.get_device_properties(0)
                compute_cap = f"{device_props.major}{device_props.minor}"
                print(f"GPU Compute Capability: {device_props.major}.{device_props.minor} (sm_{compute_cap})")
                
                # GPU 사용 시도
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                device = 0  # GPU 0번 사용
                print(f"✓ GPU 사용 설정: {torch.cuda.get_device_name(0)}")
            except RuntimeError as e:
                print(f"⚠️  GPU 사용 불가: {e}")
                print("   PyTorch nightly 빌드 설치 필요:")
                print("   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124")
                raise RuntimeError("GPU를 사용할 수 없습니다. PyTorch nightly 빌드를 설치하세요.")
        
        # YOLO 모델 초기화 (명시적으로 device 지정)
        self.yolo_model = YOLO(self.model_path)
        self.face_model = YOLO(face_model_path)
        
        # GPU로 모델 이동
        if device != 'cpu':
            self.yolo_model.to(device)
            self.face_model.to(device)

        # botsort.yaml 파일 경로 찾기 (소스 코드와 설치된 환경 모두 지원)
        # 소스 코드: src/camera_node/config/botsort.yaml
        # 설치된 환경: install/camera_node/share/camera_node/config/botsort.yaml
        possible_paths = [
            Path(__file__).parent.parent / "config" / "botsort.yaml",  # 소스 코드 위치
            Path(__file__).parent.parent.parent.parent / "config" / "botsort.yaml",  # 프로젝트 루트
        ]
        
        # 설치된 환경에서 share 디렉토리 찾기
        # __file__이 install/camera_node/lib/python3.10/site-packages/camera_node/tracker_manager.py인 경우
        # install/camera_node/share/camera_node/config/botsort.yaml을 찾아야 함
        current_path = Path(__file__).resolve()
        # install 디렉토리 찾기
        parts = current_path.parts
        if 'install' in parts:
            install_idx = parts.index('install')
            if install_idx + 1 < len(parts):
                install_base = Path(*parts[:install_idx + 2])  # install/camera_node까지
                share_path = install_base / "share" / "camera_node" / "config" / "botsort.yaml"
                possible_paths.insert(0, share_path)  # 가장 먼저 확인
        
        tracker_cfg = None
        for path in possible_paths:
            if path.exists():
                tracker_cfg = path
                break
        
        if tracker_cfg is None:
            # 기본값: 소스 코드 위치
            tracker_cfg = Path(__file__).parent.parent / "config" / "botsort.yaml"
            print(f"Warning: botsort.yaml not found in expected locations. Using: {tracker_cfg}")
        
        self.tracker_cfg = str(tracker_cfg)
        self.tracker_type = "botsort_reid"
        
        # 추적 상태 관리
        self.state = TrackingState.IDLE
        self.target_track_id: Optional[int] = None  # 추적 대상 ID
        self.lost_frames = 0  # 놓친 프레임 수
        self.max_lost_frames = 45  # 최대 놓친 프레임 수 (약 1.5초, 30FPS 기준)
        self.frame_center = None  # 프레임 중심점 (가장 가까운 사람 선택용)
        
        # Manual 모드 지원
        self.manual_mode = False  # True면 상태 자동 전이 비활성화
        
        # Interaction 모드 지원 (True: 타겟 자동 선택 활성화, False: IDLE 모드)
        self.interaction_mode = False
        
        # 타겟이 명시적으로 설정되었는지 표시 (상태 머신이 덮어쓰지 않도록)
        self.target_explicitly_set = False
        
        # 타겟 후보가 되기 위한 최소 지속 시간 (초)
        self.min_target_duration = 1.4
        # 각 track_id의 첫 등장 시간 추적
        self.track_id_first_seen: Dict[int, float] = {}
        
        # WAIST_FOLLOWER 전이를 위한 변수들 (목 각도 기반)
        self.neck_stable_start_time: Optional[float] = None  # 목 각도가 안정되기 시작한 시간
        self.neck_stable_duration = 5.0  # 목 각도가 안정되어야 하는 최소 시간 (초) - 6초
        self.neck_stable_threshold_deg = 3.0  # 목 각도 안정성 임계값 (도) - 5도 이내 변화면 안정으로 간주
        self.last_neck_yaw_rad: Optional[float] = None  # 이전 목 각도 (라디안)
        self.neck_stable_reference_yaw_rad: Optional[float] = None  # 안정성 기준 목 각도 (라디안)
        self.pending_face_check: bool = False  # 얼굴 검출 대기 플래그
        
        # 기존 중앙 영역 방식 (사용 안 함, 호환성을 위해 유지)
        self.center_zone_start_time: Optional[float] = None  # 중앙 영역에 머물기 시작한 시간
        self.center_zone_duration = 5.0  # 중앙 영역에 머물러야 하는 최소 시간 (초)
        self.center_zone_radius_ratio = 0.15  # 프레임 크기 대비 중앙 영역 반경 비율 (예: 0.15 = 15%)

        # 모델 디바이스 확인 및 GPU 워밍업
        print(f"YOLO 모델 디바이스: {self.yolo_model.device}")
        print(f"Face 모델 디바이스: {self.face_model.device}")
        
        # GPU 워밍업
        print("GPU 워밍업 시작...")
        warmup_start = time.monotonic()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.yolo_model.predict(dummy, imgsz=640, verbose=False)
        warmup_time = (time.monotonic() - warmup_start) * 1000
        print(f"워밍업 추론 시간: {warmup_time:.1f}ms")
        
        if torch.cuda.is_available():
            print("GPU 동기화 작업 중 ...")
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU 동기화 완료!! (메모리 사용: {allocated:.1f}MB)")
        else:
            print("⚠️  CUDA 사용 불가 - CPU에서 실행 중")
        
    def _is_in_center_zone(self, centroid: tuple, frame_shape: tuple) -> bool:
        """
        중앙 좌표가 프레임의 중앙 영역 안에 있는지 확인
        
        Args:
            centroid: 객체의 중심 좌표 (x, y)
            frame_shape: 프레임 크기 (height, width) 또는 (height, width, channels)
            
        Returns:
            중앙 영역 안에 있으면 True, 아니면 False
        """
        cx, cy = centroid
        # frame_shape가 3차원이면 앞의 2개만 사용
        frame_height, frame_width = frame_shape[:2]
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0
        
        # 중앙 영역 반경 계산
        radius = min(frame_width, frame_height) * self.center_zone_radius_ratio
        
        # 중심점까지의 거리 계산
        distance = np.sqrt((cx - frame_center_x)**2 + (cy - frame_center_y)**2)
        
        return distance <= radius
    
    def _find_closest_person(self, tracked_objects: List[TrackedObject], frame_shape: tuple, current_time: float) -> Optional[int]:
        """
        프레임 중심에 가장 가까운 사람 찾기 (최소 2초 이상 지속된 객체만 후보)
        
        Args:
            tracked_objects: 추적된 객체 리스트
            frame_shape: 프레임 크기 (height, width)
            current_time: 현재 시간 (time.monotonic())
            
        Returns:
            가장 가까운 사람의 track_id 또는 None (2초 이상 지속된 객체만)
        """
        if not tracked_objects:
            return None
        
        # 현재 프레임에 나타난 track_id 업데이트
        current_frame_ids = set()
        for obj in tracked_objects:
            track_id = obj.track_id
            current_frame_ids.add(track_id)
            # 처음 보는 track_id면 등장 시간 기록
            if track_id not in self.track_id_first_seen:
                self.track_id_first_seen[track_id] = current_time
        
        # 사라진 track_id 제거 (메모리 관리)
        disappeared_ids = set(self.track_id_first_seen.keys()) - current_frame_ids
        for track_id in disappeared_ids:
            del self.track_id_first_seen[track_id]
        
        # 최소 지속 시간 이상인 객체만 필터링
        valid_objects = []
        for obj in tracked_objects:
            track_id = obj.track_id
            if track_id in self.track_id_first_seen:
                duration = current_time - self.track_id_first_seen[track_id]
                if duration >= self.min_target_duration:
                    valid_objects.append(obj)
        
        if not valid_objects:
            return None
        
        # 유효한 객체 중에서 가장 가까운 사람 찾기
        frame_center_y, frame_center_x = frame_shape[0] / 2, frame_shape[1] / 2
        
        min_distance = float('inf')
        closest_id = None
        
        for obj in valid_objects:
            cx, cy = obj.centroid
            # 중심점까지의 거리 계산
            distance = np.sqrt((cx - frame_center_x)**2 + (cy - frame_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_id = obj.track_id
        
        return closest_id
    
    def set_manual_mode(self, enabled: bool) -> None:
        """
        Manual 모드 설정
        
        Args:
            enabled: True면 Manual 모드 (상태 자동 전이 비활성화), False면 Auto 모드
        """
        self.manual_mode = enabled
    
    def set_interaction_mode(self, enabled: bool) -> None:
        """
        Interaction 모드 설정
        
        Args:
            enabled: True면 Interaction 모드 (타겟 자동 선택 활성화), False면 IDLE 모드
        """
        self.interaction_mode = enabled
        self.reset_timers()  # 모드 전환 시 타이머 초기화
        
        if enabled:
            # Interaction 모드 시작 시 INTERACTION 상태로 전환
            self.state = TrackingState.INTERACTION
            self.target_track_id = None
            self.target_explicitly_set = False
        else:
            # IDLE 모드로 전환 시 IDLE 상태로
            self.state = TrackingState.IDLE
            self.target_track_id = None
            self.target_explicitly_set = False
    
    def reset_timers(self) -> None:
        """모든 타이머 및 안정성 관련 변수 초기화 (STOP 시 호출)"""
        self.neck_stable_start_time = None
        self.last_neck_yaw_rad = None
        self.neck_stable_reference_yaw_rad = None
        self.center_zone_start_time = None
        self.lost_frames = 0
        self.pending_face_check = False
    
    def set_state(self, state: TrackingState, target_track_id: Optional[int] = None) -> None:
        """
        Manual 모드에서 상태를 수동으로 설정
        
        Args:
            state: 설정할 상태
            target_track_id: 타겟 track_id (TRACKING 상태일 때 필요)
        """
        # IDLE 상태로 전환 시 타이머 초기화 (Manual 모드와 상관없이)
        if state == TrackingState.IDLE:
            self.reset_timers()
        
        if not self.manual_mode:
            return  # Manual 모드가 아니면 무시
        
        self.state = state
        if target_track_id is not None:
            self.target_track_id = int(target_track_id)
            self.target_explicitly_set = True  # 타겟이 명시적으로 설정됨
        elif state != TrackingState.TRACKING:
            self.target_track_id = None
            self.target_explicitly_set = False
        self.lost_frames = 0
    
    def set_target(self, target_track_id: int) -> None:
        """
        타겟 변경 (Auto/Manual 모드 모두에서 작동) - 강제 즉시 적용
        
        이 메서드는 Auto 모드와 Manual 모드 모두에서 작동합니다.
        타겟이 명시적으로 설정되면 (target_explicitly_set=True),
        Auto 모드의 상태 머신도 이 타겟을 덮어쓰지 않습니다.
        
        Args:
            target_track_id: 변경할 타겟 track_id
        """
        self.target_track_id = int(target_track_id)
        self.state = TrackingState.TRACKING
        self.lost_frames = 0
        self.target_explicitly_set = True  # Auto 모드에서도 상태 머신이 덮어쓰지 않도록
    
    def is_facing_me(self, frame: np.ndarray, bbox: tuple) -> bool:
        """
        타겟이 나를 보고 있는지 확인 (얼굴 검출)
        
        Args:
            frame: 원본 프레임 (numpy array)
            bbox: 바운딩 박스 (x1, y1, x2, y2)
            
        Returns:
            얼굴이 검출되면 True
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # bbox 영역 슬라이싱 (O(1) - NumPy view)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        crop = frame[y1:y2, x1:x2]
        
        # 얼굴 검출
        results = self.face_model.predict(crop, conf=0.5, verbose=False)
        
        # 얼굴이 1개 이상 검출되면 True
        if results and len(results[0].boxes) > 0:
            print(f"얼굴이 검출되었습니다. {results[0].boxes}")
            return True
        return False

    def update_neck_angle(self, current_neck_yaw_rad: float) -> None:
        """
        목 각도를 업데이트하고 안정성을 확인하여 WAIST_FOLLOWER 상태로 전이
        
        Args:
            current_neck_yaw_rad: 현재 목 Yaw 각도 (라디안)
        """
        current_time = time.monotonic()
        
        # TRACKING 상태가 아니면 무시
        if self.state != TrackingState.TRACKING:
            self.neck_stable_start_time = None
            self.last_neck_yaw_rad = None
            self.neck_stable_reference_yaw_rad = None
            return
        
        # 이전 목 각도가 없으면 초기화
        if self.last_neck_yaw_rad is None:
            self.last_neck_yaw_rad = current_neck_yaw_rad
            self.neck_stable_start_time = None
            self.neck_stable_reference_yaw_rad = None
            return
        
        # 목 각도 변화량 계산 (도 단위) - 이전 프레임과의 변화
        angle_change_deg = abs(math.degrees(current_neck_yaw_rad - self.last_neck_yaw_rad))
        
        # 기준 각도가 없으면 현재 각도를 기준으로 설정
        if self.neck_stable_reference_yaw_rad is None:
            self.neck_stable_reference_yaw_rad = current_neck_yaw_rad
        
        # 기준 각도로부터의 변화량 계산 (도 단위)
        reference_change_deg = abs(math.degrees(current_neck_yaw_rad - self.neck_stable_reference_yaw_rad))
        
        # 목 각도가 기준 각도 주변에서 임계값(5도) 이내로 안정되어 있는지 확인
        # AND 이전 프레임과의 변화도 작아야 함 (연속적인 안정성)
        if reference_change_deg <= self.neck_stable_threshold_deg and angle_change_deg <= self.neck_stable_threshold_deg:
            # 안정적이면 시간 추적 시작/계속
            if self.neck_stable_start_time is None:
                self.neck_stable_start_time = current_time
                self.neck_stable_reference_yaw_rad = current_neck_yaw_rad  # 기준 각도 설정
            
            # 일정 시간 이상 안정되면 얼굴 검출 대기 플래그 설정
            elapsed_time = current_time - self.neck_stable_start_time
            if elapsed_time >= self.neck_stable_duration:
                # process()에서 얼굴 검출을 수행하도록 플래그 설정
                self.pending_face_check = True
        else:
            # 각도 변화가 크면 시간과 기준 각도 리셋
            self.neck_stable_start_time = None
            self.neck_stable_reference_yaw_rad = None  # 기준 각도 리셋
        
        # 현재 각도를 이전 각도로 저장
        self.last_neck_yaw_rad = current_neck_yaw_rad
    
    def get_center_zone_elapsed_time(self) -> Optional[float]:
        """
        목 각도 안정성 경과 시간 반환 (WAIST_FOLLOWER 전이용)
        
        Returns:
            경과 시간 (초) 또는 None (목 각도가 안정되지 않았거나 시간 추적이 시작되지 않은 경우)
        """
        if self.neck_stable_start_time is None:
            return None
        
        current_time = time.monotonic()
        elapsed_time = current_time - self.neck_stable_start_time
        return elapsed_time
    
    def process(self, frame: np.ndarray) -> tuple[List[TrackedObject], TargetInfo]:
        """
        단일 프레임에 대해 YOLO + BoT-SORT(+ReID) 추적 수행
        상태 머신을 통한 추적 대상 관리
        
        Returns:
            tuple: (추적된 객체 리스트, 타겟 정보)
                - 추적된 객체 리스트: List[TrackedObject]
                - 타겟 정보: TargetInfo(point, state, track_id)
        """
        results = self.yolo_model.track(
            frame,
            conf=self.conf_threshold,
            classes=[PERSON_CLASS_ID],   # 사람만
            tracker=self.tracker_cfg,    # botsort.yaml (ReID 포함)
            persist=True,                # 내부 tracker 상태 유지
            verbose=False,
            imgsz=640,
        )

        tracked_objects: List[TrackedObject] = []
        
        if not results:
            # 감지된 객체가 없으면 상태 업데이트 (Manual 모드가 아닐 때만 자동 전이)
            if not self.manual_mode:
                match self.state:
                    case TrackingState.TRACKING:
                        self.state = TrackingState.LOST
                        self.lost_frames = 0
                        self.center_zone_start_time = None  # 리셋
                    case TrackingState.WAIST_FOLLOWER:
                        # WAIST_FOLLOWER 상태에서는 다른 상태로 전이하지 않음
                        pass
                    case TrackingState.LOST:
                        self.lost_frames += 1
                        if self.lost_frames >= self.max_lost_frames:
                            self.state = TrackingState.SEARCHING
                            self.target_track_id = None
                    case _:
                        pass
            
            # 타겟 정보 생성 (객체 없음) - 설정된 타겟 ID는 유지
            target_info = TargetInfo(
                point=None,
                state=self.state,
                track_id=self.target_track_id
            )
            return tracked_objects, target_info

        r = results[0]
        boxes = r.boxes
        # 감지된 객체가 없거나 아직 ID가 안 붙었으면 빈 리스트 반환
        if boxes is None or boxes.id is None:
            # Manual 모드가 아닐 때만 자동 상태 전이
            if not self.manual_mode:
                match self.state:
                    case TrackingState.TRACKING:
                        self.state = TrackingState.LOST
                        self.lost_frames = 0
                        self.center_zone_start_time = None  # 리셋
                    case TrackingState.WAIST_FOLLOWER:
                        # WAIST_FOLLOWER 상태에서는 다른 상태로 전이하지 않음
                        pass
                    case TrackingState.LOST:
                        self.lost_frames += 1
                        if self.lost_frames >= self.max_lost_frames:
                            self.state = TrackingState.SEARCHING
                            self.target_track_id = None
                    case _:
                        pass
            
            # 타겟 정보 생성 (박스 없음) - 설정된 타겟 ID는 유지
            target_info = TargetInfo(
                point=None,
                state=self.state,
                track_id=self.target_track_id
            )
            return tracked_objects, target_info
            

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # 모든 추적 객체 생성
        all_objects = []
        for tid, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            all_objects.append(
                TrackedObject(
                    track_id=int(tid),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    centroid=(float(cx), float(cy)),
                    state="tracking",
                    confidence=float(conf),
                    age=0,
                )
            )

        # 현재 시간 기록 (타겟 후보 필터링용)
        current_time = time.monotonic()
        
        # 타겟이 설정되어 있으면 현재 프레임에 존재하는지 확인
        target_exists = (
            self.target_track_id is not None and
            any(obj.track_id == self.target_track_id for obj in all_objects)
        )
        
        # WAIST_FOLLOWER, INTERACTION 상태에서는 자동으로 다른 상태로 전이하지 않음
        # INTERACTION은 완전히 독립적인 상태 머신
        # 단, ControllerManager에서 명시적으로 상태 변경을 요청한 경우는 허용
        # (예: 목표 도착 완료 시 TRACKING으로 복귀)
        if target_exists and self.state not in (TrackingState.WAIST_FOLLOWER, TrackingState.INTERACTION):
            if self.state not in (TrackingState.TRACKING, TrackingState.WAIST_FOLLOWER, TrackingState.INTERACTION):
                self.state = TrackingState.TRACKING
            self.lost_frames = 0
        
        # 상태 머신 처리 (Manual 모드가 아닐 때만 자동 전이)
        # 타겟이 명시적으로 설정된 경우(target_explicitly_set=True)에는 타겟을 변경하지 않음
        if not self.manual_mode:
            match self.state:
                case TrackingState.IDLE:
                    # IDLE 상태: Auto 모드면 자동 타겟 선택, Manual 모드면 GUI 선택 대기
                    # GUI에서 명시적으로 타겟을 선택한 경우에는 TRACKING으로 전이
                    if self.target_explicitly_set and target_exists:
                        self.state = TrackingState.TRACKING
                        self.lost_frames = 0
                    elif not self.manual_mode:
                        # Auto 모드: 가장 가까운 사람 자동 선택
                        if self.target_track_id is None:
                            closest_id = self._find_closest_person(all_objects, frame.shape, current_time)
                            if closest_id is not None:
                                self.target_track_id = closest_id
                                self.state = TrackingState.TRACKING
                                self.lost_frames = 0
                                self.target_explicitly_set = False  # 자동 선택
                        elif target_exists:
                            # 설정된 타겟이 존재하면 TRACKING 상태로 전환
                            self.state = TrackingState.TRACKING
                            self.lost_frames = 0
                
                case TrackingState.INTERACTION:
                    # Interaction 상태: 단일 상태로 타겟 추적 (BB Box만 침)
                    # 타겟이 없으면 가장 가까운 사람 자동 선택
                    # 타겟을 잃으면 즉시 새 타겟 선택 (LOST/SEARCHING 없음)
                    if self.target_track_id is None or not target_exists:
                        # 타겟이 없거나 잃었으면 새 타겟 선택
                        closest_id = self._find_closest_person(all_objects, frame.shape, current_time)
                        if closest_id is not None:
                            self.target_track_id = closest_id
                            self.target_explicitly_set = False  # 자동 선택
                    # INTERACTION 상태 유지 (다른 상태로 전이 없음)
                
                case TrackingState.TRACKING:
                    # 추적 대상이 있는지 확인
                    if target_exists:
                        # 얼굴 검출 대기 플래그가 설정되어 있으면 얼굴 검출 수행
                        if self.pending_face_check:
                            # 타겟의 bbox 찾기
                            target_obj = next((obj for obj in all_objects if obj.track_id == self.target_track_id), None)
                            if target_obj is not None:
                                if self.is_facing_me(frame, target_obj.bbox):
                                    # 얼굴이 보이면 WAIST_FOLLOWER로 전이
                                    self.state = TrackingState.WAIST_FOLLOWER
                                    self.neck_stable_start_time = None
                                    self.last_neck_yaw_rad = None
                                    self.neck_stable_reference_yaw_rad = None
                                else:
                                    # 얼굴이 안 보이면 TRACKING 유지, 타이머 리셋
                                    self.neck_stable_start_time = None
                                    self.last_neck_yaw_rad = None
                                    self.neck_stable_reference_yaw_rad = None
                            self.pending_face_check = False
                    
                    # 타겟이 명시적으로 설정된 경우에는 타겟을 변경하지 않음
                    if not target_exists and self.target_track_id is not None and not self.target_explicitly_set:
                        # 추적 대상 놓침 (자동 선택된 타겟만)
                        self.state = TrackingState.LOST
                        self.lost_frames = 0
                        self.neck_stable_start_time = None  # 리셋
                        self.last_neck_yaw_rad = None  # 리셋
                        self.neck_stable_reference_yaw_rad = None  # 리셋
                        self.pending_face_check = False  # 리셋
                
                case TrackingState.LOST:
                    # WAIST_FOLLOWER 상태가 아니면만 처리
                    if self.state != TrackingState.WAIST_FOLLOWER:
                        # 추적 대상이 다시 나타났는지 확인
                        if target_exists:
                            # 찾았으면 다시 추적
                            self.state = TrackingState.TRACKING
                            self.lost_frames = 0
                        else:
                            # 계속 놓침
                            self.lost_frames += 1
                            if self.lost_frames >= self.max_lost_frames:
                                # 너무 오래 놓쳤으면 주변 탐색
                                # 단, 타겟이 명시적으로 설정되어 있으면 유지
                                if self.target_track_id is None or not self.target_explicitly_set:
                                    if self.target_track_id is not None:
                                        self.target_track_id = None  # 자동 선택된 타겟만 해제
                                        self.target_explicitly_set = False
                                    self.state = TrackingState.SEARCHING
                
                case TrackingState.SEARCHING:
                    # SEARCHING 상태: 사람을 찾으면 TRACKING으로 전환
                    if all_objects:
                        # 가장 가까운 사람 자동 선택
                        closest_id = self._find_closest_person(all_objects, frame.shape, current_time)
                        if closest_id is not None:
                            self.target_track_id = closest_id
                            self.state = TrackingState.TRACKING
                            self.lost_frames = 0
                            self.target_explicitly_set = False  # 자동 선택
                
                case TrackingState.WAIST_FOLLOWER:
                    # WAIST_FOLLOWER 상태에서는 타겟을 잃어도 상관없음
                    # 허리와 목 제어만 계속 진행 (타겟 추적은 중단)
                    # 이 상태에서는 타겟 추적에 의존하지 않고 이미 설정된 동작을 수행
                    # 단, ControllerManager에서 목표 도착 완료 시 TRACKING으로 상태 변경을 요청할 수 있음
                    # (상태 변경은 ControllerManager에서 직접 처리)
                    pass
                
                case _:
                    pass

        # 상태에 따라 객체에 상태 할당 및 타겟 정보 추출
        target_point = None
        target_track_id = None
        
        for obj in all_objects:
            if obj.track_id == self.target_track_id:
                # 추적 대상은 "target" 상태로 표시
                tracked_objects.append(
                    TrackedObject(
                        track_id=obj.track_id,
                        bbox=obj.bbox,
                        centroid=obj.centroid,
                        state="target",
                        confidence=obj.confidence,
                        age=obj.age,
                    )
                )
                # 타겟 정보 저장 - 바운딩 박스 높이의 0.2 지점 (머리 쪽)
                x1, y1, x2, y2 = obj.bbox
                target_point = ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.2)
                target_track_id = obj.track_id
            else:
                # 다른 객체는 현재 시스템 상태에 따라 표시
                tracked_objects.append(
                    TrackedObject(
                        track_id=obj.track_id,
                        bbox=obj.bbox,
                        centroid=obj.centroid,
                        state=self.state.value,
                        confidence=obj.confidence,
                        age=obj.age,
                    )
                )

        # 타겟 정보 생성 - 타겟이 현재 프레임에 없어도 설정된 타겟 ID 유지
        target_info = TargetInfo(
            point=target_point,
            state=self.state,
            track_id=target_track_id if target_track_id is not None else self.target_track_id
        )

        return tracked_objects, target_info