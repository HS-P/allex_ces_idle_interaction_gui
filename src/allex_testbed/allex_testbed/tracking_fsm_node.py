#!/usr/bin/env python3
"""
Tracking FSM Node 타입 정의 (테스트베드용)
기존 패키지의 타입을 재정의하여 독립적으로 사용
"""
from collections import namedtuple
from enum import Enum

# 다른 파일에서 사용하는 타입들 export
TrackedObject = namedtuple('TrackedObject', [
    'track_id', 'bbox', 'centroid', 'state', 'confidence', 'age'
])

TargetInfo = namedtuple('TargetInfo', [
    'point',      # 타겟 중심점 (x, y) 또는 None
    'state',      # 현재 추적 상태 (TrackingState)
    'track_id',   # 타겟 track_id 또는 None
])

class TrackingState(Enum):
    """추적 상태"""
    IDLE = "idle"
    TRACKING = "tracking"
    LOST = "lost"
    SEARCHING = "searching"
    WAIST_FOLLOWER = "waist_follower"
    HELLO = "hello"
    INTERACTION = "interaction"

