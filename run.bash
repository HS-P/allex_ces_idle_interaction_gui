#!/bin/bash
set -euo pipefail

# ============================================
# ALLEX Idle Interaction 실행 스크립트
# PC1 (로컬): GUI + Joystick
# PC2 (mars): 카메라
# PC3 (dgx_allex_one): 제어 노드
#
# 종료 시:
#  - 원격(PC2/PC3): setsid로 만든 "세션(SID)만" pkill -s 로 종료 (INT→TERM→KILL)
#  - 로컬(PC1): 다른 패널 보호 위해 "세션/그룹 kill 금지"
#              대신 "내가 띄운 root PID 트리만" INT→TERM→KILL
# ============================================

# ---------- 원격 PC 설정 ----------
PC2_IP="192.168.80.8"
PC2_USER="mars"
PC2_PW="0000"

PC3_IP="192.168.80.9"
PC3_USER="dgx_allex_one"
PC3_PW="00000000"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

# ---------- PID / SID ----------
PC1_PID=""          # 로컬: ros2 launch root PID
PC2_PID=""          # 원격: setsid로 띄운 프로세스 PID
PC3_PID=""

PC2_SID=""          # 원격: 실제 세션 ID
PC3_SID=""

# ---------- helpers ----------
# 로컬: root_pid의 "프로세스 트리"만 재귀 종료 (세션/그룹 전체 kill 금지)
kill_local_tree() {
  local root_pid="${1:-}"
  [ -z "$root_pid" ] && return 0

  # 이미 죽었으면 종료
  if ! kill -0 "$root_pid" 2>/dev/null; then
    return 0
  fi

  # 자식 PID 재귀 수집/종료 (자식부터)
  local kids=""
  kids=$(pgrep -P "$root_pid" 2>/dev/null || true)
  for p in $kids; do
    kill_local_tree "$p"
  done

  # 자신 종료 (INT → TERM → KILL)
  kill -INT  "$root_pid" 2>/dev/null || true
  sleep 0.5
  kill -TERM "$root_pid" 2>/dev/null || true
  sleep 0.5
  kill -KILL "$root_pid" 2>/dev/null || true
}

# 원격: PID로부터 SID 조회
remote_get_sid_by_pid() {
  local ip="$1" user="$2" pw="$3" pid="$4"
  sshpass -p "$pw" ssh "${SSH_OPTS[@]}" "$user@$ip" \
    "ps -o sid= -p $pid 2>/dev/null | tr -d ' ' || true" 2>/dev/null
}

# 원격: 세션(SID) 단위 종료 (INT → TERM → KILL)
kill_remote_session() {
  local ip="$1" user="$2" pw="$3" sid="${4:-}" label="$5"
  [ -z "$sid" ] && return 0
  echo "[종료] $label session kill (SID): $sid"
  sshpass -p "$pw" ssh "${SSH_OPTS[@]}" "$user@$ip" \
    "pkill -INT  -s $sid 2>/dev/null || true; sleep 1;
     pkill -TERM -s $sid 2>/dev/null || true; sleep 1;
     pkill -KILL -s $sid 2>/dev/null || true" 2>/dev/null || true
}

cleanup() {
  echo -e "\n[종료] 원격 및 로컬 프로세스 정리 시작..."

  # 원격: 내가 setsid로 만든 세션만 종료
  kill_remote_session "$PC3_IP" "$PC3_USER" "$PC3_PW" "$PC3_SID" "PC3 (제어기)"
  kill_remote_session "$PC2_IP" "$PC2_USER" "$PC2_PW" "$PC2_SID" "PC2 (카메라)"

  # 로컬: 내 프로세스 트리만 종료 (다른 패널 보호)
  if [ -n "${PC1_PID:-}" ]; then
    echo "[종료] PC1 tree kill (root PID): $PC1_PID"
    kill_local_tree "$PC1_PID"
  fi

  echo "[종료] 모든 프로세스 정리 완료"
  exit 0
}

trap cleanup SIGINT SIGTERM

# ---------- 워크스페이스 소스 확인 (로컬) ----------
if [ -f "$HOME/allex_ces_idle_interaction_gui/install/setup.bash" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$HOME/allex_ces_idle_interaction_gui/install/setup.bash"
  set -u
  echo "[설정] 로컬 워크스페이스 소스 완료"
elif [ -f "./install/setup.bash" ]; then
  set +u
  # shellcheck disable=SC1091
  source "./install/setup.bash"
  set -u
  echo "[설정] 로컬 워크스페이스 소스 완료 (상대 경로)"
else
  echo "[경고] 워크스페이스 setup.bash를 찾을 수 없습니다."
  echo "      수동으로 source를 실행하거나 절대 경로를 확인하세요."
fi

# ---------- PC3 실행 (원격: setsid로 새 세션 생성, PID/SID 저장) ----------
echo "[진행] PC3 (제어기) 노드 실행 중..."
PC3_PID=$(
  sshpass -p "$PC3_PW" ssh "${SSH_OPTS[@]}" "$PC3_USER@$PC3_IP" \
  "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0};
   source ~/allex_ces_idle_interaction_gui/install/setup.bash;
   setsid bash -lc 'exec ros2 launch allex_ces_idle_interaction allex_idle_interaction.launch.py' \
     >/tmp/allex_pc3.log 2>&1 < /dev/null & echo \$!"
)
PC3_SID="$(remote_get_sid_by_pid "$PC3_IP" "$PC3_USER" "$PC3_PW" "$PC3_PID")"
echo "[진행] PC3 remote PID: $PC3_PID"
echo "[진행] PC3 remote SID: $PC3_SID"

sleep 5

# ---------- PC2 실행 (원격: setsid로 새 세션 생성, PID/SID 저장) ----------
echo "[진행] PC2 (카메라) 노드 실행 중..."
PC2_PID=$(
  sshpass -p "$PC2_PW" ssh "${SSH_OPTS[@]}" "$PC2_USER@$PC2_IP" \
  "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0};
   source ~/allex_ces_idle_interaction/install/setup.bash;
   setsid bash -lc 'exec ros2 launch orbbec_camera femto_bolt.launch.py' \
     >/tmp/allex_pc2.log 2>&1 < /dev/null & echo \$!"
)
PC2_SID="$(remote_get_sid_by_pid "$PC2_IP" "$PC2_USER" "$PC2_PW" "$PC2_PID")"
echo "[진행] PC2 remote PID: $PC2_PID"
echo "[진행] PC2 remote SID: $PC2_SID"

sleep 5

# ---------- PC1 실행 (로컬: setsid로 띄우되, 종료는 PID 트리 kill) ----------
echo "[진행] PC1 (로컬) GUI 및 Joystick 노드 실행 중..."
setsid bash -lc "exec ros2 launch allex_ces_idle_interaction pc1_gui_joystick.launch.py" \
  >/tmp/allex_pc1.log 2>&1 < /dev/null &
PC1_PID=$!
echo "[진행] PC1 local PID: $PC1_PID"

echo ""
echo "============================================"
echo "[대기] 모든 노드가 실행 중입니다."
echo "      Ctrl+C를 누르면 이 스크립트가 띄운 프로세스만 종료됩니다."
echo "      (로컬은 세션 kill 금지: 다른 패널 보호)"
echo "============================================"

wait