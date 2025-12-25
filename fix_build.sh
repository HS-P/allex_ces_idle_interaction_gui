#!/bin/bash
# 빌드 오류 해결 스크립트
# --editable 오류 및 경로 경고 해결

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "빌드 오류 해결 스크립트"
echo "=========================================="
echo ""

# 1. Python 버전 확인 및 setuptools 업그레이드
echo "[1/4] Python 버전 확인 및 setuptools 업그레이드..."
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
echo "  현재 Python 버전: $PYTHON_VERSION"

CURRENT_VERSION=$(pip show setuptools 2>/dev/null | grep Version | awk '{print $2}')
echo "  현재 setuptools 버전: ${CURRENT_VERSION:-알 수 없음}"

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
    echo "  Python 3.12 이상 감지: setuptools를 최신 버전으로 업그레이드 중..."
    pip install --upgrade setuptools
else
    echo "  Python 3.11 이하: setuptools 58.1.0 설치 중..."
    pip install setuptools==58.1.0
fi
echo ""

# 2. 환경 변수 정리
echo "[2/4] 환경 변수 정리..."
if [ ! -z "$AMENT_PREFIX_PATH" ]; then
    echo "  AMENT_PREFIX_PATH 정리 중..."
    # 존재하지 않는 경로 제거
    NEW_AMENT_PATH=""
    IFS=':' read -ra ADDR <<< "$AMENT_PREFIX_PATH"
    for i in "${ADDR[@]}"; do
        if [ -d "$i" ]; then
            if [ -z "$NEW_AMENT_PATH" ]; then
                NEW_AMENT_PATH="$i"
            else
                NEW_AMENT_PATH="$NEW_AMENT_PATH:$i"
            fi
        fi
    done
    export AMENT_PREFIX_PATH="$NEW_AMENT_PATH"
    echo "  정리된 AMENT_PREFIX_PATH: $AMENT_PREFIX_PATH"
fi

if [ ! -z "$CMAKE_PREFIX_PATH" ]; then
    echo "  CMAKE_PREFIX_PATH 정리 중..."
    # 존재하지 않는 경로 제거
    NEW_CMAKE_PATH=""
    IFS=':' read -ra ADDR <<< "$CMAKE_PREFIX_PATH"
    for i in "${ADDR[@]}"; do
        if [ -d "$i" ]; then
            if [ -z "$NEW_CMAKE_PATH" ]; then
                NEW_CMAKE_PATH="$i"
            else
                NEW_CMAKE_PATH="$NEW_CMAKE_PATH:$i"
            fi
        fi
    done
    export CMAKE_PREFIX_PATH="$NEW_CMAKE_PATH"
    echo "  정리된 CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
fi
echo ""

# 3. 빌드 캐시 정리 (선택사항)
echo "[3/4] 빌드 캐시 정리 (선택사항)..."
read -p "  빌드 캐시를 삭제하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  빌드 캐시 삭제 중..."
    rm -rf build/ install/ log/
    echo "  완료!"
else
    echo "  건너뜀."
fi
echo ""

# 4. 빌드 실행
echo "[4/4] 빌드 실행..."
echo "  colcon build --symlink-install 실행 중..."
colcon build --symlink-install

echo ""
echo "=========================================="
echo "완료!"
echo "=========================================="

