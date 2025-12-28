# irim_doio_macro_panel

DOIO 매크로 키보드 입력(evdev)을 받아서 ROS2 String 토픽으로 publish 하는 GUI 노드입니다.

- JSON 매핑 파일을 로드/수정/즉시 저장
- 9개 레이어별로 keycode -> action/suffix 매핑
- (옵션) chord(조합키): "B를 누른 상태에서 A를 누르면" 같은 조건 지원
- 선택된 로봇 이름을 prefix로 붙여서 publish

## Run

```bash
# build
colcon build --symlink-install
source install/setup.bash

# run (first time: ~/.config/.../mapping.json is auto-created)
ros2 run irim_doio_macro_panel doio_macro_panel
```

## Notes

- Linux에서 같은 KEY_A를 여러 물리 키에 매핑하면, 소프트웨어에서 구분할 수 없습니다.
  "키 하나하나"를 software에서 구분하고 싶으면, 펌웨어(QMK/VIA)에서 각 버튼에 **서로 다른 keycode**를 배치하세요.
