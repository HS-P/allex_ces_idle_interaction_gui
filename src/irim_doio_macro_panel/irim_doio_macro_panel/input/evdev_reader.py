from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from PyQt5 import QtCore
from evdev import InputDevice, categorize, ecodes
from glob import glob


def _parse_int_maybe(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("0x"):
            return int(s, 16)
        return int(s, 10)
    return None


class EvdevHotplugReader(QtCore.QObject):
    """
    Reads EV_KEY from multiple evdev devices using QSocketNotifier.
    - Supports grab/ungrab.
    - Periodically attempts to reconnect to missing devices.
    - Can dynamically resolve /dev/input/event* by match rules (Bluetooth-safe).
    """
    key_event = QtCore.pyqtSignal(str, bool, bool)  # (code, is_down, is_up)
    device_status = QtCore.pyqtSignal(str)          # human-readable status lines

    def __init__(
        self,
        device_paths: Optional[List[str]] = None,
        grab: bool = True,
        reconnect_period_ms: int = 500,
        parent=None,
        match: Optional[Dict] = None,   # <-- NEW: {"name":..., "vendor":..., "product":..., "uniq":..., "phys":...}
    ):
        super().__init__(parent)
        self._grab = bool(grab)
        self._reconnect_period_ms = int(reconnect_period_ms)

        self._static_paths: List[str] = list(device_paths or [])
        self._match: Dict = dict(match or {})

        # normalize match fields
        self._match_name: Optional[str] = self._match.get("name")
        self._match_vendor: Optional[int] = _parse_int_maybe(self._match.get("vendor"))
        self._match_product: Optional[int] = _parse_int_maybe(self._match.get("product"))
        self._match_uniq: Optional[str] = (self._match.get("uniq") or None)
        self._match_phys: Optional[str] = (self._match.get("phys") or None)

        # if match is provided, ignore static paths unless user explicitly sets paths too
        self._use_dynamic = bool(self._match_name or self._match_vendor or self._match_product or self._match_uniq or self._match_phys)

        self._devices: List[InputDevice] = []
        self._notifiers: List[QtCore.QSocketNotifier] = []

        self._reconnect_timer = QtCore.QTimer(self)
        self._reconnect_timer.timeout.connect(self._check_and_reconnect)
        self._reconnect_timer.setInterval(self._reconnect_period_ms)

        self._open_all()

        # 연결돼있으면 타이머 끔(= 연결 중엔 스캔 안함)
        if not self._devices:
            self._reconnect_timer.start(self._reconnect_period_ms)
        else:
            self._reconnect_timer.stop()
        

    def close(self):
        for dev in list(self._devices):
            self._close_device(dev)

        for nt in list(self._notifiers):
            try:
                nt.setEnabled(False)
                nt.deleteLater()
            except Exception:
                pass

        self._devices.clear()
        self._notifiers.clear()

    # -------------------------

    def _resolve_paths(self) -> List[str]:
        """
        If match rules exist, scan /dev/input/event* and return matching paths.
        Otherwise, return the static paths list.
        """
        if not self._use_dynamic:
            return list(self._static_paths)

        matched: List[str] = []
        for p in sorted(glob("/dev/input/event*")):
            try:
                dev = InputDevice(p)
            except Exception:
                continue

            try:
                # name
                if self._match_name is not None and dev.name != self._match_name:
                    continue
                # vendor/product
                if self._match_vendor is not None and dev.info.vendor != self._match_vendor:
                    continue
                if self._match_product is not None and dev.info.product != self._match_product:
                    continue
                # uniq / phys (string exact match)
                if self._match_uniq is not None and (dev.uniq or "").lower() != self._match_uniq.lower():
                    continue
                if self._match_phys is not None and (dev.phys or "").lower() != self._match_phys.lower():
                    continue

                # must be key device
                caps = dev.capabilities(verbose=False)
                if ecodes.EV_KEY not in caps:
                    continue

                matched.append(p)

            finally:
                # IMPORTANT: close this temporary handle to avoid FD leaks
                try:
                    dev.close()
                except Exception:
                    pass

        if not matched:
            self.device_status.emit(
                "[WAIT] No matching input device yet. "
                f"(name={self._match_name}, vendor={self._match_vendor}, product={self._match_product}, "
                f"uniq={self._match_uniq}, phys={self._match_phys})"
            )

        return matched

    def _open_all(self):
        for p in self._resolve_paths():
            self._try_open_path(p)

    def _try_open_path(self, path: str):
        if any(d.path == path for d in self._devices):
            return
        try:
            dev = InputDevice(path)
        except FileNotFoundError:
            self.device_status.emit(f"[WAIT] Device not found: {path}")
            return
        except PermissionError:
            self.device_status.emit(f"[ERROR] Permission denied: {path} (try sudo or udev rules)")
            return
        except Exception as e:
            self.device_status.emit(f"[ERROR] Open failed: {path} ({e})")
            return

        if self._grab:
            try:
                dev.grab()
            except Exception as e:
                self.device_status.emit(f"[WARN] grab() failed: {path} ({e})")

        self._devices.append(dev)

        # 하나라도 붙었으면 이제 폴링 중단 (UI 멈칫 원인 제거)
        if self._devices and self._reconnect_timer.isActive():
            self._reconnect_timer.stop()

        nt = QtCore.QSocketNotifier(dev.fileno(), QtCore.QSocketNotifier.Read, self)
        nt.activated.connect(lambda _fd, d=dev: self._on_activity(d))
        self._notifiers.append(nt)
        self.device_status.emit(
            f"[OK] Opened: {path} ({dev.name}) "
            f"[vendor={hex(dev.info.vendor)} product={hex(dev.info.product)} uniq={dev.uniq} phys={dev.phys}]"
        )

    def _close_device(self, dev: InputDevice):
        try:
            if self._grab:
                dev.ungrab()
        except Exception:
            pass
        try:
            dev.close()
        except Exception:
            pass

    def _remove_device(self, dev: InputDevice):
        try:
            idx = self._devices.index(dev)
        except ValueError:
            return
        nt = self._notifiers[idx]
        try:
            nt.setEnabled(False)
            nt.deleteLater()
        except Exception:
            pass
        self._notifiers.pop(idx)

        self._close_device(dev)
        self._devices.pop(idx)

        if not self._devices:
            # 끊긴 순간부터만 재탐색 시작
            self._reconnect_timer.setInterval(self._reconnect_period_ms)
            self._reconnect_timer.start()

    def _on_activity(self, dev: InputDevice):
        try:
            for event in dev.read():
                if event.type != ecodes.EV_KEY:
                    continue
                ke = categorize(event)
                is_down = (ke.keystate == ke.key_down)
                is_up = (ke.keystate == ke.key_up)

                keycode = ke.keycode
                codes = keycode if isinstance(keycode, list) else [keycode]
                for c in codes:
                    self.key_event.emit(str(c), bool(is_down), bool(is_up))

        except BlockingIOError:
            return
        except OSError as e:
            self.device_status.emit(f"[WARN] Device lost: {dev.path} ({e})")
            self._remove_device(dev)

    def _check_and_reconnect(self):
        if self._devices:
            # 연결 중에는 스캔 자체를 하지 않음
            if self._reconnect_timer.isActive():
                self._reconnect_timer.stop()
            return

        desired_paths = self._resolve_paths()
        desired_set = set(desired_paths)

        # close devices that are no longer in desired set (ex: event number changed)
        for dev in list(self._devices):
            if dev.path not in desired_set:
                self.device_status.emit(f"[INFO] Closing outdated device: {dev.path}")
                self._remove_device(dev)

        # open missing desired devices
        existing = {d.path for d in self._devices}
        for p in desired_paths:
            if p not in existing:
                self._try_open_path(p)