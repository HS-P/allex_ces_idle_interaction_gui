import pyautogui
import time
import datetime

def take_screenshots():
    while True:
        # 현재 시간 정보를 이용하여 파일 이름 생성
        now = datetime.datetime.now()
        filename = now.strftime("screenshot_%Y%m%d_%H%M%S.png")
        
        # 스크린샷 찍기 및 저장
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"스크린샷 저장됨: {filename}")
        
        # 15분(900초) 대기
        time.sleep(15 *60)

if __name__ == "__main__":
    take_screenshots()
