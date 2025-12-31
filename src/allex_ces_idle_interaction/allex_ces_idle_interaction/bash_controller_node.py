#!/usr/bin/env python3
"""
Bash Controller Node - INT 메시지를 받아서 bash 스크립트를 실행/종료하는 노드
0번 INT: B Bash 종료 후 A Bash 실행
1번 INT: A Bash 종료 후 B Bash 실행
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import subprocess
import signal
import os
import threading
from typing import Optional


class BashControllerNode(Node):
    """Bash 스크립트 실행/종료 제어 노드"""
    
    def __init__(self):
        super().__init__('bash_controller_node')
        
        # 현재 실행 중인 프로세스
        self.process_a: Optional[subprocess.Popen] = None
        self.process_b: Optional[subprocess.Popen] = None
        
        # 프로세스 락 (스레드 안전성)
        self.process_lock = threading.Lock()
        
        # Bash 스크립트 경로 (파라미터로 설정 가능)
        self.declare_parameter('bash_a_path', '')
        self.declare_parameter('bash_b_path', '')
        
        self.bash_a_path = self.get_parameter('bash_a_path').get_parameter_value().string_value
        self.bash_b_path = self.get_parameter('bash_b_path').get_parameter_value().string_value
        
        # 경로가 설정되지 않은 경우 기본값
        if not self.bash_a_path:
            self.get_logger().warn('bash_a_path 파라미터가 설정되지 않았습니다.')
        if not self.bash_b_path:
            self.get_logger().warn('bash_b_path 파라미터가 설정되지 않았습니다.')
        
        # INT 메시지 구독
        self.command_subscription = self.create_subscription(
            Int32,
            '/bash_controller/command',
            self._command_callback,
            10
        )
        
        self.get_logger().info('Bash Controller Node가 시작되었습니다.')
        self.get_logger().info(f'Bash A 경로: {self.bash_a_path}')
        self.get_logger().info(f'Bash B 경로: {self.bash_b_path}')
    
    def _command_callback(self, msg: Int32):
        """명령 메시지 콜백"""
        command = msg.data
        
        if command == 0:
            self.get_logger().info('명령 0 수신: B Bash 종료 후 A Bash 실행')
            self._switch_to_bash_a()
        elif command == 1:
            self.get_logger().info('명령 1 수신: A Bash 종료 후 B Bash 실행')
            self._switch_to_bash_b()
        else:
            self.get_logger().warn(f'알 수 없는 명령: {command} (0 또는 1만 지원)')
    
    def _stop_process(self, process: Optional[subprocess.Popen], process_name: str):
        """프로세스를 Ctrl+C와 동일하게 종료 (프로세스 그룹에 SIGINT 전송)"""
        if process is None:
            return
        
        try:
            # 프로세스가 아직 실행 중인지 확인
            if process.poll() is None:  # None이면 아직 실행 중
                self.get_logger().info(f'{process_name} 프로세스 그룹에 SIGINT 전송 중... (PID: {process.pid})')
                
                # 프로세스 그룹에 SIGINT 전송 (Ctrl+C와 동일, 자식 프로세스까지 종료)
                try:
                    # os.setsid()로 생성된 프로세스 그룹에 SIGINT 전송
                    os.killpg(os.getpgid(process.pid), signal.SIGINT)
                except ProcessLookupError:
                    # 프로세스가 이미 종료된 경우
                    self.get_logger().info(f'{process_name} 프로세스가 이미 종료되었습니다.')
                    return
                except AttributeError:
                    # os.getpgid가 없는 경우 (Windows 등), 개별 프로세스에만 전송
                    process.send_signal(signal.SIGINT)
                
                # 프로세스가 종료될 때까지 대기 (최대 5초)
                try:
                    process.wait(timeout=5.0)
                    self.get_logger().info(f'{process_name} 프로세스가 정상적으로 종료되었습니다.')
                except subprocess.TimeoutExpired:
                    # 5초 내에 종료되지 않으면 프로세스 그룹에 SIGTERM 전송
                    self.get_logger().warn(f'{process_name} 프로세스가 5초 내에 종료되지 않아 SIGTERM 전송...')
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except (ProcessLookupError, AttributeError):
                        process.terminate()
                    
                    try:
                        process.wait(timeout=2.0)
                        self.get_logger().info(f'{process_name} 프로세스가 종료되었습니다.')
                    except subprocess.TimeoutExpired:
                        # 여전히 종료되지 않으면 강제 종료
                        self.get_logger().warn(f'{process_name} 프로세스 강제 종료 (SIGKILL)...')
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except (ProcessLookupError, AttributeError):
                            process.kill()
                        process.wait()
                        self.get_logger().info(f'{process_name} 프로세스가 강제 종료되었습니다.')
            else:
                self.get_logger().info(f'{process_name} 프로세스는 이미 종료되었습니다. (exit code: {process.returncode})')
        except Exception as e:
            self.get_logger().error(f'{process_name} 프로세스 종료 중 오류 발생: {e}')
        finally:
            # 프로세스 참조 정리
            if process:
                try:
                    process.stdout.close()
                    process.stderr.close()
                except:
                    pass
    
    def _start_bash_a(self):
        """Bash A 스크립트 실행"""
        if not self.bash_a_path:
            self.get_logger().error('Bash A 경로가 설정되지 않았습니다.')
            return
        
        if not os.path.exists(self.bash_a_path):
            self.get_logger().error(f'Bash A 파일을 찾을 수 없습니다: {self.bash_a_path}')
            return
        
        if not os.access(self.bash_a_path, os.X_OK):
            self.get_logger().error(f'Bash A 파일에 실행 권한이 없습니다: {self.bash_a_path}')
            return
        
        try:
            self.get_logger().info(f'Bash A 실행 중: {self.bash_a_path}')
            
            # bash 스크립트 실행 (stdout/stderr를 파이프로 연결하여 로그 확인 가능)
            self.process_a = subprocess.Popen(
                ['bash', self.bash_a_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 새 프로세스 그룹 생성 (자식 프로세스까지 종료 가능)
            )
            
            self.get_logger().info(f'Bash A 프로세스가 시작되었습니다. (PID: {self.process_a.pid})')
            
            # 비동기로 stdout/stderr 읽기 (선택사항)
            threading.Thread(
                target=self._read_process_output,
                args=(self.process_a, 'Bash A'),
                daemon=True
            ).start()
            
        except Exception as e:
            self.get_logger().error(f'Bash A 실행 중 오류 발생: {e}')
            self.process_a = None
    
    def _start_bash_b(self):
        """Bash B 스크립트 실행"""
        if not self.bash_b_path:
            self.get_logger().error('Bash B 경로가 설정되지 않았습니다.')
            return
        
        if not os.path.exists(self.bash_b_path):
            self.get_logger().error(f'Bash B 파일을 찾을 수 없습니다: {self.bash_b_path}')
            return
        
        if not os.access(self.bash_b_path, os.X_OK):
            self.get_logger().error(f'Bash B 파일에 실행 권한이 없습니다: {self.bash_b_path}')
            return
        
        try:
            self.get_logger().info(f'Bash B 실행 중: {self.bash_b_path}')
            
            # bash 스크립트 실행
            self.process_b = subprocess.Popen(
                ['bash', self.bash_b_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # 새 프로세스 그룹 생성
            )
            
            self.get_logger().info(f'Bash B 프로세스가 시작되었습니다. (PID: {self.process_b.pid})')
            
            # 비동기로 stdout/stderr 읽기 (선택사항)
            threading.Thread(
                target=self._read_process_output,
                args=(self.process_b, 'Bash B'),
                daemon=True
            ).start()
            
        except Exception as e:
            self.get_logger().error(f'Bash B 실행 중 오류 발생: {e}')
            self.process_b = None
    
    def _read_process_output(self, process: subprocess.Popen, process_name: str):
        """프로세스 출력을 읽어서 로그로 출력 (비동기)"""
        try:
            # stdout 읽기
            if process.stdout:
                for line in iter(process.stdout.readline, b''):
                    if line:
                        self.get_logger().info(f'[{process_name} stdout] {line.decode("utf-8", errors="ignore").strip()}')
            
            # stderr 읽기
            if process.stderr:
                for line in iter(process.stderr.readline, b''):
                    if line:
                        self.get_logger().warn(f'[{process_name} stderr] {line.decode("utf-8", errors="ignore").strip()}')
        except Exception as e:
            self.get_logger().error(f'{process_name} 출력 읽기 중 오류: {e}')
    
    def _switch_to_bash_a(self):
        """B Bash 종료 후 A Bash 실행"""
        with self.process_lock:
            # B Bash 종료
            if self.process_b is not None:
                self._stop_process(self.process_b, 'Bash B')
                self.process_b = None
            
            # A Bash 실행
            if self.process_a is None or self.process_a.poll() is not None:
                self._start_bash_a()
            else:
                self.get_logger().info('Bash A가 이미 실행 중입니다.')
    
    def _switch_to_bash_b(self):
        """A Bash 종료 후 B Bash 실행"""
        with self.process_lock:
            # A Bash 종료
            if self.process_a is not None:
                self._stop_process(self.process_a, 'Bash A')
                self.process_a = None
            
            # B Bash 실행
            if self.process_b is None or self.process_b.poll() is not None:
                self._start_bash_b()
            else:
                self.get_logger().info('Bash B가 이미 실행 중입니다.')
    
    def destroy_node(self):
        """노드 종료 시 실행 중인 프로세스들 정리"""
        self.get_logger().info('노드 종료 중... 실행 중인 프로세스들을 정리합니다.')
        
        with self.process_lock:
            if self.process_a is not None:
                self._stop_process(self.process_a, 'Bash A')
                self.process_a = None
            
            if self.process_b is not None:
                self._stop_process(self.process_b, 'Bash B')
                self.process_b = None
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BashControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

