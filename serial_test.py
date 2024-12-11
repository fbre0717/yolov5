import serial
import time

py_serial = serial.Serial(
    port = '/dev/ttyACM0',
    baudrate=57600,
)

try:
    command = 'r'
    py_serial.write(command.encode('utf-8'))
    print("전송된 명령:", command)
    
    # 타임아웃 설정
    start_time = time.time()
    timeout = 10  # 10초 타임아웃
    
    while True:
        if py_serial.in_waiting > 0:
            response = py_serial.readline().decode('utf-8').strip()
            print("로봇으로부터 받은 응답:", response)
            break
            
        # 타임아웃 체크
        if time.time() - start_time > timeout:
            print("타임아웃: 응답 대기 시간 초과")
            break
            
        time.sleep(0.1)

except serial.SerialException as e:
    print("시리얼 통신 에러:", e)

finally:
    py_serial.close()