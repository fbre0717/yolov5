import serial
import time


py_serial = serial.Serial(  port = '/dev/ttyACM0',  baudrate=57600)


# right_command = 'r'
# py_serial.write(right_command.encode('utf-8'))
# print(right_command)
# stop_command = 's'
# py_serial.write(right_command.encode('utf-8'))
# print(stop_command)
# time.sleep(100)

# command = 'r'
# py_serial.write(command.encode('utf-8'))
# print(command)

command = 'r'
py_serial.write(command.encode('utf-8'))
print(command)

time.sleep(5)

# command = 'sdw'
# py_serial.write(command.encode('utf-8'))
# print(command)

# time.sleep(100)