import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from math import dist, sqrt, sin, cos
import robotArm
import numpy as np
from math import pi
import serial.tools.list_ports
import threading



delta_t = 0.1
filtered_signal = [0, 0, 0]
predict_vel = [0, 0, 0]
limit = [0.05, 0.05, 0.0]
check_time = time.time()

current_q_rad = np.array([0, pi/2, -pi/2, 0, 0, pi/2])
pre_q_rad = current_q_rad
# Define robot to compute kinematic and inverse kinematic
robot = robotArm.Arm()


# H(239, 0 ,258), A(299, 0, 93)
def path1(t):
    x_E = 239 + 20*t
    y_E = 0
    z_E = 258 - 55*t
    return x_E, y_E, z_E

# A(299, 0, 93), H(239, 0 ,258)
def path2(t):
    x_E = 299 - 20*t
    y_E = 0
    z_E = 96 + 55*t
    return x_E, y_E, z_E

# H(239, 0 ,258), H'(-239, 0 ,258)
def path3(t):
    x_E = 239*cos(pi*t/5)
    y_E = 239*sin(pi*t/5)
    z_E = 258 
    return x_E, y_E, z_E

# H'(-239, 0 ,258), B(-299, 0, 93)
def path4(t):
    x_E = -239 - 20*t
    y_E = 0
    z_E = 258 - 55*t
    return x_E, y_E, z_E

def path5(t):
    x_E = -299 + 20*t
    y_E = 0
    z_E = 99 + 55*t
    return x_E, y_E, z_E

def path6(t):
    x_E = -239*cos(pi*t/5)
    y_E = 239*sin(pi*t/5)
    z_E = 258 
    return x_E, y_E, z_E

serialInst = serial.Serial('COM4', 115200, timeout=.1)
# Send the angle to arduino

def send_command2():
    global current_q_rad, delta_t
    while True:
        current_q_degree = np.degrees(current_q_rad)
        print("q_6 = ", current_q_degree[5])
        # --------------------------------------------------------------
        #               "*** Limit of Input: 0-255 ***"
      
        mode = 1
        angle_int = [0.]*6
        angle_dcm = [0.]*6
        
        for i in range(6):
            angle_int[i] = current_q_degree[i] // 1
            angle_dcm[i] = round(current_q_degree[i]%1, 2) * 100
            
            
        arr_angle = bytearray([0x09,
                            mode,
                            int(angle_int[0]),
                            int(angle_dcm[0]),
                            int(angle_int[1]),
                            int(angle_dcm[1]),
                            int(angle_int[2] + 180),
                            int(angle_dcm[2]),
                            int(angle_int[3] + 45),
                            int(angle_dcm[3]),
                            int(angle_int[4] + 60),
                            int(angle_dcm[4]),
                            int(angle_int[5]),
                            int(angle_dcm[5]),
                            0xab])

        checksum = sum(arr_angle)
        # print("Sender   (Sum / SumDiv / SumMod) :", checksum, "/", checksum//256, "/", checksum%256)
        arr_angle.insert(14,checksum // 256)
        arr_angle.insert(15,checksum % 256)
   
        
        serialInst.write(arr_angle)
        time.sleep(delta_t)
        
def receive_command():
    while True:
        while serialInst.in_waiting > 0:
            packet = serialInst.readline()
            print(packet.decode().rstrip('\n'))
            
            
thread_send = threading.Thread(target=send_command2)
thread_send.start()

start_time = time.time()



time.sleep(3)
# H(239, 0 ,258), A(299, 0, 99)


for t in range (31):
    check_time = time.time()
    xE, yE, zE = path1(0.1 * t)

    position = np.array([[xE], [yE], [zE]])
   
    current_q_rad = robot.ik(X_0=position, q_=pre_q_rad)
    pre_q_rad = current_q_rad
    
    
    while time.time() - check_time < 0.1:
        pass

time.sleep(1)
current_q_rad[5] = 0
time.sleep(1)

for t in range (31):
    check_time = time.time()
    xE, yE, zE = path2(0.1 * t)
  
 
    current_q_rad = robot.ik(X_0=position, q_=pre_q_rad)
    pre_q_rad = current_q_rad
    
    current_q_degree = np.degrees(current_q_rad)
   
    while time.time() - check_time < 0.1:
        pass

time.sleep(1)
for t in range (51):
    check_time = time.time()
    xE, yE, zE = path3(0.1 * t)
  
    position = np.array([[xE], [yE], [zE]])
  
    current_q_rad = robot.ik(X_0=position, q_=pre_q_rad)
    pre_q_rad = current_q_rad
    

    while time.time() - check_time < 0.1:
        pass
    
for t in range (31):
    check_time = time.time()
    xE, yE, zE = path4(0.1 * t)

    position = np.array([[xE], [yE], [zE]])
 
    current_q_rad = robot.ik(X_0=position, q_=pre_q_rad)
    pre_q_rad = current_q_rad
    
    
    while time.time() - check_time < 0.1:
        pass
time.sleep(1)
current_q_rad[5] = pi/2
time.sleep(1)

for t in range (31):
    check_time = time.time()
    xE, yE, zE = path5(0.1 * t)

    position = np.array([[xE], [yE], [zE]])
 
    current_q_rad = robot.ik(X_0=position, q_=pre_q_rad)
    pre_q_rad = current_q_rad
    

    while time.time() - check_time < 0.1:
        pass
    
for t in range (51):
    check_time = time.time()
    xE, yE, zE = path6(0.1 * t)

    position = np.array([[xE], [yE], [zE]])
  
    current_q_rad = robot.ik(X_0=position, q_=pre_q_rad)
    pre_q_rad = current_q_rad
    
 
    while time.time() - check_time < 0.1:
        pass