import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from math import dist
import robotArm
import numpy as np
from math import pi
import serial.tools.list_ports
import threading
import pygame

sound_path = ['./sound/Start_sound.mp3', 
              './sound/End_sound.mp3']

# initial sound_effect
pygame.init()
pygame.mixer.init()


delta_t = 0.1
filtered_signal = [0, 0, 0]
predict_vel = [0, 0, 0]
limit = [0.05, 0.05, 0.03]
check_time = time.time()

current_q_rad = np.array([pi/2, pi/2, -pi/2, 0, 0, 0])
pre_q_rad = current_q_rad
# Define robot to compute kinematic and inverse kinematic
robot = robotArm.Arm()
pre_position = robot.fk(pi/2, pi/2, -pi/2, 0, 0)


# Define the alpha-beta filter function
def alpha_beta_filter(signal, order):
    alpha, beta = 0.5, 0.5
    if order == 2:
        alpha, beta = 0.5, 0.5
        
    global filtered_signal, predict_vel
    pre_signal = filtered_signal[order]
    velocity = predict_vel[order]
    predicted_signal = pre_signal + velocity * delta_t   
    filtered_signal[order] = predicted_signal + alpha * (signal - predicted_signal)
    velocity = velocity + beta * ((signal - predicted_signal) / delta_t)
    
    if abs(filtered_signal[order]) < limit[order]:
        filtered_signal[order] = 0
    return filtered_signal[order]

# Compute the landmark of finger
def get_landmark(result, idx):
    hand_landmarks = results.multi_hand_world_landmarks[0]
    hand_landmark = hand_landmarks.landmark[idx]
    x = hand_landmark.x * 480 / 100
    y = hand_landmark.y * 640 / 100
    z = hand_landmark.z * 10
    return x, y, -z

pre_active_mode = False
active_mode = False
ACTIVE_COUNT = 0
# Active mode of robot

def is_active( y_8, y_12, y_16, y_20, offset):
    global active_mode
    global ACTIVE_COUNT
        
    change_mode = None
    if y_8 < offset and y_12 > offset and y_16 > offset and y_20 > offset:
        change_mode = True
    else:
        change_mode = False
    
    if active_mode == change_mode:
        ACTIVE_COUNT = 0
    else: 
        ACTIVE_COUNT += 1
    
    if ACTIVE_COUNT == 10:
        active_mode = not active_mode 
    
    return active_mode

# Gripper mode
# def is_gripper(distance):
#     if distance < 0.4:
#         return True
#     return False

# Gripper mode
gripper_mode = False
GRIPPER_COUNT = 0
def is_gripper(distance):
    global gripper_mode
    global GRIPPER_COUNT
    change_mode = None
    if distance < 0.4:
        change_mode = True
    else:
        change_mode = False
        
    if gripper_mode == change_mode:
        GRIPPER_COUNT = 0
    else: 
        GRIPPER_COUNT += 1
    
    if GRIPPER_COUNT == 10:
        gripper_mode = not gripper_mode 
 
    return gripper_mode

# Compute the direction and velocity of hand movement 
def compute_dir(delta_x, delta_y, delta_z):
    x_dir = "Same" if delta_x == 0 else ("Right" if delta_x > 0 else "Left")
    y_dir = "Same" if delta_y == 0 else ("Down" if delta_y > 0 else "Up" )
    z_dir = "Same" if delta_z == 0 else ("Out" if delta_z > 0 else "In" )
    
    t = 60
    d_x_norm = int(delta_x*t)
    d_y_norm = int(delta_y*t)
    d_z_norm = int(delta_z*t*5)
    
    min_value , max_value = -100, 100
    d_x_norm = max(min(d_x_norm, max_value), min_value)
    d_y_norm = max(min(d_y_norm, max_value), min_value)
    d_z_norm = max(min(d_z_norm, max_value), min_value)
    
    result = [[x_dir, d_x_norm], [y_dir, d_y_norm], [z_dir, d_z_norm]]
    
    return result


def layout(image, gripper_mode = None, dir = None):
    global active_mode, pre_active_mode

    borded_img = image
    top, bottom, left, right = 10, 10, 10, 10 # size of the border
    
    if active_mode == True and pre_active_mode == False:      
        pygame.mixer.music.load(sound_path[0])
        pygame.mixer.music.play(0)      
    elif active_mode == False and pre_active_mode == True:
        pygame.mixer.music.load(sound_path[1])
        pygame.mixer.music.play(0)
          
    if active_mode:
        borded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (0, 255, 0))
    else:
        borded_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (0, 0, 255))
    
    if gripper_mode:
        logo = cv2.imread('.\images\Gripper.png') 
        size = 50
        logo = cv2.resize(logo, (size, size)) 

        # Create a mask of logo 
        img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY) 
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY) 
        roi = borded_img[-size-10:-10, -size-10:-10] 
        roi[np.where(mask)] = 0
        roi += logo 
   
    return borded_img


serialInst = serial.Serial('COM4', 115200, timeout=.1)
# Send the angle to arduino

def send_command2():
    global current_q_rad, delta_t
    while True:
        current_q_degree = np.degrees(current_q_rad)
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
            
            
# Initialize MediaPipe Hand Gesture Recognizer
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from the webcam
cap = cv2.VideoCapture(0)
iterator = 0


raw_signal_x = []
filtered_signal_x = []

pre_x = 0.
pre_y = 0.
pre_z = 0.


thread_send = threading.Thread(target=send_command2)
# thread_recv = threading.Thread(target=receive_command)
thread_send.start()
# thread_recv.start()

start_time = time.time()


while cap.isOpened():
    
    t0 = time.time()
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    x_shape, y_shape, c = frame.shape
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=rgb_frame
    )
    
    # Check if the time_poin equal to 0.01 or not
    while time.time() < check_time + delta_t:
        pass
    # Update check_time
    check_time = time.time()

    # Process the frame with MediaPipe Hand Gesture Recognizer
    results = hands.process(image=rgb_frame)

 
    # If hand landmarks are detected, iterate through them
    if results.multi_hand_landmarks:    
    
        x_4, y_4, z_4 = get_landmark(results, 4) # Thumb tip
        x_8, y_8, z_8 = get_landmark(results, 8) # Index finger tip
        _, y_12, _ = get_landmark(results, 12) # Middle finger tip
        _, y_16, _ = get_landmark(results, 16) # Ring finger tip
        _, y_20, _ = get_landmark(results, 20) # Ring finger tip
        _, offset, _ = get_landmark(results, 10) # Ring finger tip 
        x_0, y_0, z_0 = get_landmark(results, 8)
        
        
        # Check if the gesture is working or not 
        active_mode = is_active(y_8, y_12, y_16, y_20, offset)
        # print("Active_mode: ", active_mode)
        
        # delta_x, delta_y, delta_z = x_8 - pre_x, y_8 - pre_y, z_8 - pre_z
        delta_x, delta_y, delta_z = x_0 - pre_x, y_0 - pre_y, z_0 - pre_z

        # filter the signal
        delta_x_filterd = alpha_beta_filter(delta_x, 0)
        delta_y_filterd = alpha_beta_filter(delta_y, 1)
        delta_z_filterd = alpha_beta_filter(delta_z, 2)
        print("z_0 = ", z_0)
        print("delta_x_filter = ", delta_x)
        print("delta_y_filter = ", delta_y)
        print("delta_z_filter = ", delta_z)
        print("active_mode = ", active_mode)
        print("\n")
        dir = compute_dir(delta_x_filterd, delta_y_filterd, delta_z_filterd)
       
        delta_position = np.array([dir[0][1], -dir[1][1], -dir[2][1]])

        # If active_mode is true, update the postition
        if active_mode:
            
            current_position = pre_position + delta_position 
            position = np.array([[current_position[0]], [current_position[2]], [current_position[1]]])
            current_q_rad = robot.ik(X_0=position, q_=pre_q_rad)
            
            # Check the gripper working or not 
            grip_distance = dist((x_4, y_4, z_4), (x_8, y_8, z_8))
            gripper_mode = is_gripper(grip_distance)
            current_q_rad[5] = 0 if gripper_mode == True else pi/2
            
            pre_position = current_position 
            pre_q_rad = current_q_rad

        
        # Update the position
        pre_x, pre_y, pre_z = x_0, y_0, z_0 
        # pre_x, pre_y, pre_z = x_8, y_8, z_8 

    frame = layout(image=frame)
    # Show the frame with the landmarks
    cv2.imshow('Hand Landmarks', frame)
    
    # Update the active mode
    pre_active_mode = active_mode
    
    iterator = iterator + 1
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        end_time = time.time()
        break
    

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()


elapsed_time = end_time - start_time
period = elapsed_time / (iterator-1)
print("Period: ", period)