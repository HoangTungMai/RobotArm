import cv2
import mediapipe as mp

from mediapipe.python.solutions.hands_connections import HAND_PALM_CONNECTIONS
from mediapipe.python.solutions.hands_connections import HAND_THUMB_CONNECTIONS
from mediapipe.python.solutions.hands_connections import HAND_INDEX_FINGER_CONNECTIONS
from mediapipe.python.solutions.hands_connections import HAND_MIDDLE_FINGER_CONNECTIONS
from mediapipe.python.solutions.hands_connections import HAND_RING_FINGER_CONNECTIONS
from mediapipe.python.solutions.hands_connections import HAND_PINKY_FINGER_CONNECTIONS

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def alpha_beta(signal):
    pass

ax = plt.axes(projection='3d', proj_type = 'ortho')
ax.view_init(0, 0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

use_webcam = True

cap = cv2.VideoCapture(0 if use_webcam else r"sample.mp4")

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    def animate(frame):
        plt.cla()

        ax.set_xlabel('x')
        # swap z and y for visualization
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.axes.set_xlim3d(left=0.0, right=0.5)
        ax.axes.set_ylim3d(bottom=0.25, top=-0.25)
        ax.axes.set_zlim3d(bottom=1, top=0.0)

        success, image = cap.read()
        if not success:
            if use_webcam:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                sys.exit(1)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return

        y_scale = float(image.shape[0]) / image.shape[1]

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                def draw_hand_lines(connecions):
                    x = []
                    y = []
                    z = []
                    for pair in connecions:
                        x.append(hand_landmarks.landmark[pair[0]].x)
                        x.append(hand_landmarks.landmark[pair[1]].x)
                        # Note y and z are swapped for visualization
                        y.append(hand_landmarks.landmark[pair[0]].z)
                        y.append(hand_landmarks.landmark[pair[1]].z)
                        z.append(hand_landmarks.landmark[pair[0]].y * y_scale)
                        z.append(hand_landmarks.landmark[pair[1]].y * y_scale)
                        print(pair)
                        print(hand_landmarks.landmark[pair[0]].z)
                        print('\n')

                    ax.scatter3D(np.array(x), np.array(y), np.array(z), s = 100)
                    ax.plot(np.array(x), np.array(y), np.array(z))

                draw_hand_lines(HAND_PALM_CONNECTIONS)
                draw_hand_lines(HAND_THUMB_CONNECTIONS)
                draw_hand_lines(HAND_INDEX_FINGER_CONNECTIONS)
                draw_hand_lines(HAND_MIDDLE_FINGER_CONNECTIONS)
                draw_hand_lines(HAND_RING_FINGER_CONNECTIONS)
                draw_hand_lines(HAND_PINKY_FINGER_CONNECTIONS)

                #plt.savefig(os.path.join('imgs', str(frame).zfill(3) +'.png'))
                #cv2.imwrite(os.path.join('imgs', str(frame).zfill(3) +'.png'), image)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            sys.exit(0)

    ani = FuncAnimation(plt.gcf(), animate, interval=33)

    plt.tight_layout()
    plt.show()



def alpha_beta(singal):
    return 0
