import cv2
import mediapipe as mp
import time

# Khởi tạo Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        t1 = time.time()
        success, image = cap.read()
        if not success:
            continue

        # Chuyển đổi hình ảnh sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Để cải thiện hiệu suất, có thể đánh dấu hình ảnh không thể sửa đổi
        image.flags.writeable = False
        results = hands.process(image)

        # Vẽ các kết quả lên hình ảnh gốc
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # In ra tọa độ của các điểm mốc
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                    # print(f'ID: {id}, X: {cx}, Y: {cy}, Z: {cz}')
        
        t2 = time.time()
        print("time = ", t2-t1)
        cv2.imshow('Hand Tracking', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
