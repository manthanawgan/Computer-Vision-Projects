import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(
    statis_image_mode = False,                      # Real-time tracking
    max_num_hands = 2,                              #max number of hands
    min_detection_confidence = 0.5,                 #minimum confidence for detection
    min_tracking_confidence = 0.5                   #minimum confidence for detection
) as hands:

    while cam.isOpened():
        ref, frame = cam.read()

        cv2.imshow('frame feed',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = cv2.flip(frame, 1)  

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBB) 

        results = hands.process(rgb_frame) 


        if results.multi_hand_landmark:
            for hand_landmark in results.multi_hand_landmark:
                mp_drawing.draw_landmark(
                    frame, hand_landmark, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow(rgb_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cam.release()
cv2.destroyAllWindows()