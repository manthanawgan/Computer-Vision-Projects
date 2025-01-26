import cv2
import mediapipe as mp
import numpy as np

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


frame_width = 640
frame_height = 480

paddle1_width = 20
paddle1_height = 120
paddle1_color = (255, 0, 0)  # Blue
paddle1_x, paddle1_y = 20, (frame_height // 2) - (paddle1_height // 2)

# Paddle 2 properties (right paddle)
paddle2_width = 20
paddle2_height = 120
paddle2_color = (0, 255, 0)  # Green
paddle2_x, paddle2_y = frame_width - 50, (frame_height // 2) - (paddle2_height // 2)



ball_radius = 15
ball_color = (0,0,255) #red

ball_x, ball_y = frame_width // 2, frame_height // 2
#create blank frames 
#frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

'''#drawing paddle
cv2.rectangle(frame, (paddle1_x,paddle1_y), (paddle1_x + paddle_width, paddle1_y + paddle_height), paddle_color, -1)
cv2.rectangle(frame, (paddle2_x,paddle2_y), (paddle1_x + paddle_width, paddle2_y + paddle_height), paddle_color, -1)

#drawing ball
cv2.circle(frame,(ball_x,ball_y), ball_radius, ball_color, -1)
        
cv2.imshow('pong game frame', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

with mp_hands.Hands(
        static_image_mode = False,                      # Real-time tracking
        max_num_hands = 4,                              #max number of hands
        min_detection_confidence = 0.5,                 #minimum confidence for detection
        min_tracking_confidence = 0.5                   #minimum confidence for detection
    ) as hands:

        while cam.isOpened():
            ret, frame = cam.read()

            cv2.imshow('frame feed',frame)

            frame = cv2.flip(frame, 1)  
            # Draw paddle 1
            cv2.rectangle(frame, (paddle1_x, paddle1_y), 
                        (paddle1_x + paddle1_width, paddle1_y + paddle1_height), 
                        paddle1_color, -1)

            # Draw paddle 2
            cv2.rectangle(frame, (paddle2_x, paddle2_y), 
                        (paddle2_x + paddle2_width, paddle2_y + paddle2_height), 
                        paddle2_color, -1)

            cv2.circle(frame, (ball_x, ball_y), ball_radius, ball_color, -1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

            results = hands.process(rgb_frame) 

            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmark, mp_hands.HAND_CONNECTIONS
                    )



            cv2.imshow('frame feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


cam.release()
cv2.destroyAllWindows()
