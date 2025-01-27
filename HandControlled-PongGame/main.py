import cv2
import mediapipe as mp
import numpy as np
import random

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

frame_width = 640
frame_height = 480

#left paddle
paddle1_width = 20
paddle1_height = 120
paddle1_color = (255, 0, 0)  # Blue
paddle1_x, paddle1_y = 20, (frame_height // 2) - (paddle1_height // 2)

#right paddle
paddle2_width = 20
paddle2_height = 120
paddle2_color = (0, 255, 0)  # Green
paddle2_x, paddle2_y = frame_width - 50, (frame_height // 2) - (paddle2_height // 2)

#Ball
ball_radius = 15
ball_color = (0, 0, 255)  # Red
ball_x, ball_y = frame_width // 2, frame_height // 2
ball_velocity = [random.choice([-6, 6]), random.choice([-6,6])]

with mp_hands.Hands(
    static_image_mode=False,                            # Real-time tracking
    max_num_hands=2,                                    # Max number of hands
    min_detection_confidence=0.5,                       # Minimum confidence for detection
    min_tracking_confidence=0.5                         # Minimum confidence for tracking
) as hands:

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (paddle1_x, paddle1_y),
                      (paddle1_x + paddle1_width, paddle1_y + paddle1_height),
                      paddle1_color, -1)
        cv2.rectangle(frame, (paddle2_x, paddle2_y),
                      (paddle2_x + paddle2_width, paddle2_y + paddle2_height),
                      paddle2_color, -1)

        cv2.circle(frame, (ball_x, ball_y), ball_radius, ball_color, -1)

        #Ball movement
        ball_x += ball_velocity[0]
        ball_y += ball_velocity[1]

        #Ball collision with top and bottom walls
        if ball_y <= ball_radius or ball_y >= frame_height - ball_radius:
            ball_velocity[1] = -ball_velocity[1]

        #Ball collision with paddles
        if (paddle1_x < ball_x - ball_radius < paddle1_x + paddle1_width and
                paddle1_y < ball_y < paddle1_y + paddle1_height) or \
            (paddle2_x < ball_x + ball_radius < paddle2_x + paddle2_width and
                paddle2_y < ball_y < paddle2_y + paddle2_height):
            ball_velocity[0] = int(ball_velocity[0] * 1.1)
            ball_velocity[1] = int(ball_velocity[1] * 1.1)      #for increaseing speed
            ball_velocity[0] = -ball_velocity[0]

        max_speed = 15
        ball_velocity[0] = max(-max_speed, min(max_speed, ball_velocity[0]))
        ball_velocity[1] = max(-max_speed, min(max_speed, ball_velocity[1]))

        #reset ball
        if ball_x < 0 or ball_x > frame_width:
            ball_x, ball_y = frame_width // 2, frame_height // 2
            ball_velocity = [random.choice([-6, 6]), random.choice([-6, 6])]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Controling paddles
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                x = int(hand_landmark.landmark[mp_hands.HandLandmark.WRIST].x * frame_width)
                y = int(hand_landmark.landmark[mp_hands.HandLandmark.WRIST].y * frame_height)

                #Left paddle control
                if x < frame_width // 2:
                    paddle1_y = max(0, min(frame_height - paddle1_height, y - paddle1_height // 2))

                #Right paddle control
                else:
                    paddle2_y = max(0, min(frame_height - paddle2_height, y - paddle2_height // 2))


                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),  # Landmarks style
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Connections style
                )

        cv2.imshow('Pong Game', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
