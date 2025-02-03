import cv2
import mediapipe as mp
import numpy as np
import pygame
import math

pygame.mixer.init()

sounds = {
    4: pygame.mixer.Sound(r"D:\coding\projects\Opencv-Visualization-Projects\Musical-Fingering\classic.mp3"),    # Thumb → Piano
    8: pygame.mixer.Sound(r"D:\coding\projects\Opencv-Visualization-Projects\Musical-Fingering\a_LWOKJGC.mp3"),    # Index → Drums
    12: pygame.mixer.Sound(r"D:\coding\projects\Opencv-Visualization-Projects\Musical-Fingering\cantina-band-flute-cover-10_5i81c11.mp3"),   # Middle → Flute
    16: pygame.mixer.Sound(r"D:\coding\projects\Opencv-Visualization-Projects\Musical-Fingering\sad-violin-15.mp3"),  # Ring → Violin
    20: pygame.mixer.Sound(r"D:\coding\projects\Opencv-Visualization-Projects\Musical-Fingering\synth-pad.mp3")    # Pinky → Synth
}

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

fingertip_ids = [4,8,12,16,20]
prev_position = {id: (0,0) for id in fingertip_ids}
movement_threshold = 10

mp_draw = mp.solutions.drawing_utils

if not cam.isOpened():
    print("cannot open the cameras")
    exit()
while True:
    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id in fingertip_ids:
                landmark = hand_landmark.landmark[id]
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                px,py = prev_position[id]

                movement = math.sqrt((cy - px) ** 2 + (cy - py) ** 2)

                pitch = max(0.5, 2.0 - (cy / h) * 1.5)

                volume = np.interp(cx, [0, w], [0.1,0.1])

                if movement > movement_threshold:
                    sound = sounds[id]
                    sound.set_volume(volume)
                    sound.play()

                    print(f"finger {id} - pitch: {pitch:.2f}, volume: {volume:.2f}, speed:{movement:.2f}")

                    prev_position[id] = (cx,cy)

                cv2.circle(frame,(cx,cy), 10, (0,255,0), -1)

    cv2.imshow('hand music', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
