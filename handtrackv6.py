"""
Hand Mouse Control
- Click: pollice + indice < 0.09
- Drag: click mantenuto > 1 sec
- Raggio di azione limitato per raggiungere estremi dello schermo
- Visualizza landmarks e scritte sullo schermo
- Sempre on top
"""

import cv2
import mediapipe as mp
import pyautogui
import math
import ctypes
import time

# Parametri
SMOOTHING = 0.15
CLICK_DIST_THRESH = 0.09
DRAG_HOLD_TIME = 1.0   # secondi per considerare drag
FRAME_MARGIN = 0.2      # margine della webcam per mapping (0.2 = 20%)

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Schermo
SCREEN_W, SCREEN_H = pyautogui.size()
prev_x, prev_y = 0, 0
dragging = False
click_start_time = None

def norm_distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def set_window_topmost(window_name='Hand Mouse'):
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd:
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0,0,0,0, 3)

def main():
    global prev_x, prev_y, dragging, click_start_time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        last_click_time = 0
        click_cooldown = 0.25

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            # sfondo scritte
            cv2.rectangle(frame, (0,0), (250,80), (0,0,0), -1)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                lm = hand.landmark

                index_tip = (lm[8].x, lm[8].y)
                thumb_tip = (lm[4].x, lm[4].y)

                # Limita raggio d'azione
                ix = max(FRAME_MARGIN, min(1-FRAME_MARGIN, index_tip[0]))
                iy = max(FRAME_MARGIN, min(1-FRAME_MARGIN, index_tip[1]))

                # Mapping ai pixel dello schermo
                norm_x = (ix - FRAME_MARGIN) / (1 - 2*FRAME_MARGIN)
                norm_y = (iy - FRAME_MARGIN) / (1 - 2*FRAME_MARGIN)
                screen_x = SCREEN_W * norm_x
                screen_y = SCREEN_H * norm_y

                # Smoothing
                smooth_x = prev_x + SMOOTHING * (screen_x - prev_x)
                smooth_y = prev_y + SMOOTHING * (screen_y - prev_y)
                prev_x, prev_y = smooth_x, smooth_y

                pyautogui.moveTo(smooth_x, smooth_y, duration=0)

                # Click pollice + indice
                d_thumb_index = norm_distance(index_tip, thumb_tip)
                now = time.time()
                click_text = ""
                if d_thumb_index < CLICK_DIST_THRESH:
                    if click_start_time is None:
                        click_start_time = now
                    elif not dragging and now - click_start_time >= DRAG_HOLD_TIME:
                        pyautogui.mouseDown()
                        dragging = True
                        click_text = "DRAG"
                    # click singolo
                    if not dragging and now - last_click_time > click_cooldown:
                        last_click_time = now
                        pyautogui.click()
                        click_text = "CLICK"
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    click_start_time = None

                # Landmark e testo
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (int(index_tip[0]*w), int(index_tip[1]*h)), 8, (255,0,0), -1)
                cv2.putText(frame, f"D:{d_thumb_index:.2f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
                if click_text:
                    cv2.putText(frame, click_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            cv2.imshow('Hand Mouse', frame)
            set_window_topmost('Hand Mouse')

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if dragging:
                    pyautogui.mouseUp()
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
