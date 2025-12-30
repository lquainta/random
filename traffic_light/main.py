import cv2
import numpy as np
import warnings, os, time
from playsound import playsound

warnings.filterwarnings("ignore")

cap = cv2.VideoCapture(0)

last_color = None
red_start_time = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(BASE_DIR, 'sound.mp3')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 80, 80])
    upper_green = np.array([85, 255, 255])

    lower_red1 = np.array([0, 150, 200])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 150, 200])
    upper_red2 = np.array([179, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_green, mask_red)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None
    max_area = 0
    detected_color = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        circularity = area / circle_area if circle_area > 0 else 0

        if circularity < 0.6 or area < max_area:
            continue

        cx, cy = int(x), int(y)

        if mask_green[cy, cx] > 0:
            detected_color = "GREEN"
            color_draw = (0, 255, 0)
        elif mask_red[cy, cx] > 0:
            detected_color = "RED"
            color_draw = (0, 0, 255)
        else:
            continue

        best_circle = (cx, cy, int(radius), color_draw)
        max_area = area

    current_time = time.time()
    if detected_color == "RED":
        if red_start_time is None:
            red_start_time = current_time  # start counting
    else:
        if last_color == "RED" and red_start_time is not None:
            red_duration = current_time - red_start_time
            if red_duration >= 5:  # only play if red lasted ≥5 seconds
                playsound(sound_path)
        red_start_time = None  # reset timer after red ends

    last_color = detected_color

    if best_circle is not None:
        x, y, r, draw_color = best_circle
        cv2.circle(frame, (x, y), r, draw_color, 3)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        if detected_color is not None:
            print(f"{detected_color} traffic light detected")

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
