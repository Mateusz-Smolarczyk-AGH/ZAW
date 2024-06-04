import cv2
import numpy as np
# from tensorflow.keras.models import load_model

cap = cv2.VideoCapture('grupowe/video_1.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

def segment_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definiowanie zakresów dla czerwonych piłek
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    
    # Definiowanie zakresu dla żółtych piłek
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Maski dla czerwonych i żółtych kolorów
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    return mask_red, mask_yellow

def detect_balls(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    
    for contour in contours:
        # Sprawdzamy, czy kontur odpowiada kształtowi piłki (przybliżony do okręgu)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 5:  # Ignorujemy małe kontury
            balls.append((int(x), int(y), int(radius)))
    
    return balls
cnt=0
while cap.isOpened():
    cnt+=1
    ret, frame = cap.read()
    if not ret:
        break
    
    mask_red, mask_yellow = segment_colors(frame)
    kernel = np.ones((5,5),np.uint8)
    cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

    red_balls = detect_balls(mask_red)
    yellow_balls = detect_balls(mask_yellow)
    cv2.imshow('mask_red', mask_red)
    cv2.imshow('mask_yellow', mask_yellow)

    for (x, y, radius) in red_balls:
        cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
    
    for (x, y, radius) in yellow_balls:
        cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)
    
    cv2.imshow('Frame', frame)
    cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
print(cnt)
cv2.destroyAllWindows()
