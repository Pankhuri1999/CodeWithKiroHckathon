"""
Real-Time Hindi Sign Language Prediction with Mediapipe Skeleton
----------------------------------------------------------------
- Opens webcam with ROI
- Detects hand using MediaPipe
- Draws skeleton with blue lines + green points
- Real-time prediction using trained CNN
- Stabilizes prediction using buffer
- Displays Hindi + English transliteration + confidence
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from collections import deque

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:/Users/pankh/cnn_model99.h5"
FONT_PATH = r"C:/Users/pankh/NotoSansDevanagari-Regular.ttf"
ROI_WIDTH, ROI_HEIGHT = 300, 300  # width, height of ROI
TOP_MARGIN = 100  # distance from top
BUFFER_SIZE = 5   # number of frames to smooth prediction

# ---------------- Map index → Hindi + transliteration ----------------
lettersMap = {
    0: ('_', '_'), 1: ('अं', 'aM'), 2: ('अः', 'aH'), 3: ('आ', 'aa'), 4: ('इ', 'i'), 5: ('ई', 'ii'), 6: ('उ', 'u'),
    7: ('ऊ', 'uu'), 8: ('ऋ', 'Ri'), 9: ('ए', 'e'), 10: ('ऐ', 'ai'), 11: ('ओ', 'o'), 12: ('औ', 'au'),
    13: ('क', 'k'), 14: ('क्ष', 'ksh'), 15: ('ख', 'kh'), 16: ('ग', 'g'), 17: ('घ', 'gh'), 18: ('ङ', 'ng'),
    19: ('च', 'ch'), 20: ('छ', 'chh'), 21: ('ज', 'j'), 22: ('ज्ञ', 'gy'), 23: ('झ', 'jh'), 24: ('ञ', 'ny'),
    25: ('ट', 'T'), 26: ('ठ', 'Th'), 27: ('ड', 'D'), 28: ('ढ', 'Dh'), 29: ('ण', 'N'), 30: ('त', 't'),
    31: ('त्र', 'tr'), 32: ('थ', 'th'), 33: ('द', 'd'), 34: ('ध', 'dh'), 35: ('न', 'n'), 36: ('प', 'p'),
    37: ('फ', 'ph'), 38: ('ब', 'b'), 39: ('भ', 'bh'), 40: ('म', 'm'), 41: ('य', 'y'), 42: ('र', 'r'),
    43: ('ल', 'l'), 44: ('व', 'v'), 45: ('श', 'sh'), 46: ('ष', 'Sh'), 47: ('स', 's'), 48: ('ह', 'h')
}

# ---------------- Load model ----------------
print("🔹 Loading model...")
model = load_model(MODEL_PATH)
_, H, W, C = model.input_shape
print(f"✅ Model loaded. Input shape: {(H, W, C)}")

# ---------------- Prediction buffer ----------------
pred_buffer = deque(maxlen=BUFFER_SIZE)

# ---------------- Mediapipe setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# Draw skeleton with blue lines + green points
def draw_skeleton(roi, hand_landmarks):
    h, w = roi.shape[:2]
    out = np.zeros_like(roi)
    pts = []
    for lm in hand_landmarks.landmark:
        x_px, y_px = int(lm.x * w), int(lm.y * h)
        pts.append((x_px, y_px))
    # Draw connections
    for c in mp_hands.HAND_CONNECTIONS:
        cv2.line(out, pts[c[0]], pts[c[1]], (255, 0, 0), 2)
    # Draw points
    for (x, y) in pts:
        cv2.circle(out, (x, y), 4, (0, 255, 0), -1)
    return out

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("❌ Cannot open webcam")

font = ImageFont.truetype(FONT_PATH, 32)
print("📷 Webcam ready. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    # Shift ROI to upper-right
    x = frame_w - ROI_WIDTH - 10  # 10 px from right edge
    y = TOP_MARGIN
    w, h = ROI_WIDTH, ROI_HEIGHT

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi = frame[y:y+h, x:x+w]
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_roi)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        skel_img = draw_skeleton(roi, hand_landmarks)

        # Preprocess for CNN
        if C == 1:
            gray = cv2.cvtColor(skel_img, cv2.COLOR_BGR2GRAY)
            input_img = cv2.resize(gray, (W,H)).reshape(1,H,W,1)/255.0
        else:
            input_img = cv2.resize(skel_img, (W,H)).reshape(1,H,W,3)/255.0

        # Predict
        preds = model.predict(input_img, verbose=0)
        idx = int(np.argmax(preds[0]))
        pred_buffer.append(idx)
        most_common = max(set(pred_buffer), key=pred_buffer.count)
        hindi, translit = lettersMap.get(most_common, (f"Class {most_common}", f"Class {most_common}"))
        conf = float(np.max(preds[0]))

        # Overlay text
        img_pil = Image.fromarray(skel_img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f"{hindi} ({translit}) [{conf:.2f}]", font=font, fill=(0,0,255))
        skel_img = np.array(img_pil)
        frame[y:y+h, x:x+w] = skel_img
    else:
        pred_buffer.clear()

    cv2.imshow("Hindi Sign Language Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
