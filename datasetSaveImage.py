import cv2
import mediapipe as mp
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create dataset folder
dataset_dir = "C:/Users/pankh/OneDrive/Documents/sign_skeleton_dataset/dataset"
os.makedirs(dataset_dir, exist_ok=True)

# ROI parameters (x,y,w,h)
ROI_X, ROI_Y, ROI_W, ROI_H = 100, 100, 300, 300

# Webcam
cap = cv2.VideoCapture(0)

# Counter and flag
img_count = 0
save_requested = False

# Tkinter setup
root = tk.Tk()
root.title("Hand ROI Capture")

# Tkinter Frame for video
video_label = ttk.Label(root)
video_label.pack()

# Function to save image when button is clicked
def save_image():
    global save_requested
    save_requested = True

# Save button
save_btn = ttk.Button(root, text="Save ROI Image", command=save_image)
save_btn.pack(pady=10)

# Update loop for video feed
def update_frame():
    global img_count, save_requested

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Draw ROI rectangle
    cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), (0, 255, 0), 2)

    # Extract ROI
    roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

    # Process with Mediapipe
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                roi,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

        if save_requested:
            img_path = os.path.join(dataset_dir, f"roi_{img_count}.jpg")
            cv2.imwrite(img_path, roi)
            print(f"Saved {img_path}")
            img_count += 1
            save_requested = False

    # Convert image for Tkinter
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, update_frame)

# Close the app gracefully
def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start updating frames
update_frame()
root.mainloop()
