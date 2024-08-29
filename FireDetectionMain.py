from ultralytics import YOLO
import cvzone
import cv2
import math
import pygame
import time
import threading

# Initialize pygame mixer
pygame.mixer.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound('208. Fire siren - sound effect.mp3')

# Function to play the alarm sound in a loop
def play_alarm():
    while fire_detected:  # Continue playing while fire is detected
        if not pygame.mixer.get_busy():
            alarm_sound.play()

# Function to stop the alarm sound
def stop_alarm():
    pygame.mixer.stop()

# Running real-time from webcam
cap = cv2.VideoCapture(1) # 1 is for internal webcam and 0 external webcam
model = YOLO('best.pt')

# Reading the classes
classnames = ['fire']

# Additional points to reduce false positives
confidence_threshold = 80  # Confidence threshold for fire detection
min_box_area = 5000   # Minimum area of bounding boxes to consider as fire

# Initialize timer variable
fire_detected_time = None
fire_duration_threshold = 0 # seconds
fire_detected = False
alarm_thread = None

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    fire_detected_in_frame = False  # Flag to check if fire is detected in the current frame

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            box_area = (x2 - x1) * (y2 - y1)

            # Debugging information
            detection_info = f"Detected: {classnames[Class]} | Confidence: {confidence}% | Area: {box_area}"
            print(detection_info)  # Print to console
            cv2.putText(frame, detection_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if confidence > confidence_threshold and box_area > min_box_area:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

                fire_detected_in_frame = True  # Set flag if fire is detected

    # Handling fire detection duration
    if fire_detected_in_frame:
        if fire_detected_time is None:  # If fire detected for the first time
            fire_detected_time = time.time()
        elif time.time() - fire_detected_time >= fire_duration_threshold:
            # Fire detected continuously for the threshold duration
            if not fire_detected:
                fire_detected = True
                cv2.putText(frame, "Fire detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("Fire detected!")  # Debug print
                alarm_thread = threading.Thread(target=play_alarm)
                alarm_thread.start()
    else:
        fire_detected_time = None  # Reset the timer if no fire detected
        if fire_detected:
            fire_detected = False
            print("No fire detected, stopping alarm.")  # Debug print
            cv2.putText(frame, "No fire detected, stopping alarm.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            stop_alarm()  # Stop the sound immediately when fire is no longer detected

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
