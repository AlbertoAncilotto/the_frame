import cv2
import numpy as np
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Define font and colors for countdown timer
font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 12
font_thickness = 10
color = (255, 255, 255)

# Define duration of countdown timer and flash
countdown_duration = 3  # seconds
flash_duration = 10  # seconds

# Initialize timer and flag for countdown and flash
timer_start = None
countdown_flag = False
flash_flag = False

while True:
    # Capture frame from camera
    ret, frame = cap.read()

    # Check if frame is valid
    if not ret:
        break

    # Check if 'P' key is pressed to start countdown and flash
    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        timer_start = time.time()
        countdown_flag = True

    # Check if countdown is active
    if countdown_flag:
        # Calculate time left for countdown
        time_left = countdown_duration - int(time.time() - timer_start)

        # Check if countdown is done
        if time_left <= 0:
            countdown_flag = False
            flash_flag = True

        # Draw current number on image
        text_size, _ = cv2.getTextSize(str(time_left), font, font_scale, font_thickness)
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = int((frame.shape[0] + text_size[1]) / 2)
        cv2.putText(frame, str(time_left), (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    # Check if flash is active
    if flash_flag:
        # Draw white rectangle over image
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)

        # Check if flash is done
        if time.time() - timer_start >= flash_duration:
            flash_flag = False

    # Display frame on screen
    cv2.imshow('Camera', frame)

    # Check if 'Q' key is pressed to quit
    if key == ord('q'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()