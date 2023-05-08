import cv2
import time

class SnapCamera:
    def __init__(self, countdown_duration=3):

        # Define font and colors for countdown timer
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 12
        self.font_thickness = 10
        self.color = (255, 255, 255)

        # Define duration of countdown timer and flash
        self.countdown_duration = countdown_duration  # seconds

        # Initialize timer and flag for countdown and flash
        self.timer_start = None
        self.countdown_flag = False
        self.flash_flag = False
        self.time_left = None

    def start_snap(self):
        self.timer_start = time.time()
        self.countdown_flag = True
        self.flash_flag = False
        return self.countdown_duration

    def snap(self, frame):
        # Check if frame is valid
        if frame is None:
            return None, None

        # Check if countdown is active
        if self.countdown_flag:
            # Calculate time left for countdown
            self.time_left = self.countdown_duration - int(time.time() - self.timer_start)

            # Check if countdown is done
            if self.time_left <= 0:
                self.countdown_flag = False
                self.flash_flag = True

            # Draw current number on image
            text_size, _ = cv2.getTextSize(str(self.time_left), self.font, self.font_scale, self.font_thickness)
            text_x = int((frame.shape[1] - text_size[0]) / 2)
            text_y = int((frame.shape[0] + text_size[1]) / 2)
            cv2.putText(frame, str(self.time_left), (text_x, text_y), self.font, self.font_scale, self.color, self.font_thickness, cv2.LINE_AA)

        # Check if flash is active
        if self.flash_flag:
            # Draw white rectangle over image
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)

        return frame, self.time_left
