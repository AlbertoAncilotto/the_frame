import os
import cv2

class ImageWriter:
    def __init__(self, out_path='out'):
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path, exist_ok=True)

    def save_image(self, frame):
        path = self.out_path
        index = 1
        while os.path.exists(os.path.join(path, f"{index}.jpg")):
            index += 1
        filename = f"{index}.jpg"
        cv2.imwrite(os.path.join(path, filename), frame)