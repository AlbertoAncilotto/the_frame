import cv2
import numpy as np
from keypad import GPIOPinReader
import os


class MemesReel:
    def __init__(self, height=480, width=320, window_name='out', folder = 'resources/memes'):
        self.width = width
        self.height = height
        self.gpio = GPIOPinReader()
        self.folder = folder
        self.window_name = window_name

    def resize_image(self, image):
        h, w = image.shape[:2]
        ar = w / float(h)
        nw = self.width
        nh = int(nw / ar)
        if nh > self.height:
            nh = self.height
            nw = int(nh * ar)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        top = (self.height - nh) // 2
        bottom = self.height - nh - top
        left = (self.width - nw) // 2
        right = self.width - nw - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return image
    
    def memes_reel(self):
        for filename in os.listdir(self.folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image = cv2.imread(os.path.join(self.folder, filename))
                image = self.resize_image(image)
                cv2.imshow(self.window_name, image)
                key = self.gpio.waitKey(10000)
                if key == 1 or key == 2:
                    return
        

if __name__=='__main__':
    sn = MemesReel()
    sn.memes_reel()