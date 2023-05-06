import cv2
import numpy as np

from camera import Camera
from keypad import GPIOPinReader
from style_transfer.style_transfer import StyleTransfer

from e00_starry_night import StarryNightSnap


cv2.namedWindow('out', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('out',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

if __name__=='__main__':
    cam = Camera(None, 640, 480)
    style_model = StyleTransfer('style_transfer/brush_300x480.onnx', preserve_color=True, alpha=0.5)
    gpio = GPIOPinReader()

    # sn = StarryNightSnap(cam=cam, window_name='out')
    sn = StarryNightSnap(cam=cam, window_name='out', static_bg= cv2.imread('starry_night_clone_small.jpg'), invert_drawing=False)

    effects = [[sn.starry_night_snap]]
    # sn.starry_night_snap()

    while True:
        frame = cam.get_frame()
        out_frame = style_model.transfer_style(frame)
        cv2.imshow('out', out_frame)
        key = gpio.waitKey(1)

        if key==0:
            effects[key][0]()