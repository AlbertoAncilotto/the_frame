import cv2
import numpy as np

from camera import Camera
from keypad import GPIOPinReader
from style_transfer.style_transfer import StyleTransfer

from e00_starry_night import StarryNightSnap
from e01_warhol_monroe import WarholMonroeSnap
from e02_munch_scream import MunchScreamSnap
from e03_face_replace import FaceReplace

cv2.namedWindow('out')
# cv2.namedWindow('out', cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty('out',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

if __name__=='__main__':
    cam = Camera(None, 320, 480)
    gpio = GPIOPinReader()

    sn1 = StarryNightSnap(cam=cam, window_name='out')
    sn2 = StarryNightSnap(cam=cam, window_name='out', static_bg= cv2.imread('starry_night_clone_small.jpg'), invert_drawing=False)
    wm = WarholMonroeSnap(cam=cam, window_name='out')
    ms = MunchScreamSnap(cam=cam, window_name='out')
    fr1 = FaceReplace(cam=cam, window_name='out', static_bg='resources/mona_lisa.png', face_area=[0.27, 0.145, 0.61, 0.55])
    fr2 = FaceReplace(cam=cam, window_name='out', static_bg='resources/gogh_self_portrait.jpeg', style_path='style_transfer/gogh_bg_200x200.onnx', face_area=[0.28, 0.18, 0.63, 0.64])
   


    effects = [[sn1.starry_night_snap, sn2.starry_night_snap],[wm.monroe_snap, ms.munch_scream_snap],[fr1.face_replace_snap, fr2.face_replace_snap]]
    selected = [0,0,0]
    # sn.starry_night_snap()

    while True:
        frame = cam.get_frame()
        cv2.imshow('out', frame)
        key = gpio.waitKey(1)

        if key in [0,1,2]:
            effects[key][selected[key]]()
            selected[key]+=1
            selected[key]%=len(effects[key])
            print(selected)