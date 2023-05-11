import cv2
import numpy as np

from camera import Camera
from keypad import GPIOPinReader

from e00_starry_night import StarryNightSnap
from e01_warhol_monroe import WarholMonroeSnap
from e02_munch_scream import MunchScreamSnap
from e03_face_replace import FaceReplace
from e04_face_replace_multi import FaceReplaceMulti
from e05_style_multi import StyleMultiSnap

cv2.namedWindow('out')
# cv2.namedWindow('out', cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty('out',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

if __name__=='__main__':
    cam = Camera(None, 320, 480)
    gpio = GPIOPinReader()

    sm = StyleMultiSnap(cam=cam, window_name='out')
    sm_face = StyleMultiSnap(cam=cam, window_name='out', focus_face=True)
    fm = FaceReplaceMulti(cam=cam, window_name='out')
    sn1 = StarryNightSnap(cam=cam, window_name='out')
    sn2 = StarryNightSnap(cam=cam, window_name='out', static_bg= cv2.imread('resources/starry_night_clone_small.jpg'), invert_drawing=False)
    sn3 = StarryNightSnap(cam=cam, window_name='out', style_model =None, static_fg= cv2.imread('resources/starry_night_clone_small.jpg'), static_bg= cv2.imread('resources/starry_dark_bg.png'), preserve_color=False)
    wm = WarholMonroeSnap(cam=cam, window_name='out')
    ms = MunchScreamSnap(cam=cam, window_name='out')
    fr1 = FaceReplace(cam=cam, window_name='out', static_bg='resources/mona_lisa.png', face_area=[0.27, 0.145, 0.61, 0.55])

    effect_names = ['style multi', 'style multi F', 'face multi', 'starry clone', 'starry night', 'starry night 2', 'warhol', 'munch', 'mona lisa']
    effects = [sm.multi_snap, sm_face.multi_snap, fm.face_replace_snap, sn1.starry_night_snap, sn2.starry_night_snap, sn3.starry_night_snap, wm.monroe_snap, ms.munch_scream_snap, fr1.face_replace_snap]
    
    selected = 0
    text_duration = 0.5

    while True:
        try:
            frame = cam.get_frame()
            cv2.putText(frame, effect_names[selected], (10, cam.height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
            cv2.imshow('out', frame)
            key = gpio.waitKey(1)

            if key == 0:
                try:
                    effects[selected]()
                except Exception as e:
                    print(e)

                while gpio.waitKey(1)!= -1:
                    cv2.waitKey(50)
            
            if key == 1 or key == 2:
                selected = selected + 1 if key==1 else selected - 1
                selected%=len(effects)
                print(effect_names[selected])
        except Exception as e:
            print(e)
            gpio.waitKey(50)