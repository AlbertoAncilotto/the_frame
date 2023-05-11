from camera import Camera
from segmentation.segment import SegmentationModel
from style_transfer.style_transfer import StyleTransfer
from countdown_timer_gui.snap_camera import SnapCamera
from face_detection.face_detector import FaceDetector
from transition.image_morpher import ImageMorpher
from keypad import GPIOPinReader
import cv2
import numpy as np
import threading
from queue import Queue
from skimage import exposure
import os

class FaceReplaceMulti:
    def __init__(self, height=480, width=320, cam=None, window_name=None, softer_mask = 'resources/mask.jpg'):
        self.width = width
        self.heigth = height
        self.gpio = GPIOPinReader()

        self.cam = Camera('cv2', self.width, self.heigth) if cam is None else cam
        self.window_name = window_name

        self.people_seg = SegmentationModel()
        self.face_det = FaceDetector()
        self.snap_camera = SnapCamera()

        self.softer_mask = cv2.imread(softer_mask)/255.0

        backgrounds = ['mona_lisa.png',
                       'courbet.png',
                       'girl-with-a-pearl-earring.jpg',
                       'nascita_di_venere.jpg',
                        'Dama_ermellino.jpg'
                            ]
        
        face_areas = [
                            [0.27, 0.145, 0.61, 0.55],
                            [0.31, 0.22, 0.80, 0.70],
                            [0.31, 0.275, 0.55, 0.60],
                            [0.21, 0.19, 0.56, 0.56],
                            [0.45, 0.15, 0.68, 0.39]
                            ]
        
        self.backgrounds = [cv2.resize(cv2.imread(os.path.join('resources',bg)), (self.width, self.heigth)) for bg in backgrounds]
        self.backgrounds_nf = [cv2.resize(cv2.imread(os.path.join('resources','noface',bg)), (self.width, self.heigth)) for bg in backgrounds]
        self.face_areas = [[int(x1*width), int(y1*height), int(x2*width), int(y2*height)] for [x1,y1,x2,y2] in face_areas]


    def face_replace_snap(self):
        
        # key = cv2.waitKey(1) & 0xFF
        # while not key == ord('p'):
        #     frame = self.cam.get_frame()
        #     cv2.imshow(self.window_name,frame)
        #     key = cv2.waitKey(1) & 0xFF

        seconds_left = self.snap_camera.start_snap()
        while seconds_left > 0:
            frame = self.cam.get_frame()            
            boxes = self.face_det.find_single_face(frame)
            self.face_det.draw_bounding_boxes(frame, boxes)
            display_frame, seconds_left = self.snap_camera.snap(frame.copy())
            #print(seconds_left)
            cv2.imshow(self.window_name,display_frame)
            cv2.waitKey(1)

        im_id = 0
        while True:
            frame = self.cam.get_frame() 
            boxes = self.face_det.find_single_face(frame)
            try:
                styled_frame= self.merge_images(boxes, frame, self.softer_mask, im_id)
            except:
                continue

            styled_frame = cv2.resize(styled_frame, (self.width, self.heigth))
            cv2.imshow(self.window_name, styled_frame)
            key = self.gpio.waitKey(1)
            if key == 0:
                im_id +=1
                im_id%= len(self.backgrounds)
            elif key != -1:
                return


    def merge_images(self, boxes, frame, softer_mask=None, im_id=0):
        crop_size = (self.face_areas[im_id][2]-self.face_areas[im_id][0], self.face_areas[im_id][3]-self.face_areas[im_id][1])
        box = boxes[0]
        crop = frame[box[1]:box[3], box[0]:box[2]]
        crop = cv2.resize(crop, crop_size)

        bg_crop = self.backgrounds[im_id][self.face_areas[im_id][1]:self.face_areas[im_id][3], self.face_areas[im_id][0]:self.face_areas[im_id][2]]
        bg_crop_nf = self.backgrounds_nf[im_id][self.face_areas[im_id][1]:self.face_areas[im_id][3], self.face_areas[im_id][0]:self.face_areas[im_id][2]]
        
        crop = exposure.match_histograms(crop, bg_crop, multichannel=True)

        self.people_seg.segment(crop)
        crop = self.people_seg.apply(crop, bg_crop_nf)
        styled_frame = self.backgrounds_nf[im_id].copy()
        
        if softer_mask is not None:
            mask = cv2.resize(softer_mask, crop_size)
            styled_frame[self.face_areas[im_id][1]:self.face_areas[im_id][3],self.face_areas[im_id][0]:self.face_areas[im_id][2]] = (crop*mask + bg_crop_nf*(1-mask)).astype(np.uint8)
        else:
            styled_frame[self.face_areas[im_id][1]:self.face_areas[im_id][3],self.face_areas[im_id][0]:self.face_areas[im_id][2]] = crop
        return styled_frame
        

if __name__=='__main__':
    # sn = FaceReplace(static_bg='resources/mona_lisa.png', face_area=[0.27, 0.145, 0.61, 0.55])
    sn = FaceReplaceMulti(window_name='out')
    sn.face_replace_snap()