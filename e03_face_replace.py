from camera import Camera
from segmentation.segment import SegmentationModel
from style_transfer.style_transfer import StyleTransfer
from countdown_timer_gui.snap_camera import SnapCamera
from face_detection.face_detector import FaceDetector
from transition.image_morpher import ImageMorpher
import cv2
import numpy as np
import threading
from queue import Queue
from skimage import exposure
from keypad import GPIOPinReader


class FaceReplace:
    def __init__(self, height=480, width=320, cam=None, window_name=None, static_bg=None, face_area=None, style_path=None, style_w=200, style_h=200, softer_mask = 'resources/mask.jpg'):
        self.width = width
        self.heigth = height
        self.gpio = GPIOPinReader()

        self.cam = Camera('cv2', self.width, self.heigth) if cam is None else cam
        self.window_name = window_name
        self.background = cv2.resize(cv2.imread(static_bg), (self.width, self.heigth))

        self.people_seg = SegmentationModel()
        self.face_det = FaceDetector()
        if style_path is not None:
            self.style_model = StyleTransfer(style_path, height=style_h, width=style_w, preserve_color=True)
        else:
            self.style_model = None
        self.snap_camera = SnapCamera()

        self.softer_mask = cv2.imread(softer_mask)/255.0 if softer_mask is not None else None

        if face_area is None:
            boxes = self.face_det.find_single_face(self.background)
            print(boxes[0])
            self.face_area = boxes[0]
            # self.face_det.draw_bounding_boxes(self.background, boxes)
            # cv2.imshow('face',self.background)
            # cv2.waitKey()
        else:
            self.face_area =[int(face_area[0]*width),
                             int(face_area[1]*height),
                             int(face_area[2]*width),
                             int(face_area[3]*height)]
    
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
        print('starting inference')
        # print(self.gpio.waitKey(1))
        while self.gpio.waitKey(1) == -1:
            frame = self.cam.get_frame() 
            boxes = self.face_det.find_single_face(frame)
            try:
                styled_frame= self.merge_images(boxes, frame, self.softer_mask)
            except Exception as e:
                print(e)
                continue

            styled_frame = cv2.resize(styled_frame, (self.width, self.heigth))
            cv2.imshow(self.window_name, styled_frame)


    def merge_images(self, boxes, frame, softer_mask=None):
        crop_size = (self.face_area[2]-self.face_area[0], self.face_area[3]-self.face_area[1])
        box = boxes[0]

        crop = frame[box[1]:box[3], box[0]:box[2]]
        crop = cv2.resize(crop, crop_size)
        bg_crop = self.background[self.face_area[1]:self.face_area[3], self.face_area[0]:self.face_area[2]]

        self.people_seg.segment(crop)
        crop = self.people_seg.apply(crop, bg_crop)
        
        try:
            crop = exposure.match_histograms(crop, bg_crop, channel_axis=2) #skimage 0.20
        except:
            crop = exposure.match_histograms(crop, bg_crop, multichannel=True) #older skimage

        styled_frame = self.background.copy()
        
        if softer_mask is not None:
            mask = cv2.resize(softer_mask, crop_size)
            styled_frame[self.face_area[1]:self.face_area[3],self.face_area[0]:self.face_area[2]] = (crop*mask + bg_crop*(1-mask)).astype(np.uint8)
        else:
            styled_frame[self.face_area[1]:self.face_area[3],self.face_area[0]:self.face_area[2]] = crop

        
        if self.style_model is not None:
            styled_frame =  self.style_model.transfer_style(styled_frame)
        return styled_frame
        

if __name__=='__main__':
    # sn = FaceReplace(static_bg='resources/mona_lisa.png', face_area=[0.27, 0.145, 0.61, 0.55])
    sn = FaceReplace(static_bg='resources/gogh_self_portrait.jpeg', style_path='style_transfer/gogh_bg_300x480.onnx', style_w=300, style_h=480, face_area=[0.29, 0.22, 0.6, 0.64])
    sn.face_replace_snap()