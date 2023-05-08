from camera import Camera
from segmentation.segment import SegmentationModel
from style_transfer.style_transfer import StyleTransfer
from face_detection.face_detector import FaceDetector
from style_transfer.utils import transfer_color
from animegan.to_sketch import SketchModel
from countdown_timer_gui.snap_camera import SnapCamera
from transition.image_morpher import ImageMorpher
import cv2
import numpy as np
import threading
from queue import Queue
from keypad import GPIOPinReader
import time
import copy


class StyleMultiSnap:
    def __init__(self, height=480, width=320, cam=None, window_name=None):
        self.width = width
        self.heigth = height
        self.gpio = GPIOPinReader()

        self.cam = Camera('cv2', self.width, self.heigth) if cam is None else cam
        self.window_name = window_name

        self.people_seg = SegmentationModel()
        self.face_det = FaceDetector()
        self.drawing_model = SketchModel()
        self.snap_camera = SnapCamera()
        self.style_models =[
                            StyleTransfer('style_transfer/pencil_450x720.onnx', width=450, height=720, preserve_color=False),
                            StyleTransfer('style_transfer/pen_450x720.onnx', width=450, height=720, preserve_color=False),
                            StyleTransfer('style_transfer/color_starry_300x480.onnx', width=300, height=480, preserve_color=True),
                            StyleTransfer('style_transfer/matisse_450x720.onnx', width=450, height=720, preserve_color=False),
                            StyleTransfer('style_transfer/afremov_450x720.onnx', width=450, height=720, preserve_color=False),
                            SketchModel(width=480, heigth=720),
                            ]
    
    def multi_snap(self, cd=3):
        seconds_left = self.snap_camera.start_snap()
        while seconds_left > 0:
            frame = self.cam.get_frame()
            display_frame, seconds_left = self.snap_camera.snap(frame.copy())
            #print(seconds_left)
            cv2.imshow(self.window_name,display_frame)
            cv2.waitKey(1)

        print('starting inference')
        self.seg_map = self.people_seg.segment(frame)

        result_queue = Queue()

        thread1 = threading.Thread(target=self.threading_compute_images, args=(frame.copy(), result_queue))

        # Start both threads
        thread1.start()
        # Show animation
        while result_queue.empty():
            print('waiting for first result')
            cv2.waitKey(30)

        styled_frames = []
        curr_frame = frame.copy()
        frame_id = 0
        while True:
            while not result_queue.empty():
                styled_frames.append(result_queue.get())
                if len(styled_frames) == len(self.style_models):
                    drawing = styled_frames[-1]
                    bg = cv2.resize(cv2.imread('resources/papyrus.jpg'), (drawing.shape[1], drawing.shape[0]))
                    styled_frames[-1] = np.minimum(drawing, bg)
                    styled_frames[-2] = np.maximum(cv2.bitwise_not(drawing), styled_frames[-2])

            frame_id%=len(styled_frames)
            styled_frame = styled_frames[frame_id]
            frame_id+=1
            im = ImageMorpher(curr_frame, styled_frame, 20)
            curr_frame = styled_frame
            im.animate(self.window_name, delay_ms=30)
            cv2.imshow(self.window_name, styled_frame)
            ret = self.gpio.waitKey()
            if ret == 2:
                break


    
    def threading_compute_images(self, frame, result_queue):
        for st_id,st in enumerate(self.style_models):
            start = time.time()
            input_frame = frame.copy()
            try:
                styled_frame = st.transfer_style(input_frame, self.seg_map)
            except Exception as e:
                styled_frame = st.transfer_style(input_frame)
            result_queue.put(styled_frame)
            print('Computed style',st_id,'in',time.time()-start)


if __name__=='__main__':
    sn = StyleMultiSnap(window_name='out')
    sn.multi_snap()