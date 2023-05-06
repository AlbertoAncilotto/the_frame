from camera import Camera
from segmentation.segment import SegmentationModel
from style_transfer.style_transfer import StyleTransfer
from style_transfer.utils import transfer_color
from animegan.to_sketch import SketchModel
from countdown_timer_gui.snap_camera import SnapCamera
from transition.image_morpher import ImageMorpher
import cv2
import numpy as np
import threading
from queue import Queue


class StarryNightSnap:
    def __init__(self, height=480, width=320, cam=None, window_name=None, style_model='style_transfer/color_starry_300x480.onnx', style_w=300, style_h=480, static_bg=None, static_fg=None, invert_drawing=False, preserve_color=True):
        self.width = width
        self.heigth = height

        self.cam = Camera('cv2', self.width, self.heigth) if cam is None else cam
        self.window_name = window_name

        self.people_seg = SegmentationModel()
        self.drawing_model = SketchModel()
        self.preserve_color=preserve_color
        if static_bg is None and static_fg is None:
            self.style_model = StyleTransfer(style_model, width=style_w, height=style_h, preserve_color=self.preserve_color, alpha=0.5)
            self.background = None
            self.foreground = None
        else:
            self.background = cv2.resize(static_bg, (self.width, self.heigth)) if static_bg is not None else None
            self.foreground = cv2.resize(static_fg, (self.width, self.heigth)) if static_fg is not None else None
        self.invert_drawing = invert_drawing
        self.snap_camera = SnapCamera()
    
    def starry_night_snap(self):
        
        # key = cv2.waitKey(1) & 0xFF
        # while not key == ord('p'):
        #     frame = self.cam.get_frame()
        #     cv2.imshow(self.window_name,frame)
        #     key = cv2.waitKey(1) & 0xFF

        seconds_left = self.snap_camera.start_snap()
        while seconds_left > 0:
            frame = self.cam.get_frame()
            display_frame, seconds_left = self.snap_camera.snap(frame.copy())
            #print(seconds_left)
            cv2.imshow(self.window_name,display_frame)
            cv2.waitKey(1)

        print('starting inference')
        if self.background is None and self.foreground is None:
            segment_map = self.people_seg.segment(frame.copy())
            # cv2.imshow('frame', frame)
            styled_frame = self.style_model.transfer_style(frame.copy(), seg_map=segment_map)
            # cv2.imshow('styled_frame', styled_frame)
            # cv2.waitKey()
        else:
            segment_map = self.people_seg.segment(frame.copy())
            styled_frame = self.background if self.foreground is None else self.people_seg.apply(self.foreground, self.background)
            if self.preserve_color:
                styled_frame = transfer_color(frame.copy(), styled_frame, segment_map, alpha=0.5)

        result_queue = Queue()
        frames_queue = Queue()

        thread1 = threading.Thread(target=self.threading_compute_images, args=(frame.copy(), result_queue))
        thread2 = threading.Thread(target=self.threading_morph_images, args=(frame.copy(), styled_frame, frames_queue))

        # Start both threads
        thread1.start()
        thread2.start()

        # Show animation
        while frames_queue.empty():
            cv2.waitKey(30)

        while not frames_queue.empty():
            intermediate_frame = frames_queue.get()
            intermediate_frame = cv2.resize(intermediate_frame, (self.width, self.heigth))
            cv2.imshow(self.window_name, intermediate_frame)
            cv2.waitKey(30)

        # Wait for both threads to finish before continuing
        thread1.join()
        thread2.join()

        # Retrieve the result from the queue
        drawing = result_queue.get()
        styled_frame = cv2.resize(styled_frame, (self.width, self.heigth))
        out_frame = np.minimum(styled_frame, drawing) if not self.invert_drawing else np.maximum(styled_frame, cv2.bitwise_not(drawing))
        out_morpher = ImageMorpher(styled_frame, out_frame, n_frames=20)
        out_morpher.animate(window_name=self.window_name)

        cv2.imshow(self.window_name, out_frame)
        cv2.waitKey(10000)
    
    def threading_compute_images(self, frame, result_queue):
        drawing = self.drawing_model.transfer_style(frame.copy())
        result_queue.put(drawing)

    def threading_morph_images(self, frame1, frame2, frames_queue):
        morpher = ImageMorpher(frame1, frame2, n_frames=100)
        morpher.animate(frames_queue=frames_queue)

if __name__=='__main__':
    sn = StarryNightSnap()
    sn.starry_night_snap()