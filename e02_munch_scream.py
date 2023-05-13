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
from keypad import GPIOPinReader
from image_writer import ImageWriter

class MunchScreamSnap:
    def __init__(self, height=480, width=320, cam=None, window_name=None, invert_drawing=False, bg_path='resources/munch_scream_bg.jpg', preserve_color=True):
        self.width = width
        self.heigth = height
        self.gpio = GPIOPinReader()

        self.cam = Camera('cv2', self.width, self.heigth) if cam is None else cam
        self.window_name = window_name

        self.people_seg = SegmentationModel()
        self.drawing_model = SketchModel()
        self.preserve_color=preserve_color
        self.style_model = StyleTransfer('style_transfer/color_starry_300x480.onnx', preserve_color=self.preserve_color)

        self.background = cv2.resize(cv2.imread(bg_path), (self.width, self.heigth))
        self.invert_drawing = invert_drawing
        self.snap_camera = SnapCamera()

    def frame_process(self, frame, style=False):
        segment_map = self.people_seg.segment(frame.copy())
        segment_map = np.clip(segment_map, 0, 1)
        segment_map = cv2.cvtColor(segment_map,cv2.COLOR_GRAY2BGR)
        out_frame = (frame*segment_map + self.background*(1-segment_map)).astype(np.uint8)
        return out_frame
    
    def munch_scream_snap(self):
        
        # key = cv2.waitKey(1) & 0xFF
        # while not key == ord('p'):
        #     frame = self.cam.get_frame()
        #     cv2.imshow(self.window_name,frame)
        #     key = cv2.waitKey(1) & 0xFF
        
        image_writer = ImageWriter()

        seconds_left = self.snap_camera.start_snap()
        while seconds_left > 0:
            frame = self.cam.get_frame()
            frame = self.frame_process(frame)
            display_frame, seconds_left = self.snap_camera.snap(frame.copy())
            #print(seconds_left)
            cv2.imshow(self.window_name,display_frame)
            cv2.waitKey(1)

        
        image_writer.save_image(frame)
        print('starting inference')
        if self.background is None:
            segment_map = self.people_seg.segment(frame.copy())
            styled_frame = self.style_model.transfer_style(frame.copy(), seg_map=segment_map)
        else:
            segment_map = self.people_seg.segment(frame.copy())
            styled_frame = self.background
            styled_frame = transfer_color(frame.copy(), styled_frame, segment_map, alpha=0.25)

        result_queue = Queue()
        frames_queue = Queue()

        print('frame:', frame.shape, frame.dtype, np.min(frame), np.max(frame), 'styled frame:', styled_frame.shape, styled_frame.dtype, np.min(styled_frame), np.max(styled_frame))

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

        
        image_writer.save_image(out_frame)
        cv2.imshow(self.window_name, out_frame)
        self.gpio.waitKey(10000)
    
    def threading_compute_images(self, frame, result_queue):
        drawing = self.drawing_model.transfer_style(frame.copy())
        result_queue.put(drawing)

    def threading_morph_images(self, frame1, frame2, frames_queue):
        morpher = ImageMorpher(frame1, frame2, n_frames=100)
        morpher.animate(frames_queue=frames_queue)

if __name__=='__main__':
    sn = MunchScreamSnap()
    sn.munch_scream_snap()