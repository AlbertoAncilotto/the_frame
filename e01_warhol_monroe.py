from camera import Camera
from face_detection.face_detector import FaceDetector
from animegan.to_sketch import SketchModel
from countdown_timer_gui.snap_camera import SnapCamera
from transition.image_morpher import ImageMorpher
import cv2
import numpy as np
import threading
from queue import Queue


class WarholMonroeSnap:
    def __init__(self, height=480, width=320, cam=None, window_name=None):
        self.width = width
        self.heigth = height

        self.cam = Camera('cv2', self.width, self.heigth) if cam is None else cam
        self.window_name = window_name

        self.drawing_model = SketchModel()
        self.snap_camera = SnapCamera()
        self.face_det = FaceDetector()

        # set criteria and number of clusters
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        self.K = 6

        # define kernel for morphological operations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process_frame(self, gray, num_thresholds=5):
        mean, sigma = cv2.meanStdDev(gray)
        thresh_step=2.5*sigma/num_thresholds
        thresh = [int(mean - 2.5*sigma + n*thresh_step) for n in range(num_thresholds)]
        thresh_imgs = [cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)[1] for t in thresh]
        morph_imgs = [cv2.morphologyEx(img, cv2.MORPH_OPEN, self.kernel) for img in thresh_imgs]
        morph_imgs = [cv2.morphologyEx(img, cv2.MORPH_CLOSE, self.kernel) for img in morph_imgs]
        return morph_imgs

    def create_first_image(self, masks, gray):
        colors = [(120,10,100),
                (255, 172, 51),  # bright orange
                (240, 98, 146),  # fuchsia
                (255, 222, 89),  # yellow
                (102, 191, 255),] # sky blue

        # create an empty white image with the same size as the masks
        output = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)
        output.fill(255)

        # iterate over the masks and colors in reverse order
        for mask, color in zip(reversed(masks), reversed(colors)):
            # large_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            # output[large_mask == 0] = [0,0,0]
            output[mask == 0] = color

        output[gray == 0] = [0,0,0]
        return output

    def shift_hue(self,frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_shift = 50
        hsv[..., 0] = (hsv[..., 0]*2 + hue_shift) % 180
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def to_lines(self, image, seg_map=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
        thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        out_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        if seg_map is not None:

            mask = (seg_map == [0, 0, 0]).all(axis=2)
            out_image[mask] = [0, 255, 141]
        return out_image


    def to_flat(self, frame):
        # reshape and convert to np.float32
        Z = frame.reshape((-1, 3))
        Z = np.float32(Z)

        # apply k-means clustering
        ret, label, center = cv2.kmeans(Z, self.K, None, self.criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

        # convert back to uint8 and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((frame.shape))

        # apply morphological opening and closing
        res2 = cv2.morphologyEx(res2, cv2.MORPH_OPEN, self.kernel)
        res2 = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, self.kernel)

        # return the resulting frame
        return res2
    
    def monroe_snap(self):

        seconds_left = self.snap_camera.start_snap()
        while seconds_left > 0:
            frame = self.cam.get_frame()
            display_frame, seconds_left = self.snap_camera.snap(frame.copy())
            print(seconds_left)
            cv2.imshow(self.window_name,display_frame)
            cv2.waitKey(1)

        print('starting inference')
        boxes = self.face_det.find_single_face(frame.copy())

        if len(boxes)>0:
            face_crop = self.face_det.get_crops(boxes, frame, self.width//2, self.heigth//2, padding=self.heigth//2, zoom=0.8)[0]
        else:
            face_crop = cv2.resize(frame, (self.width//2, self.heigth//2))
        
        # face_crop_segmented = (face_crop*segment_map).astype(np.uint8)
        face_crop_segmented = face_crop
        gray = cv2.cvtColor(face_crop_segmented, cv2.COLOR_BGR2GRAY)

        masks = self.process_frame(gray)
        out_1 = self.create_first_image(masks, gray)
        out_2 = self.shift_hue(out_1)#(self.to_flat(face_crop)*segment_map).astype(np.uint8)
        out_3 = self.shift_hue(out_2)
        out_4 = self.shift_hue(out_3)

        out = np.zeros((self.heigth, self.width, 3), dtype=np.uint8)

        # Place the input images in the output image
        out[0:self.heigth//2, 0:self.width//2] = out_1
        out[0:self.heigth//2, self.width//2:self.width] = out_2
        out[self.heigth//2:self.heigth, 0:self.width//2] = out_3
        out[self.heigth//2:self.heigth, self.width//2:self.width] = out_4

        result_queue = Queue()
        frames_queue = Queue()

        frame4x = np.vstack([np.hstack([face_crop, face_crop]), np.hstack([face_crop, face_crop])])
        thread1 = threading.Thread(target=self.threading_compute_images, args=(cv2.resize(face_crop, (self.width, self.heigth)), result_queue))
        thread2 = threading.Thread(target=self.threading_morph_images, args=(frame4x, out, frames_queue))

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
        drawing = cv2.resize(drawing, (self.width//2, self.heigth//2))
        drawing4x = np.vstack([np.hstack([drawing, drawing]), np.hstack([drawing, drawing])])

        out_frame = np.minimum(out, drawing4x) #if not self.invert_drawing else np.maximum(styled_frame, cv2.bitwise_not(drawing))
        out_morpher = ImageMorpher(out, out_frame, n_frames=20, transition='threshold')
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
    ms = WarholMonroeSnap(window_name='out')
    ms.monroe_snap()