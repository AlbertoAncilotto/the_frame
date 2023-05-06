import cv2
import numpy as np

class ImageMorpher:
    def __init__(self, img1, img2, n_frames=30, transition = 'threshold'):
        self.img2 = img1
        self.img1 = img2
        (self.height, self.width, _) = self.img2.shape
        print('ImageMorpher: image shapes', img1.shape, img2.shape)
        if img1.shape != img2.shape:
            print('ImageMorpher WARNING: DIFFERENT IMAGE SHAPES')
            self.img2 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
            print('ImageMorpher: NEW image shapes', img1.shape, img2.shape)

        self.gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        self.min_thresh = 0
        self.max_thresh = 255
        self.thresh_increment = (self.max_thresh - self.min_thresh) / n_frames
        self.curr_frame = 0
        
        self.frames = []

        if transition == 'threshold':
            self.compute_frames_theshold(n_frames)
        elif transition == 'wipe':
            self.compute_frames_wipe(n_frames)

    def compute_frames_theshold(self, n_frames):
        for i in range(n_frames):
            curr_thresh = int(self.min_thresh + i * self.thresh_increment)
            _, thresh_img = cv2.threshold(self.gray_img2, curr_thresh, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
            masked2 = cv2.bitwise_and(self.img2, self.img2, mask=thresh_img)
            masked1 = cv2.bitwise_and(self.img1, self.img1, mask=cv2.bitwise_not(thresh_img))
            out = masked1 + masked2
            self.frames.append(out)

    def _circular_mask(self, center, radius, shape):
        h, w = shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = (dist_from_center <= radius)
        return mask.astype(np.uint8)

    def compute_frames_wipe(self, n_frames):
        center = (self.width//2, self.height//2)
        radius = 0
        step = int(np.sqrt(center[0]**2 + center[1]**2)/n_frames)

        for i in range(n_frames):
            mask = self._circular_mask(center, radius, self.img1.shape)*255
            inv_mask = cv2.bitwise_not(mask)

            frame = cv2.bitwise_and(self.img2, cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR))
            frame += cv2.bitwise_and(self.img1, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

            self.frames.append(frame)
            radius += step
        
    def next_frame(self):
        if self.curr_frame < len(self.frames):
            frame = self.frames[self.curr_frame]
            self.curr_frame += 1
            return frame
        else:
            return None
        
    def animate(self, window_name=None, frames_queue=None, delay_ms=30):
    #     if window_name is not None:
    #         cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            frame = self.next_frame()
            if frame is None:
                break
            if window_name is not None:
                cv2.imshow(window_name, frame)
                cv2.waitKey(delay_ms)
            if frames_queue is not None:
                frames_queue.put(frame)
                # print('frame put int queue, queue lenght:', frames_queue.qsize())
        # if window_name is not None:
        #     cv2.destroyWindow(window_name)
            

if __name__=='__main__':
        # Load the two images you want to morph between
    img1 = cv2.imread('resources/munch_scream_bg.jpg')
    img1 = cv2.resize(img1, (320,480))
    img2 = cv2.imread('resources/starry_night_clone_small.jpg')
    # img2 = cv2.resize(img2, (320,480))

    # Instantiate ImageMorpher and show the animation
    morpher = ImageMorpher(img1, img2, n_frames=100, transition='threshold')
    morpher.animate(window_name='out')