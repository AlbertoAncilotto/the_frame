import cv2
import numpy as np

class SketchFast:
    def __init__(self, sigma=0):
        self.sigma = sigma

    def transfer_style(self, image: np.ndarray) -> np.ndarray:
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # use Canny edge detection to get edges
        v = np.median(blur)
        lower = int(max(0, (1.0 - self.sigma) * v))
        upper = int(min(255, (1.0 + self.sigma) * v))
        edges = cv2.Canny(blur, lower, upper)

        # invert the edges to get a white-on-black image
        edges = 255 - edges 
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=1)

        # convert edges to 3 channels to get a grayscale RGB image
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges