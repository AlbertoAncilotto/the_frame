import numpy as np
import cv2
import onnxruntime as ort
import style_transfer.utils as utils # assuming this module contains a transfer_color function

class StyleTransfer:
    def __init__(self, model_path= 'style_transfer/brush_300x480.onnx', height = 480, width = 300, preserve_color = True, alpha=1.0):

        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        self.preserve_color = preserve_color
        self.height = height
        self.width = width
        self.alpha = alpha
        self.added_noise =(np.random.rand(self.height, self.width, 3).astype(np.float32)-0.5)*10

    def transfer_style(self, frame: np.ndarray, seg_map: np.ndarray = None) -> np.ndarray:
        height, width, _ = frame.shape
        if height != self.height or width!= self.width:
            img = cv2.resize(frame,(self.width, self.height))
        else:
            img = frame.copy()
        img = (img + self.added_noise) / 260.0
        content_tensor = np.transpose(img, (2, 0, 1))
        content_tensor = np.expand_dims(content_tensor, axis=0)
        generated_tensor = self.ort_session.run([self.output_name], {self.input_name: content_tensor})[0]
        generated_image = generated_tensor.squeeze()
        generated_image = generated_image.transpose(1, 2, 0)

        if self.preserve_color:
            generated_image = utils.transfer_color(img, generated_image, mask=seg_map, alpha=self.alpha)

        out_frame = (generated_image*255).astype(np.uint8)
        out_frame = cv2.resize(out_frame,(width, height))

        return out_frame