import numpy as np
import cv2
import onnxruntime as ort

class SketchModel:
    def __init__(self, model_path='animegan/AnimeGANv3_PortraitSketch_25.onnx'):
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def transfer_style(self, frame: np.ndarray) -> np.ndarray:
        content_tensor = frame.astype(np.float32) / 127.5 - 1.0
        content_tensor = np.expand_dims(content_tensor, axis=0)
        generated_tensor = self.ort_session.run([self.output_name], {self.input_name: content_tensor})[0]
        generated_image = (generated_tensor.squeeze() + 1.) / 2 * 255
        out_frame = generated_image.astype(np.uint8)


        return out_frame