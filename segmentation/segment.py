import numpy as np
import cv2
import onnxruntime as ort

class SegmentationModel:
    def __init__(self, model_path = "segmentation/SINet.onnx", width=320, height=320):
        self.width = width
        self.heigth = height
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
    def segment(self, frame):
        h,w,c = frame.shape
        img_seg = ((cv2.resize(frame, (self.width, self.heigth)).astype(np.float32) - 128.0)/66.0)/255.0
        seg_tensor = np.transpose(img_seg, (2, 0, 1))
        seg_tensor = np.expand_dims(seg_tensor, axis=0)
        seg_map = self.ort_session.run([self.output_name], {self.input_name: seg_tensor})[0][0][1]
        seg_map = cv2.resize(seg_map, (w,h))
        self.seg_map = seg_map
        return seg_map
    
    def apply(self, frame, background=None, invert=False):
        segment_map = np.clip(self.seg_map, 0, 1)
        if invert:
            segment_map = 1-segment_map
        segment_map = cv2.cvtColor(segment_map,cv2.COLOR_GRAY2BGR)
        if background is not None:
            return (frame*segment_map + background*(1-segment_map)).astype(np.uint8)
        else:
            return (frame*segment_map).astype(np.uint8)
    