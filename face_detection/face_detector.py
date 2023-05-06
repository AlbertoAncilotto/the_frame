import cv2
import numpy as np
import face_detection.box_utils_numpy as box_utils
import onnxruntime as ort
# from face_detection.sort import Sort




# class BoxTracker:
#     def __init__(self, alpha=0.35):
#         self.tracker = Sort()
#         self.prev_boxes = []
#         self.alpha=alpha

#     def smooth_old(self, boxes):
#         self.curr_boxes = boxes
#         if len(self.prev_boxes) == 0:
#             self.prev_boxes = boxes
#         else:
#             try:
#                 self.prev_boxes = self.tracker.update(boxes)
#             except:
#                 self.prev_boxes = boxes

#         smoothed_boxes = []
#         for i, box in enumerate(self.prev_boxes):
#             if np.all(box == 0):
#                 smoothed_boxes.append(self.curr_boxes[i].astype(np.int16))
#             else:
#                 # print('curr boxes',self.curr_boxes)
#                 print('box',box)
#                 smoothed_boxes.append((((1-self.alpha) * self.curr_boxes[i]) + (self.alpha * box[:4])).astype(np.int16))

#         return smoothed_boxes

class FaceDetector:
    def __init__(self, onnx_path= "face_detection/models/onnx/version-RFB-320.onnx", width =320, height=240, threshold=0.7, alpha=0.35):

        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.width = width
        self.heigth =  height

        self.threshold = threshold
        self.alpha = alpha
        self.prev_box = []
        # self.box_tracker = BoxTracker(alpha=self.alpha)

    def smooth_box(self, box):
        if len(self.prev_box) == 0:
            return box
        else:
            smoothed_box = (1 - self.alpha) * self.prev_box + self.alpha * box
            return smoothed_box.astype(int)
        
    def predict(self, width, height, confidences_list, boxes_list, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes_list[0]
        confidences = confidences_list[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def draw_bounding_boxes(self, frame, boxes, color=(0,255,0), thickness=2):
        if boxes is not None and len(boxes)>0:
            for box in boxes:
                (x1, y1, x2, y2) = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)



    def find_multiple_faces(self, frame, orig_image=None, nmax_faces=1):
        if orig_image is None:
            orig_image = frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.width, self.heigth))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, _, _ = self.predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, self.threshold, top_k=nmax_faces)

        # boxes = self.box_tracker.smooth(boxes)
        return boxes
    

    
    def find_single_face(self, frame, orig_image=None):
        if orig_image is None:
            orig_image = frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.width, self.heigth))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        confidences, boxes = self.ort_session.run(None, {self.input_name: image})
        boxes, _, _ = self.predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, self.threshold, top_k=1)

        if len(boxes) > 0:
            box = self.smooth_box(boxes[0])
            self.prev_box = box
            return [box]
        else:
            return []
        
    def get_crops(self, boxes, frame, width, height, padding=128, zoom=1):
        padded_frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_REFLECT)
        crops = []
        for box in boxes:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            box_width, box_height = int((box[2] - box[0])/zoom), int((box[3] - box[1])/zoom)
            aspect_ratio = width / height
            if box_width / box_height < aspect_ratio:
                crop_width, crop_height = int(box_height * aspect_ratio), box_height
            else:
                crop_width, crop_height = box_width, int(box_width / aspect_ratio)
            crop_x, crop_y = int(cx - crop_width / 2), int(cy - crop_height / 2)
            crops.append(cv2.resize(padded_frame[crop_y+padding:crop_y+crop_height+padding, crop_x+padding:crop_x+crop_width+padding], (width, height)))
        return crops