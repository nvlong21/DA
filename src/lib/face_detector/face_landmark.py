import onnxruntime
import numpy as np
import cv2
import os
pwd = os.path.dirname(os.path.realpath(__file__))

class FaceLandmark:
    def __init__(self):
        model_path = os.path.join(pwd, "landmark_detection_56.onnx")
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.mean = np.asarray([ 0.485, 0.456, 0.406 ])
        self.std = np.asarray([ 0.229, 0.224, 0.225 ])

    def get_landmarks(self, face_img):
        h, w = face_img.shape[:2]
        assert h == w, 'face_img is not square.'
        cropped_face = cv2.resize(face_img, (56, 56))
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        cropped_face = cropped_face / 255.
        cropped_face = (cropped_face - self.mean) / self.std
        cropped_face = np.transpose(cropped_face, (2, 0, 1))
        cropped_face = cropped_face[np.newaxis]
        cropped_face = cropped_face.astype(np.float32)
        ort_inputs = {self.ort_session.get_inputs()[0].name: cropped_face}
        ort_outs = self.ort_session.run(None, ort_inputs)
        landmark = ort_outs[0]
        landmark = landmark.reshape(-1,2)
        landmark = landmark * np.array([w, h])
        # landmark = landmark.astype(np.int32)
        return landmark

