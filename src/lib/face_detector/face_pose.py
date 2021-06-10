import os
pwd = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import cv2
import math
# from src.detector.gaze_tracking import Eye

from src.detector.face_landmark import FaceLandmark

class FacePose:
    def __init__(self, size = None):
        self.fl = FaceLandmark()
        self.size = size
        if size is not None:
            self._setup_camera(size[0], size[1])
        self.model_points_68 = self._get_full_model_points()
        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

    def verify_pose(self, image, box):
        if self.size is None:
            height, width = image.shape[:2]
            self._setup_camera(height, width)
        box = _move_box(box)
        if box_in_image(box, image): #, "the box is outside image"
            face_img = image[box[1]: box[3], box[0]: box[2]]
            landmarks = self.fl.get_landmarks(face_img)
            landmarks[:, 0] += box[0]
            landmarks[:, 1] += box[1]
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68,
                landmarks,
                self.camera_matrix,
                self.dist_coeefs)
            rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
            proj_matrix = np.hstack((rvec_matrix, translation_vector))
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
            pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
            pitch = math.degrees(math.asin(math.sin(pitch)))
            roll = -math.degrees(math.asin(math.sin(roll)))
            yaw = math.degrees(math.asin(math.sin(yaw)))
            return True, {"pitch": int(pitch), "yaw": int(yaw), "roll": int(roll)}, landmarks
        else:
            return False, {}, []

    def eye_blink(self, gray, COUNTER, TOTAL):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy

        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < cfg.EYE_AR_THRESH:
            COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= cfg.EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0
        return COUNTER, TOTAL

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        eye_ratio = (A + B) / (2.0 * C)
        # return the eye
        return eye_ratio

    def _setup_camera(self, height, width):
        self.size = (height, width)
        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")
    

    def _get_full_model_points(self):
        """Get all 68 3D model points from file"""
        filename = os.path.join(pwd, "facepose_3d.txt")
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        model_points[:, 2] *= -1
        return model_points

def _move_box(box):
    offset_y = int(abs((box[3] - box[1]) * 0.15))
    offset = [0, offset_y]
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        pass
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'
    return [left_x, top_y, right_x, bottom_y]


def box_in_image(box, image):
    """Check if the box is in image"""
    rows, cols = image.shape[:2]
    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows
