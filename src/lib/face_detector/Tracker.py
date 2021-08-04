import os
import sys
import numpy as np
import cv2
import torch
from torch.autograd import Variable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centerface_pytorch.centerface import centerFace_MobileNet
from centerface_pytorch.utils import centerFace_Utils
from centerface_pytorch.align_trans import get_reference_facial_points, warp_and_crop_face
# from trackingv2 import Sort
# from head_pose import PoseEstimator
from face_pose import FacePose
def sort_list(list1): 
    z = [list1.index(x) for x in sorted(list1, reverse = True)] 
    return z 

class Tracker(object):
    def __init__(self, conf=None):
        self.im_scale = None
        self.sort = True
        self.conf = conf
        self.camera_config = {}
        self.tracker = {}
        self._startup_init_(conf)
        self._detector_init_(conf)
        
    def register_camera(self, cam_id, m = 15, s= 10):
        self.tracker[cam_id] = Sort(m, s)

    def _startup_init_(self, conf):
        self.target_size = conf.scales[0]
        self.max_size = conf.scales[1]
        self.limit = conf.face_limit
        self.min_face_size = conf.min_face_size
        self.threshold = conf.threshold
        self.device = conf.device
        self.refrence = get_reference_facial_points(default_square = True)
        self.utils = centerFace_Utils(conf.nms_thresholds)

    def _detector_init_(self, conf):
        self.model = centerFace_MobileNet()
        self.model = self.model.to(self.device)
        checkpoint = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'centerface_pytorch/checkpoint.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        del checkpoint

    def img_process(self, img):
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > self.max_size:
            im_scale = float(self.max_size) / float(im_size_max)
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = im[:, :, 2 - i] 
        return im_tensor, im_scale#, top, left
    
    def detec_and_track(self, img, with_landmark = True):
        # t = time.time()
        faces = []
        faces_2_tensor = []
        lst_head_pose = [] 
        lst_boxes = []
        lm68_list = []
        dict_result = {}
        
        h, w, _ = img.shape
        if "%dx%d"%(w, h) not in self.camera_config.keys():
            im_tensor, im_scale  = self.img_process(img)
            self.camera_config["%dx%d"%(w, h)] = {"im_scale": im_scale,
                                                    "head_pose_predict": FacePose(size=(h, w))}
        else:
            im_scale = self.camera_config["%dx%d"%(w, h)]["im_scale"]
            im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
            for i in range(3):
                im_tensor[0, i, :, :] = im[:, :, 2 - i]

        im = torch.from_numpy(im_tensor).to(self.device)
        im_tensor = Variable(im.contiguous())
        output = self.model(im_tensor)
        boxes, landmarks = self.utils.detect(im, output, self.threshold, im_scale)    
        if self.limit:
            boxes, landmarks = boxes[:self.limit], landmarks[:self.limit]

        if self.sort:
            face_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3]-boxes[:, 1])
            indexs = np.argsort(face_area)[::-1]
            boxes = boxes[indexs].astype(np.int)
            landmarks = landmarks[indexs]

        if len(boxes) > 0:
            for i, landmark in enumerate(landmarks):
                f = True
                if with_landmark:
                    f, pose, lm68 = self.camera_config["%dx%d"%(w, h)]["head_pose_predict"].verify_pose(img, boxes[i])

                if f:
                    if with_landmark:
                        lm68_list.append(lm68)
                        lst_head_pose.append(pose)
                    lst_boxes.append(boxes[i])
                    # warped_face, face_img_tranform = warp_and_crop_face(img, landmark, self.refrence, crop_size=(112,112))
                    # faces.append(warped_face)
                    # faces_2_tensor.append(torch.FloatTensor(face_img_tranform).contiguous().unsqueeze(0).to(self.device))

        num_face = len(lst_boxes)
        dict_result["num_face"] = num_face
        dict_result["bboxs"] = boxes
        dict_result["poses"] = lst_head_pose
        return dict_result