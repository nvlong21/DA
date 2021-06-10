import os
import sys
import numpy as np
import cv2
import torch
import gc
from torch.autograd import Variable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from retinaface_pytorch.retinaface import load_retinaface_mbnet, RetinaFace_MobileNet
from retinaface_pytorch.utils import RetinaFace_Utils
from retinaface_pytorch.align_trans import get_reference_facial_points, warp_and_crop_face
import time
from head_pose import PoseEstimator
# from torchvision import transforms as trans

def sort_list(list1): 
    z = [list1.index(x) for x in sorted(list1, reverse = True)] 
    return z 


def img_process_tensorrt(img, target_size, max_size):
    im_shape = img.shape

    img = cv2.resize(img, (320, 256))
    im_size_min = target_size #np.min(im_shape[0:2])
    im_size_max = max_size #np.max(im_shape[0:2])
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] 

    im_scale = np.zeros((2, 1))

    im_scale[0] = max_size/im_shape[1] 
    im_scale[1] = target_size / im_shape[0]
    return im_tensor, im_scale

class Retinaface_Detector(object):
    def __init__(self, conf):
        self.im_scale = None
        self.sort = True
        self.conf = conf
        self.camera_config = {}
        self._startup_init(conf)
        self._detector_init(conf)

    def _startup_init(self, conf):
        self.target_size = conf.scales[0]
        self.max_size = conf.scales[1]
        self.limit = conf.face_limit
        self.min_face_size = conf.min_face_size
        self.threshold = conf.threshold
        self.device = conf.device
        self.refrence = get_reference_facial_points(default_square = True)
        self.utils = RetinaFace_Utils(conf.nms_thresholds)
        

# self.head_pose_predict = 
    def _detector_init(self, conf):
        self.model = RetinaFace_MobileNet()
        self.model = self.model.to(self.device)
        checkpoint = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'retinaface_pytorch/checkpoint.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        del checkpoint
        gc.collect()

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

    def align_multi(self, img):
        # t = time.time()
        faces = []
        faces_2_tensor = []
        lst_head_pose = [] 
        dict_result = {}
        h, w, _ = img.shape
        if "%dx%d"%(w, h) not in self.camera_config.keys():
            im_tensor, im_scale  = self.img_process(img)
            self.camera_config["%dx%d"%(w, h)] = {"im_scale": im_scale,
                                                    "head_pose_predict": PoseEstimator(img_size=(h, w))}
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
        boxes = boxes.astype(np.int)
        landmarks = landmarks.astype(np.int)
        face_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3]-boxes[:, 1])
        if len(boxes) > 0 and self.sort:
            face_area = (boxes[:,2] - boxes[:,0])*(boxes[:,3]-boxes[:, 1])
            indexs = np.argsort(face_area)[::-1]
            boxes = boxes[indexs]
            landmarks = landmarks[indexs]
            for i, landmark in enumerate(landmarks):
                warped_face, face_img_tranform = warp_and_crop_face(img, landmark, self.refrence, crop_size=(112,112))
                pose = self.camera_config["%dx%d"%(w, h)].face_orientation(landmark)
                face = Image.fromarray(warped_face)
                faces.append(face)
                faces_2_tensor.append(torch.FloatTensor(face_img_tranform).contiguous().unsqueeze(0).to(self.device)) #

        num_face = len(boxes)
        dict_result["num_face"] = num_face
        dict_result["bboxs"] = boxes
        dict_result["faces"] = faces
        dict_result["faces_tensor"] = faces_2_tensor
        # print(time.time() - t)   
        return dict_result