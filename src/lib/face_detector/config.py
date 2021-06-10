from easydict import EasyDict as edict
import torch
import os

def detector_config(device = None):
    conf = edict()
    if device is None:
        conf.device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else 
    else:
        conf.device = torch.device(device)

    conf.face_limit = 10  #when inference, at maximum detect 5 faces in one image, my laptop is slow
    conf.min_face_size = 32.0
    conf.threshold = 0.6
    conf.nms_thresholds = 0.4
    conf.scales = [320, 480]
    return conf
