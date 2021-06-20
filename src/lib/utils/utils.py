from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cv2
import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def mouse_handler(event, x, y, flags, data) :
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < data['num_point'] :
            data['points'].append([x,y])

def get_four_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    data['num_point'] = 4
    
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    
    return points
def get_two_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    data['num_point'] = 2
    
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    
    return points

def find_zone(centroid_dict, criteria):
    redZone = []
    greenZone = []
    for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
        distance = Euclidean_distance(p1[0:2], p2[0:2])
        if distance < criteria:
            if id1 not in redZone:
                redZone.append(int(id1))
            if id2 not in redZone:
                redZone.append(int(id2))

    for idx, box in centroid_dict.items():
        if idx not in redZone:
            greenZone.append(idx)
    return (redZone, greenZone)

def find_couples(self, relation, criteria):
    couples = dict()
    coupleZone = list()
    for pair in relation:
            proxTime = relation[pair]
            if proxTime > criteria:
                    coupleZone.append(pair[0])
                    coupleZone.append(pair[1])
                    couplesBox = center_of_2box(self._centroid_dict[pair[0]], self._centroid_dict[pair[1]])
                    if self._couples.get(pair):
                            couplesID = self._couples[pair]['id']
                            self._couples[pair]['box'] = couplesBox
                    else:
                            couplesID = len(self._couples) + 1
                            self._couples[pair] = {'id':couplesID,  'box':couplesBox}
                    couples[pair] = self._couples[pair]

    return (couples, coupleZone)