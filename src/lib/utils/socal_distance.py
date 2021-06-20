import copy
import cv2, numpy as np
import torch
from itertools import combinations
from .deepsocial import birds_eye, birds_eye2, Euclidean_distance, Euclidean_distance_seprate, Apply_trackmap, ColorGenerator, center_of_2box
from .utils import get_four_points, get_two_points
from lib.face_detector.Tracker import Tracker
from lib.face_detector.config import detector_config
from lib.model.mobilenetv3 import mobilenetv3
from PIL import Image
from torchvision import transforms
from shapely import geometry
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
def projection_4_i2b(M, p):
  px = (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])
  py = (M[1][0] * p[0] + M[1][1] * p[1] + M[1][2]) / (M[2][0] * p[0] + M[2][1] * p[1] + M[2][2])

  return ( int(px), int(py))
class Socal_Distance(object):
  def __init__(self):
    self._centroid_dict = dict()
    self._numberOFpeople = list()
    self._greenZone = list()
    self._redZone = list()
    self._yellowZone = list()
    self._final_redZone = list()
    self._relation = dict()
    self._couples = dict()
    self.CorrectionShift  = 1                    # Ignore people in the margins of the video
    self.HumanHeightLimit = 200
    ######################## Frame number
    self.StartFrom  = 0 
    EndAt      = 500                       #-1 for the end of the video
    ######################## (0:OFF/ 1:ON) Outputs
    self.CouplesDetection    = 1                # Enable Couple Detection 
    self.DTC                 = 1                # Detection, Tracking and Couples 
    self.SocialDistance      = 1
    self.CrowdMap            = 1
    # MoveMap             = 0
    # ViolationMap        = 0             
    # RiskMap             = 0
    ######################## Units are Pixel
    
    self.colorPool = ColorGenerator(size = 3000)
    self._is_init = False
    self.mask_track = {}
    self._init_face_detect()
    self._init_facemask()
  def is_init(self):
    return self._is_init
  
  def _init_face_detect(self):
    _config = detector_config()
    self.detector = Tracker(_config)
  def _init_facemask(self):
    checkpoint = torch.load('checkpoint/mask_detection.pth.tar')
    self.model_facemask = mobilenetv3().cuda()
    self.model_facemask.load_state_dict(checkpoint['state_dict'])
    self.model_facemask.eval()


  def face_detect(self, image, with_landmark = True):
    if len(image.shape) == 2:
        image = np.stack([image] * 3, 2)
    elif image.shape[2] ==4:
        image = image[:,:,:3]
    results = self.detector.detec_and_track(image, with_landmark)
    return results
  

  def init_transform_matrix(self, frame):
    height, width, _ = frame.shape
    pts_src = get_four_points(frame) #polygon detect and view 
    pts_dis = get_two_points(frame) #get distance 2M
    self.calibration = pts_src
    self.poly_bounds = [pts_src[0], pts_src[1], pts_src[3], pts_src[2]]
    self.polygon = geometry.Polygon(self.poly_bounds)
    # with open("../conf/config_birdview.yml", "r") as ymlfile:
    #   cfg = yaml.load(ymlfile)
    #   for section in cfg:
    #     width = int(cfg["image_parameters"]["width_og"])
    #     height = int(cfg["image_parameters"]["height_og"])
    #     img_path = cfg["image_parameters"]["img_path"]
    #     # size_frame = cfg["image_parameters"]["size_frame"]
    b_height = height
    b_width = int(0.65*height)
    self.b_height = b_height

    self.b_width = b_width
    self.e = birds_eye2(self.calibration, b_width, b_height, frame)
    # self.calibration      = [[180,162],[618,0],[552,540],[682,464]]
    self.d_thresh = self.e.distance_estimate(np.array([pts_dis]))
    scale = 1
    self.ViolationDistForIndivisuals = self.d_thresh
    self.ViolationDistForCouples     = int(1.15*self.d_thresh)
    ####
    self.CircleradiusForIndivsual    = int(0.5*self.d_thresh)
    self.CircleradiusForCouples      = int(0.615* self.d_thresh)
    ######################## 
    self.MembershipDistForCouples    = (int(0.6*self.d_thresh) , int(0.4*self.d_thresh)) # (Forward, Behind) per Pixel
    self.MembershipTimeForCouples    = 35        # Time for considering as a couple (per Frame)
    self._trackMap = np.zeros((b_height, b_width, 3), dtype=np.uint8)
    self._crowdMap = np.zeros((b_height, b_width), dtype=np.int) 
    
    # self.e = birds_eye(frame, self.calibration)
    self._is_init = True
    
  def inference_facemask(self, image, b):
    #img = Image.open(image_path).convert('RGB')
    im_height, im_width = image.shape[:2]
    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(max([w, h])* 1.0)
    cx = x1 + w//2
    cy = y1 + h//2
    x1 = cx - size//2
    x2 = x1 + size
    y1 = cy - size//2
    y2 = y1 + size
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)
    edx = max(0, x2 - im_width)
    edy = max(0, y2 - im_height)
    x2 = min(im_width, x2)
    y2 = min(im_height, y2)
    cropped = image[y1:y2, x1:x2]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        print('copyMakeBorder')
        print('dy, edy, dx, edx: ', dy, edy, dx, edx)
        cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, value=[0,0,0])
    cropped = cv2.resize(cropped, (96, 96))

    img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    img = img.resize((96, 96), Image.ANTIALIAS)
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.cuda().float()

    output = self.model_facemask(img)
    softmax_output = torch.softmax(output, dim=-1)

    mask_prob = softmax_output[0][1]
    nomask_porb = softmax_output[0][0]
    return mask_prob, nomask_porb

  def find_relation(self, centroid_dict, criteria, redZone):
    pairs = list()
    memberships = dict()
    for p1 in redZone:
      for p2 in redZone:
        if p1 != p2:
          distanceX, distanceY = Euclidean_distance_seprate(centroid_dict[p1], centroid_dict[p2])
          if p1 < p2:
            pair = (p1, p2)
          else:
            pair = (p2, p1)
          if self._couples.get(pair):
            distanceX = distanceX * 0.6
            distanceY = distanceY * 0.6
          if distanceX < criteria[0]:
            if distanceY < criteria[1]:
              if memberships.get(p1):
                memberships[p1].append(p2)
              else:
                memberships[p1] = [
                p2]
              if pair not in pairs:
                pairs.append(pair)

    relation = dict()
    for pair in pairs:
      if self._relation.get(pair):
        self._relation[pair] += 1
        relation[pair] = self._relation[pair]
      else:
        self._relation[pair] = 1

    obligation = {}
    for p in memberships:
      top_relation = 0
      for secP in memberships[p]:
        if p < secP:
          pair = (
          p, secP)
        else:
          pair = (
          secP, p)
        if relation.get(pair):
          if top_relation < relation[pair]:
            top_relation = relation[pair]
            obligation[p] = secP

    couple = dict()
    for m1 in memberships:
      for m2 in memberships:
        if m1 != m2:
          if obligation.get(m1):
            if obligation.get(m2):
              if obligation[m1] == m2:
                if obligation[m2] == m1:
                  if m1 < m2:
                    pair = (
                    m1, m2)
                  else:
                    pair = (
                    m2, m1)
                  couple[pair] = relation[pair]
    return couple

  def init_track(self, results):
    for item in results:
      if item['score'] > self.opt.new_thresh:
        self.id_count += 1
        # active and age are never used in the paper
        item['active'] = 1
        item['age'] = 1
        item['tracking_id'] = self.id_count
        if not ('ct' in item):
          bbox = item['bbox']
          item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.tracks.append(item)
        self.nID = 10000
        self.embedding_bank = np.zeros((self.nID, 128))
        self.cat_bank = np.zeros((self.nID), dtype=np.int)

  def get_centroid(self, detections, ids, image):
    # e = birds_eye(image.copy(), calibration)
    centroid_dict = dict()
    now_present = list()
    if len(detections) > 0:   
      for (d, p) in zip(detections, ids):
        p = int(p)
        now_present.append(p)
        xmin, ymin, xmax, ymax = d[0], d[1], d[2], d[3]
        w = xmax - xmin
        h = ymax - ymin
        x = xmin + w/2
        y = ymax - h/2
        point = geometry.Point(x, ymax)
        

        if not self.polygon.contains(point):
          continue
        if h < self.HumanHeightLimit:
          # overley = image
          
          bird_x, bird_y = self.e.projection_on_bird((x, ymax))
          
          # if CorrectionShift:
          #     if checkupArea(overley, 0, 0.25, (x, ymin)):
          #         continue
          
          # e.setImage(overley)
          center_bird_x, center_bird_y = self.e.projection_on_bird((x, ymin))
          centroid_dict[p] = (
                      int(bird_x), int(bird_y),
                      int(x), int(ymax), 
                      int(xmin), int(ymin), int(xmax), int(ymax),
                      int(center_bird_x), int(center_bird_y))
    return centroid_dict, image
    
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

  def find_redGroups(self, centroid_dict, criteria, redZone, couples):
    # e = birds_eye(img, calibration)
    redGroups = list()
    for p1, p2 in couples:
      x, y, xmin, ymin, xmax, ymax = couples[(p1, p2)]['box']
      centerGroup_bird = self.e.projection_on_bird((x, ymax))
      for p, box in centroid_dict.items():
        if p != p1:
          if p != p2:
            center_bird = (box[0], box[1])
            distance = Euclidean_distance(center_bird, centerGroup_bird)
            if distance < criteria:
              redGroups.append(p1)
              redGroups.append(p2)
              redGroups.append(p)

    yellowZone = list()
    for p1, p2 in couples:
      if p1 not in redGroups:
        if p2 not in redGroups:
          yellowZone.append(p1)
          yellowZone.append(p2)

    red_without_yellowZone = list()
    for id, box in centroid_dict.items():
      if id in redZone:
        if id not in yellowZone:
          red_without_yellowZone.append(id)

    return (yellowZone, red_without_yellowZone, redGroups)

  def apply_ellipticbound(self, img, centroid_dict, red, green, yellow, final_redZone, coupleZone, couples, Single_radius, Couples_radius):
    RedColor = (0, 0, 255)
    GreenColor = (0, 255, 0)
    YellowColor = (0, 220, 255)
    BirdBorderColor = (255, 255, 255)
    BorderColor = (220, 220, 220)
    Transparency = 0.55
    e = birds_eye2(self.calibration, self.b_width, self.b_height, img)
    e.setImage(img)
    im_h, im_w = img.shape[:2]
    overlay = e.convrt2Bird(img)
    cv2.imwrite("overlay.jpg", overlay)
    for idx, box in centroid_dict.items():
      center_bird = (
      box[0], box[1])
      if idx in green:
        cv2.circle(overlay, center_bird, Single_radius, GreenColor, -1)
      if idx in red:
        if idx not in coupleZone:
          cv2.circle(overlay, center_bird, Single_radius, RedColor, -1)

    for p1, p2 in couples:
      x, y, xmin, ymin, xmax, ymax = couples[(p1, p2)]['box']
      centerGroup_bird = e.projection_on_bird((x, ymax))
      if p1 in yellow:
        if p2 in yellow:
          cv2.circle(overlay, centerGroup_bird, Couples_radius, YellowColor, -1)
        if p1 in final_redZone:
          if p2 in final_redZone:
            cv2.circle(overlay, centerGroup_bird, Couples_radius, RedColor, -1)

    e.setBird(overlay)
    temp_img = e.convrt2Image(overlay, [im_w, im_h])
    cv2.imwrite("temp_img.jpg", temp_img)
    # cv2.fillConvexPoly(image, np.array(self.calibration).astype(int), 0, 16)
    e.setImage(cv2.addWeighted(e.original, Transparency, temp_img, 1 - Transparency, 0))
    overlay = e.image
    for idx, box in centroid_dict.items():
      birdseye_origin = (
      box[0], box[1])
      
      circle_points = e.points_projection_on_image(birdseye_origin, Single_radius)
      
      if idx not in coupleZone:
        for x, y in circle_points:
          cv2.circle(overlay, (int(x), int(y)), 1, BorderColor, -1)

      ymin = box[5]
      ymax = box[7]
      origin = e.projection_on_image((box[0], box[1]))
      w = 3
      x = origin[0]
      top_left = (x - w, ymin)
      botton_right = (x + w, ymax)
      if idx in green:
        cv2.rectangle(overlay, top_left, botton_right, GreenColor, -1)
        cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)
      if idx in red:
        if idx not in coupleZone:
          cv2.rectangle(overlay, top_left, botton_right, RedColor, -1)
          cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)

    for p1, p2 in couples:
      x, y, xmin, ymin, xmax, ymax = couples[(p1, p2)]['box']
      birdseye_origin = e.projection_on_bird((x, ymax))
      circle_points = e.points_projection_on_image(birdseye_origin, Couples_radius)
      for x, y in circle_points:
        cv2.circle(overlay, (int(x), int(y)), 1, BorderColor, -1)

      origin = e.projection_on_image(birdseye_origin)
      w = 3
      x = origin[0]
      top_left = (x - w, ymin)
      botton_right = (x + w, ymax)
      if p1 in yellow:
        if p2 in yellow:
          cv2.rectangle(overlay, top_left, botton_right, YellowColor, -1)
          cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)
        if p1 in final_redZone:
          if p2 in final_redZone:
            cv2.rectangle(overlay, top_left, botton_right, RedColor, -1)
            cv2.rectangle(overlay, top_left, botton_right, BorderColor, 1)

    e.setImage(overlay)
    return (
    e.image, e.bird)
  def find_zone(self, centroid_dict, criteria):
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
  def step(self, image, bboxes, ids):
    results_face = []
    # cv2.imwrite("as.jpg", image)
    imh, imw  = image.shape[:2]
    # for (b, id) in zip(bboxes, ids):
    #   if id not in self.mask_track.keys():
    #     xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    #     if not ((xmin>=0 and xmin<imw) and (ymin>=0 and ymin<imh)
    #         and (xmax>=0 and xmax<imw) and (ymax>=0 and ymax<imw)):
    #       continue
    #     person_img = image[ymin:ymax, xmin:xmax]
    #     result = self.face_detect(person_img.copy())
    #     if result["num_face"]>0:
    #       if abs(result["poses"][0]["yaw"])<45:
    #         mask_prob, nomask_porb = self.inference_facemask(person_img, result["bboxs"][0])
    #         if mask_prob >= 0.8:
    #           result["face_mask"] = "mask"
    #           result["mask_prob"] = mask_prob
    #         else:
    #           result["face_mask"] = "unmask"
    #       else:
    #         result["face_mask"] = "nocheck"
        
    #       self.mask_track.update({id: result})
    #   else:
    #     result = self.mask_track[id]
    #   results_face.append((id, result))
    # print(results_face)
    cv2.polylines(image, [np.array(self.poly_bounds, np.int32)], True, (0, 255, 255), thickness=4)
    
    centroid_dict, partImage = self.get_centroid(bboxes, ids, image)
    self._centroid_dict.update(centroid_dict)
    redZone, greenZone = self.find_zone(centroid_dict, criteria=self.ViolationDistForIndivisuals)
    # self.e.setImage(image)
    if self.CouplesDetection:
        relation = self.find_relation(centroid_dict, self.MembershipDistForCouples, redZone)
        couples, coupleZone = self.find_couples(relation, self.MembershipTimeForCouples)
        yellowZone, final_redZone, redGroups = self.find_redGroups(centroid_dict, self.ViolationDistForCouples, redZone, couples )
    else:
        couples = []
        coupleZone = []
        yellowZone = []
        redGroups = redZone
        final_redZone = redZone
    # if "poses" in results.keys():  
    #   if results["poses"][0]["yaw"]>20:
    #       results["status"] = "left"
    #   elif results["poses"][0]["yaw"]<-20:
    #       results["status"] = "right"
    if self.DTC:
        DTC_image = image.copy()
        self._trackMap = Apply_trackmap(centroid_dict, self._trackMap, self.colorPool, 3)
        
        temp_img = self.e.convrt2Image(self._trackMap, [imw, imh])
        cv2.imwrite("temp_img1.jpg", image)
        # Black out polygonal area in destination image.
        image_cp = cv2.fillConvexPoly(image.copy(), np.array(self.poly_bounds).astype(int), 0, 16)
        DTCShow = cv2.add(temp_img, image_cp) 
        cv2.imwrite("temp_img2.jpg", image)
        for id, box in centroid_dict.items():
            center_bird = box[0], box[1]
            if not id in coupleZone:
                cv2.rectangle(DTCShow, (box[4], box[5]), (box[6], box[7]), (0,255,0), 2)
                cv2.rectangle(DTCShow, (box[4], box[5]-13), (box[4]+len(str(id))*10, box[5]), (0,200,255),-1)
                cv2.putText(DTCShow, str(id), (box[4]+2, box[5]-2), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0), 1, cv2.LINE_AA)
        for coupled in couples:
            p1 , p2 = coupled
            couplesID = couples[coupled]['id']
            couplesBox = couples[coupled]['box']
            cv2.rectangle(DTCShow, couplesBox[2:4], couplesBox[4:], (0,150,255), 4)
            loc = couplesBox[0] , couplesBox[3]
            offset = len(str(couplesID)*5)
            captionBox = (loc[0] - offset, loc[1]-13), (loc[0] + offset, loc[1])
            cv2.rectangle(DTCShow,captionBox[0],captionBox[1],(0,200,255),-1)
            wc = captionBox[1][0] - captionBox[0][0]
            hc = captionBox[1][1] - captionBox[0][1]
            cx = captionBox[0][0] + wc // 2
            cy = captionBox[0][1] + hc // 2
            textLoc = (cx - offset, cy + 4)
            cv2.putText(DTCShow, str(couplesID), (textLoc), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0),1, cv2.LINE_AA)
       
    if self.SocialDistance:
        cv2.imwrite("im.jpg", image)
        SDimage, birdSDimage = self.apply_ellipticbound(image.copy(), centroid_dict, redZone, greenZone, yellowZone, final_redZone, coupleZone, couples, self.CircleradiusForIndivsual, self.CircleradiusForCouples)
    return SDimage, birdSDimage, DTCShow

    

 

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)
