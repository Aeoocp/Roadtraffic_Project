from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image

from absl import app, flags, logging
from absl.flags import FLAGS
flags.DEFINE_string('text','/content/Traffuc_counting/input_txt/hlp/080841-03.txt','input text')
flags.DEFINE_string('video', '/content/Traffuc_counting/video/hlp/080841-03.mkv', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', '/content/Traffuc_counting/output_test.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools import import_txt
import imutils.video
from videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')

def main(_argv):
  input_T = FLAGS.text
  file1 = open(file, "r")
  FileContent = file1.read()
  AS = FileContent.split()
  
  inform = []
  car_num = []
  c = -1
  y = 0
  for x in range(len(AS)-1):
    xx = AS[x].split(',')
    if y == 0:
      if AS[x] == "#":
        c = c+1
        car_num.append(AS[x+3])
        skip = x+3
        inform.append([])
      elif len(xx) == 2:
        if xx[1] != '':
          y = 1
          inform[c].append(AS[x]+AS[x+1])
      elif  x > skip:
        inform[c].append(AS[x])
    else:
        y = 0

  boxes_s = []
  confidence_s = []
  classes_s = []

  for a in range(len(inform)):
    subclass = [] 
    subconfidence = []
    subbox = []
    for b in range(len(inform[a])):
      eachBox = inform[a][b]
      SEachBox = eachBox.split(',')
      Frame = SEachBox[0]
      subclass.append(SEachBox[1])
      subconfidence.append(float(SEachBox[6]))
      ssubbox = []
      ssubbox.append(float(SEachBox[2]))
      ssubbox.append(float(SEachBox[3]))
      w = float(SEachBox[4]) - float(SEachBox[2])
      h = float(SEachBox[5]) - float(SEachBox[3])
      ssubbox.append(w)
      ssubbox.append(h)
      subbox.append(ssubbox)
    classes_s.append(subclass)
    confidence_s.append(subconfidence)
    boxes_s.append(subbox)
  
  # Definition of the parameters
  max_cosine_distance = 0.3
  nn_budget = None
  nms_max_overlap = 1.0
  
  # Deep SORT
  model_filename = 'model_data/mars-small128.pb'
  encoder = gdet.create_box_encoder(model_filename, batch_size=1)
  
  metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
  tracker = Tracker(metric)
  
  show_detections = True
  writeVideo_flag = True
  asyncVideo_flag = False
    
  file_path = ' FLAGS.video '
  if asyncVideo_flag:
    video_capture = VideoCaptureAsync(file_path)
  else:
    video_capture = cv2.VideoCapture(file_path)
    
  if asyncVideo_flag:
      video_capture.start()
      
  if writeVideo_flag:
    if asyncVideo_flag:
      w = int(video_capture.cap.get(3))
      h = int(video_capture.cap.get(4))
    else:
      w = int(video_capture.get(3))
      h = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(FLAGS.output, fourcc, 30, (w, h))
    frame_index = 0

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    
    current_date = datetime.datetime.now().date()
    count_dict = {}
    total_counter = 0
    up_count = 0
    down_count = 0
    
    class_counter = Counter()  # store counts of each detected class
    already_counted = deque(maxlen=50)  # temporary memory for storing counted IDs
    intersect_info = []  # initialise intersection list
    
    memory = {}
    while True:
      print("frame", frame_index+1)
      ret, frame = video_capture.read()
      
      if ret != True:
        break
      
      t1 = time.time()
      
      image = Image.fromarray(frame[..., ::-1])
      if frame_index < len(boxes_s):
        boxes = boxes_s[frame_index]
        confidence = confidence_s[frame_index]
        classes = classes_s[frame_index]
        
      features = encoder(frame, boxes)
      detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in zip(boxes, confidence, classes, features)]
      # Run non-maxima suppression.
      boxes = np.array([d.tlwh for d in detections])
      scores = np.array([d.confidence for d in detections])
      classes = np.array([d.cls for d in detections])
      indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
      detections = [detections[i] for i in indices]
      
      # Call the tracker
      tracker.predict()
      tracker.update(detections)
      
      #สร้างและวาดเส้นผ่าน
      frameY = frame.shape[0] #360
      frameX = frame.shape[1] #640
      line = [(int(0.5 * frameX), int(0.5 * frameY)), (int(0.8 * frameX), int(0.5 * frameY))]
      cv2.line(frame, line[0], line[1], (0, 255, 255), 2)   #(image, start_point, end_point, color, thickness)
      
      for track in tracker.tracks:
          if not track.is_confirmed() or track.time_since_update > 1:
            continue
          
