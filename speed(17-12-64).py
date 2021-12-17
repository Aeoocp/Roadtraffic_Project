from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image

from absl import app, flags, logging
from absl.flags import FLAGS
flags.DEFINE_string('text','/content/Traffic_counting/input_txt/hlp/080841-03.txt','input text')
flags.DEFINE_string('video', '/content/Traffic_counting/video/hlp/080841-03.mkv', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', '/content/Traffic_counting/output.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools import splitFile
from tools import CheckCrossLine
from tools import newLine
import imutils.video
from videocaptureasync import VideoCaptureAsync

from collections import Counter
from collections import deque 
import datetime
import math

warnings.filterwarnings('ignore')

def main(_argv):
  l,x1,y1,x2,y2 = newLine.createLineSpeed2()  #get lines position
  classes_s,confidence_s,boxes_s = splitFile.spilttxt(FLAGS.text)   #get track imformation
  
  # Definition of the parameters
  max_cosine_distance = 0.3
  nn_budget = None
  nms_max_overlap = 1.0

  # Deep SORT
  model_filename = 'model_data/mars-small128.pb'
  encoder = gdet.create_box_encoder(model_filename, batch_size=1) #function
  
  metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
  tracker = Tracker(metric)

  show_detections = True
  writeVideo_flag = True
  asyncVideo_flag = False

  file_path = FLAGS.video
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
    frame_index = -1

  fps = 0.0
  fps_imutils = imutils.video.FPS().start()
  current_date = datetime.datetime.now().date()
  
  total_counter = []
  class_counter = []  # store counts of each detected class
  intersect_info = [] # initialise intersection list
  for ll in range(l):
    total_counter.append(0)
    class_counter.append(Counter())
    intersect_info.append([])
  already_counted1 = deque(maxlen=50) # temporary memory for storing counted IDs forLine1
  already_counted2 = deque(maxlen=50) # temporary memory for storing counted IDs forLine2
  memory = {}
  speed_mem = {}
  speed_mem_list = {}
  speed_list = []   # store final speed
  
  ret, frame = video_capture.read()  # frame shape 640*480*3
  #สร้างเส้นผ่าน
  frameY = frame.shape[0] #360
  frameX = frame.shape[1] #640
  line1 = []
  line2 = []
  for ll in range(l):
    xb1 = float(x1[ll*2])
    yb1 = float(y1[ll*2])
    xe1 = float(x1[ll*2+1])
    ye1 = float(y1[ll*2+1])
    line_c1 = [(int(xb1 * frameX), int(yb1* frameY)), (int(xe1 * frameX), int(ye1 * frameY))]
    line1.append(line_c1)
    xb2 = float(x2[ll*2])
    yb2 = float(y2[ll*2])
    xe2 = float(x2[ll*2+1])
    ye2 = float(y2[ll*2+1])
    line_c2 = [(int(xb2 * frameX), int(yb2* frameY)), (int(xe2 * frameX), int(ye2 * frameY))]
    line2.append(line_c2)
    
  while True:
    ret, frame = video_capture.read()  # frame shape 640*480*3

    if ret != True:
      break
 
    #วาดเส้นผ่าน
    for ll in range(l):
      line_1 = line1[ll]
      cv2.line(frame, line_1[0], line_1[1], (255, 255, 255), 2)
      line_2 = line2[ll]
      cv2.line(frame, line_2[0], line_2[1], (255, 255, 255), 2)
    
    b_size = 0
    t1 = time.time()
    
    image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
    if frame_index+1 < len(boxes_s):
      boxes = boxes_s[frame_index+1]
      confidence = confidence_s[frame_index+1]
      classes = classes_s[frame_index+1]
    b_size = len(boxes)

    features = encoder(frame, boxes)
    # represents a bounding box detection in a single image
    detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                    zip(boxes, confidence, classes, features)]
    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])        # List ของ [x y w h] ในแต่ละเฟรม
    scores = np.array([d.confidence for d in detections]) # confidence
    classes = np.array([d.cls for d in detections])       # class
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) #กรองเฟรมที่ซ้อนทับกันออก
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()   # ได้ mean vector และ covariance matrix จาก Kalman filter prediction step
    tracker.update(detections)

    for track in tracker.tracks:
      if not track.is_confirmed() or track.time_since_update > 1:
        continue
      bbox = track.to_tlbr()    # (min x, miny, max x, max y)
      track_cls = track.cls

      midpoint = track.tlbr_midpoint(bbox)
      # get midpoint respective to botton-left
      origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])

      if track.track_id not in memory:
        memory[track.track_id] = deque(maxlen=2)  

      memory[track.track_id].append(midpoint)
      previous_midpoint = memory[track.track_id][0]
      origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])
      speedList = []
      for ll in range(l):
        line_o = line1[ll]
        TC1 = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line_o[0] ,line_o[1])
        if TC1 and (track.track_id not in already_counted1):
          if track.track_id not in speed_mem:
            speed_mem[track.track_id] = deque(maxlen=2)
            speed_mem[track.track_id].append(frame_index+1)
          class_counter[ll][track_cls] += 1
          total_counter[ll] += 1
          # draw alert line
          cv2.line(frame, line_o[0], line_o[1], (0, 0, 255), 2)
          already_counted1.append(track.track_id)  # Set already counted for ID to true.
          intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
          intersect_info[ll].append([track_cls, origin_midpoint, intersection_time])
        line_o = line2[ll]
        TC2 = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line_o[0] ,line_o[1])
        if TC2 and (track.track_id not in already_counted2) and (track.track_id in already_counted1):
          speed_mem[track.track_id].append(frame_index+1)
          class_counter[ll][track_cls] += 1
          total_counter[ll] += 1
          # draw alert line
          cv2.line(frame, line_o[0], line_o[1], (0, 0, 255), 2)
          already_counted2.append(track.track_id)  # Set already counted for ID to true.
          intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
          intersect_info[ll].append([track_cls, origin_midpoint, intersection_time])
        
        if track.track_id in speed_mem:
          if(len(speed_mem[track.track_id])==2):
            trackTime1 = speed_mem[track.track_id].popleft()
            trackTime2 = speed_mem[track.track_id].popleft()
            distance = 4.5 #ระยะทางหน่วยเมตร 
            time_tract = (trackTime2-trackTime1)/30 #เวลาที่จับได้ในหน่วยวินาที
            if(time_tract!=0):
              speed = (distance/time_tract)*3.6 #คำนวณและแปลงหน่วยเป็นกิโลเมตรต่อชั่วโมง
              print("ID:",track.track_id," speed: ",speed)
              speed_mem_list[track.track_id] = speed
              speed_list.append(speed)
          
      cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[3])), 0, 1.0e-3 * frame.shape[0], (255, 255, 255), 2)
      cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
      if track.track_id in speed_mem_list:
        cv2.putText(frame, str(speed_mem_list[track.track_id]), (int(bbox[2]), int(bbox[3])), 0, 1.5e-3 * frame.shape[0], (0, 0, 255), 2)

    # Delete memory of old tracks.
    # This needs to be larger than the number of tracked objects in the frame.  
    if len(memory) > 50:
      del memory[list(memory)[0]]

    # Draw total count.
    yy = 0.1 * frame.shape[0]
    for ll in range(l):
      xx = ll+1
      cv2.putText(frame, "Total{}: {}".format(str(xx),str(total_counter[ll])), (int(0.05 * frame.shape[1]), int(yy)), 0,
                1.5e-3 * frame.shape[0], (0, 255, 255), 2)
      yy += 0.1 * frame.shape[0]
      
    cv2.putText(frame, "frame_index {}".format(str(frame_index+1)), (int(0.5 * frame.shape[1]), int(0.9 * frame.shape[0])), 0,
                  1.5e-3 * frame.shape[0], (255, 255, 255), 2)
    
    if writeVideo_flag:
        # save a frame
        out.write(frame)
        frame_index = frame_index + 1

    fps_imutils.update()

    if not asyncVideo_flag:
      fps = (fps + (1. / (time.time() - t1))) / 2

    # Press Q to stop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  fps_imutils.stop()
  print('imutils FPS: {}'.format(fps_imutils.fps()))
  
  print("Total count : ",total_counter[ll])
  print("speed_list",speed_list)
  print("speed_list_side",len(speed_list))
  if asyncVideo_flag:
    video_capture.stop()
  else:
    video_capture.release()

  if writeVideo_flag:
        out.release()

  cv2.destroyAllWindows()
    
if __name__ == '__main__':
  try:
    app.run(main)
