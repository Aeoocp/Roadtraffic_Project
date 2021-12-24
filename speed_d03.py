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

  file_path = FLAGS.video
  video_capture = cv2.VideoCapture(file_path)

  w = int(video_capture.get(3))
  h = int(video_capture.get(4))
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(FLAGS.output, fourcc, 30, (w, h))
  frame_index = -1

  fps = 0.0
  fps_imutils = imutils.video.FPS().start()
  current_date = datetime.datetime.now().date()
  
  line_tc = []    # นับจำนวนIDที่ผ่านแต่ละเส้น
  intersect_info = [] # initialise intersection list
  for ll in range(l):
    line_tc.append([0,0]) # นับจำนวน ID ที่ตรวจจับได้ของแต่ละเส้น
    intersect_info.append([])
  line1_ac = deque(maxlen=50) # temporary memory for storing counted IDs forLine1
  line2_ac = deque(maxlen=50) # temporary memory for storing counted IDs forLine2
  memory = {}     # เก็บว่าเคยพิจารณาIDนี้ไปหรือยัง + ไว้เก็บmidpointไม่เกิน 2 จุด
  time_mem = {}   # เก็บframeที่IDนั้นๆผ่านของแต่ละเส้น
  speed_list = [] # ลิสความเร็วทั้งหมดที่คำนวณได้
  speed_avg = 0   # ค่าเฉลี่ยความเร็วทั้งหมด
  
  ret, frame = video_capture.read()  # frame shape 640*480*3
  #สร้างเส้นผ่าน
  frameY = frame.shape[0] #360
  frameX = frame.shape[1] #640
  line1 = []
  line2 = []
  #สร้างเส้น1,2ของถนนแต่ละเส้น
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
#    print("frame", frame_index+1)
    ret, frame = video_capture.read()  # frame shape 640*480*3
  
    if ret != True:
      break
 
    # วาดเส้นทั้งหมดลงใน frame
    for ll in range(l):
      line_1 = line1[ll]
      cv2.line(frame, line_1[0], line_1[1], (255, 255, 255), 2)
      line_2 = line2[ll]
      cv2.line(frame, line_2[0], line_2[1], (255, 255, 255), 2)
    
    t1 = time.time()
    
    image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
    if frame_index+1 < len(boxes_s):
      boxes = boxes_s[frame_index+1]
      confidence = confidence_s[frame_index+1]
      classes = classes_s[frame_index+1]

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
      
      #วนเช็คการตัดในแต่ละเส้น
      for ll in range(l):
        line_o = line1[ll]
        # เช็คการตัดเส้น
        TC1 = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line_o[0] ,line_o[1])
        #ถ้าตัดและไม่เคยผ่านเส้น1
        if TC1 and (track.track_id not in line1_ac):
          if track.track_id not in time_mem:
            time_mem[track.track_id] = []
          time_mem[track.track_id].append(frame_index+1)
          line_tc[ll][0] += 1
          # draw alert line
          cv2.line(frame, line_o[0], line_o[1], (0, 0, 255), 2)
          line1_ac.append(track.track_id)  # เช็คว่า ID นี้ผ่านเส้นนี้แล้ว
          intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
          intersect_info[ll].append([track_cls, origin_midpoint, intersection_time])
        
        line_o = line2[ll]
        TC2 = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line_o[0] ,line_o[1])
        if TC2 and (track.track_id not in line2_ac):
          if track.track_id not in time_mem:
            time_mem[track.track_id] = []
          time_mem[track.track_id].append(frame_index+1)
          line_tc[ll][1] += 1
          # draw alert line
          cv2.line(frame, line_o[0], line_o[1], (0, 0, 255), 2)
          line2_ac.append(track.track_id)  # Set already counted for ID to true.
          intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
          intersect_info[ll].append([track_cls, origin_midpoint, intersection_time])
        
        if len(time_mem[track.track_id]) == 2:
          time1 = time_mem[track.track_id][0]
          time2 = time_mem[track.track_id][1]
          distance = 40 #ระยะทางหน่วยเมตร 
          realtime = (time2-time1)/30 # แปลงเวลาในหน่วยเฟรมเป็นวินาที
          speed = (distance/realtime)*3.6 # คำนวณและแปลงหน่วยเป็นกิโลเมตรต่อชั่วโมง
          speed_list.append(speed)
          savg = 0
          co = len(speed_list)
          for s in speed_list:
            savg += s
          speed_avg = s/co
          print("Frame:",frame_index ," ID:" ,track.track_id ," speed:" ,speed)
          
      cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
      cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 1.5e-3 * frame.shape[0], (0, 255, 0), 2)
      if track.track_id in speed_list:
        cv2.putText(frame, str(speed_list[track.track_id]), (int(bbox[2]), int(bbox[1])), 0, 1.5e-3 * frame.shape[0], (0, 0, 255), 2)
    # Delete memory of old tracks.
    # This needs to be larger than the number of tracked objects in the frame.  
    if len(memory) > 50:
      del memory[list(memory)[0]]

    # Draw total count.
    yy = 0.1 * frame.shape[0]
    for ll in range(l):
      xx = ll+1
      cv2.putText(frame, "Total{}: {},{}".format(str(xx),str(line_tc[ll][0]),str(line_tc[ll][1])), (int(0.05 * frame.shape[1]), int(yy)), 0,
                1.5e-3 * frame.shape[0], (0, 255, 255), 2)
      yy += 0.1 * frame.shape[0]
      print("Total",xx,": ",line_tc[ll])
      
    cv2.putText(frame, "frame_index {}".format(str(frame_index+1)), (int(0.5 * frame.shape[1]), int(0.9 * frame.shape[0])), 0,
                  1.5e-3 * frame.shape[0], (255, 255, 255), 2)
    cv2.putText(frame, "speed_avg {}".format('%.2f' % speed_avg), (int(0.5 * frame.shape[1]), int(0.1 * frame.shape[0])), 0,
                  1.5e-3 * frame.shape[0], (255, 255, 255), 2)
    
    out.write(frame)
    frame_index = frame_index + 1

    fps_imutils.update()
    fps = (fps + (1. / (time.time() - t1))) / 2

    # Press Q to stop!
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  fps_imutils.stop()
  print('imutils FPS: {}'.format(fps_imutils.fps()))

  video_capture.release()
  out.release()
  cv2.destroyAllWindows()
    
if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass
