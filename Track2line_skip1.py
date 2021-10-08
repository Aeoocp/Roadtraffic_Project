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
import imutils.video
from videocaptureasync import VideoCaptureAsync

from collections import Counter
from collections import deque 
import datetime
import math

warnings.filterwarnings('ignore')

def main(_argv):
  classes_s,confidence_s,boxes_s = splitFile.spilttxt(FLAGS.text)
  
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
    out = cv2.VideoWriter('output_2line_skip1.avi', fourcc, 30, (w, h))
    frame_index = -1

  fps = 0.0
  fps_imutils = imutils.video.FPS().start()
  current_date = datetime.datetime.now().date()
  count_dict = {}
  
  total_counter = 0
  total_counter2 = 0
  
  class_counter = Counter()  # store counts of each detected class
  class_counter2 = Counter()
  already_counted = deque(maxlen=50)  # temporary memory for storing counted IDs
  intersect_info = []  # initialise intersection list
  intersect_info2 = []
    
  memory = {}
  memory2 = {}
  maxId = 0

  while True:
    print("frame", frame_index+1)

    ret, frame = video_capture.read()  # frame shape 640*480*3

    if ret != True:
      break

    t1 = time.time()

    image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
    if frame_index+1 < len(boxes_s):
      boxes = boxes_s[frame_index+1]
      confidence = confidence_s[frame_index+1]
      classes = classes_s[frame_index+1]

    features = encoder(frame, boxes)
    detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                    zip(boxes, confidence, classes, features)]
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
    line = [(int(0.3 * frameX), int(0.8 * frameY)), (int(0.55 * frameX), int(0.85 * frameY))]
    cv2.line(frame, line[0], line[1], (255, 255, 255), 2)   #(image, start_point, end_point, color, thickness)
    line2 = [(int(0.05 * frameX), int(0.6 * frameY)), (int(0.2 * frameX), int(0.65 * frameY))]
    cv2.line(frame, line2[0], line2[1], (255, 255, 255), 2)   #(image, start_point, end_point, color, thickness)
    
    if(frame_index%2 == 1):
      for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
          continue
        bbox = track.to_tlbr()
        track_cls = track.cls

        if track_cls == "car":
          cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
          cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 1.5e-3 * frame.shape[0], (0, 255, 0), 1)
          cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0], (0, 255, 0), 1)
        elif track_cls == "truck":
          cv2.rectangle(frame, (int(bbox[0]+5), int(bbox[1]+5)), (int(bbox[2]+5), int(bbox[3]+5)), (0, 0, 255), 2)
          cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1]-25)), 0, 1.5e-3 * frame.shape[0], (0, 0, 255), 1)
          cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0], (0, 0, 255), 1)
        else:
          cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
          cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 1.0e-3 * frame.shape[0], (255, 0, 0), 1)
          cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0], (255, 0, 0), 1)

        midpoint = track.tlbr_midpoint(bbox)
        # get midpoint respective to botton-left
        origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])

        if track.track_id not in memory:
          memory[track.track_id] = deque(maxlen=2)  

        memory[track.track_id].append(midpoint)
        previous_midpoint = memory[track.track_id][0]
        origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])

        if track_cls == "car":
          cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)
        elif track_cls == "truck":
          cv2.line(frame, midpoint, previous_midpoint, (0, 0, 255), 2)
        else:
          cv2.line(frame, midpoint, previous_midpoint, (255, 0, 0), 2)

        TC = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line[0] ,line[1])
        TC2 = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line2[0] ,line2[1])
        if TC and (track.track_id not in already_counted):
          class_counter[track_cls] += 1
          total_counter += 1
          # draw alert line
          cv2.line(frame, line[0], line[1], (0, 255, 255), 2)
          already_counted.append(track.track_id)  # Set already counted for ID to true.
          intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
          intersect_info.append([track_cls, origin_midpoint, intersection_time])
        elif TC2 and (track.track_id not in already_counted):
          class_counter2[track_cls] += 1
          total_counter2 += 1
          # draw alert line
          cv2.line(frame, line2[0], line2[1], (0, 255, 255), 2)
          already_counted.append(track.track_id)  # Set already counted for ID to true.
          intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
          intersect_info2.append([track_cls, origin_midpoint, intersection_time])

      # Delete memory of old tracks.
      # This needs to be larger than the number of tracked objects in the frame.  
      if len(memory) > 50:
        del memory[list(memory)[0]]

      # Draw total count.
      cv2.putText(frame, "Total: {}".format(str(total_counter)), (int(0.8 * frame.shape[1]), int(0.1 * frame.shape[0])), 0,
                  1.5e-3 * frame.shape[0], (0, 255, 255), 2)
      cv2.putText(frame, "Total2: {}".format(str(total_counter2)), (int(0.05 * frame.shape[1]), int(0.1 * frame.shape[0])), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)

      # display counts for each class as they appear
      y = 0.2 * frame.shape[0]
      y2 = 0.2 * frame.shape[0]
      for cls in class_counter:
        class_count = class_counter[cls]
        cv2.putText(frame, str(cls) + " " + str(class_count), (int(0.8 * frame.shape[1]), int(y)), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)
        y += 0.05 * frame.shape[0]
      for cls in class_counter2:
        class_count2 = class_counter2[cls]
        cv2.putText(frame, str(cls) + " " + str(class_count2), (int(0.05 * frame.shape[1]), int(y2)), 0,
                    1.5e-3 * frame.shape[0], (0, 255, 255), 2)
        y2 += 0.05 * frame.shape[0]
        
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
  except SystemExit:
    pass
