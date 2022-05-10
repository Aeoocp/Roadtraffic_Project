#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from loguru import logger

import cv2
import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess

import argparse
import os
import time
import numpy as np
import mmglobal

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from tools import CheckCrossLine

from collections import deque
import datetime
import math

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("--path", default="./assets/dog.jpg", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",
                        help="whether to save the inference result of image/video", )
    # exp file
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file", )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="cpu", type=str, help="device to run our model, can either be cpu or gpu", )
    parser.add_argument("--type", default="C", type=str, help="C,S,CL,ALL", )
    parser.add_argument("--skipframe", default=1, type=int, help="frame skip")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--lineN", default=None, type=int, help="Line number")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--lineC", default=None, type=str, help="Counter line")
    parser.add_argument("--lineS", default=None, type=str, help="Speed line")
    parser.add_argument("--lineCL", default=None, type=str, help="Change len line")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision evaluating.", )
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true",
                        help="Fuse conv and bn for testing.", )
    parser.add_argument("--trt", dest="trt", default=False, action="store_true",
                        help="Using TensorRT model for testing.", )
    return parser

def get_image_list(path):
  image_names = []
  for maindir, subdir, file_name_list in os.walk(path):
      for filename in file_name_list:
          apath = os.path.join(maindir, filename)
          ext = os.path.splitext(apath)[1]
          if ext in IMAGE_EXT:
              image_names.append(apath)
  return image_names
  
  def for1minute(dt1, dt2):
    m1 = int(dt1.strftime("%M"))
    m2 = int(dt2.strftime("%M"))
    s1 = int(dt1.strftime("%S"))
    s2 = int(dt1.strftime("%S"))
    if m2 - m1 < 0:
        m2 = m2+60
    if m2 - m1 > 2:
        return True
    elif m2 - m1 == 1:
        if s2 >= s1:
            return True
        else:
            return False
    else:
        return False
   
def for5minute(dt1, dt2):
    m1 = int(dt1.strftime("%M"))
    m2 = int(dt2.strftime("%M"))
    s1 = int(dt1.strftime("%S"))
    s2 = int(dt1.strftime("%S"))
    if m2 - m1 < 0:
        m2 = m2+60
    if m2 - m1 > 6:
        return True
    elif m2 - m1 == 5:
        if s2 >= s1:
            return True
        else:
            return False
    else:
        return False
    
def for10minute(dt1, dt2):
    m1 = int(dt1.strftime("%M"))
    m2 = int(dt2.strftime("%M"))
    s1 = int(dt1.strftime("%S"))
    s2 = int(dt1.strftime("%S"))
    if m2 - m1 < 0:
        m2 = m2+60
    if m2 - m1 > 11:
        return True
    elif m2 - m1 == 10:
        if s2 >= s1:
            return True
        else:
            return False
    else:
        return False
      
def estimatespeed(location1, location2, ppm, fs, skipframe):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    d_meters = d_pixels / ppm
    speed = (d_meters * fs * 3.6) / skipframe
    return speed
  
def counting(frame, track, memory, line, ac, class_n, class_counter):
    track_box = track.to_tlbr()  # (min x, miny, max x, max y)
    track_cls = track.cls
    midpoint = track.tlbr_midpoint(track_box)
    if track.track_id not in memory:
        memory[track.track_id] = deque(maxlen=2)
    memory[track.track_id].append(midpoint)
    previous_midpoint = memory[track.track_id][0]
    tc = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line[0], line[1])
    if tc and (track.track_id not in ac):
        #cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
        cls_id = int(track_cls.item())
        cls_name = class_n[cls_id]
        if cls_name == "bus":
            class_counter[0] += 1
        elif cls_name == "car":
            class_counter[1] += 1
        elif cls_name == "motorcycle":
            class_counter[2] += 1
        elif cls_name == "truck":
            class_counter[3] += 1
        class_counter[4] += 1
        ac.append(track.track_id)  # Set already counted for ID to true.
    dt = datetime.datetime.now()
    return [memory, ac, class_counter, dt]
  
def speeddetect(frame, track, fps, memory, line, ac, class_n, crossline, speed_adv, stopcid, stopcspeed, sf):
    track_box = track.to_tlbr()  # (min x, miny, max x, max y)
    track_cls = track.det_cls
    cls_name = class_n[int(track_cls.item())]
    # Calculate ppm
    x_2 = math.pow(int(track_box[2]) - int(track_box[0]), 2)
    y_2 = math.pow(int(track_box[3]) - int(track_box[1]), 2)
    ppm = math.sqrt(x_2 + y_2)
    if cls_name == "bus":
        ppm = ppm / 12
    elif cls_name == "car":
        ppm = ppm / 6
    elif cls_name == "motorcycle":
        ppm = ppm / 2
    # ไม่นับรถบรรทุกเพราะขนาดไม่แน่นอน มีหลายขนาดที่ต่างกันมาก
    midpoint = track.tlbr_midpoint(track_box)
    if track.track_id not in memory:
        memory[track.track_id] = deque(maxlen=2)
    memory[track.track_id].append(midpoint)
    previous_midpoint = memory[track.track_id][0]
    tc = CheckCrossLine.LineCrossing(midpoint, previous_midpoint, line[0], line[1])
    if tc and (track.track_id not in ac):
        ac.append(track.track_id)
        crossline[track.track_id] = []
    if track.track_id in ac:
        crossline[track.track_id].append(midpoint)
        if len(crossline[track.track_id]) > 5:
            crossline[track.track_id].pop(0)
        if len(crossline[track.track_id]) >= 2:
            l1 = crossline[track.track_id][-2]
            l2 = crossline[track.track_id][-1]
            if ppm != 0:
                e = estimatespeed(l1, l2, ppm, fps, sf)
                re = round(e, 3)
                cv2.putText(frame, "{}".format(str(re)), (int(track_box[0]), int(track_box[1])), 0,
                            1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                if re < 1:
                    if track.track_id not in stopcid:
                        stopcid.append(track.track_id)
                    stopcspeed.append(re)
                    if len(stopcid) > 3:
                        for s in stopcspeed:
                            speed_adv.append(s)
                speed_adv.append(re)
    dt_s = datetime.datetime.now()
    return [memory, ac, crossline, speed_adv, stopcid, stopcspeed, dt_s]
  
def changelen(frame, track, memory, line, ac, linecross_counter):
    track_box = track.to_tlbr()  # (min x, miny, max x, max y)
    midpoint = track.tlbr_midpoint(track_box)
    if track.track_id not in memory:
        memory[track.track_id] = deque(maxlen=2)
    memory[track.track_id].append(midpoint)
    cv2.circle(frame, midpoint, 1, (0, 255, 255), 1)
    previous_midpoint = memory[track.track_id][0]
    tc = False
    n = 0
    for l in line:
        if not tc:
            tc, ncross = CheckCrossLine.LinesCrossing(midpoint, previous_midpoint, l[0], l[1], n)
            if not tc:
                n = n+1
    if tc and (track.track_id not in ac):
        linecross_counter[ncross] = linecross_counter[ncross] + 1
        ac.append(track.track_id)
    dt_s = datetime.datetime.now()
    return [memory, ac, linecross_counter, dt_s]
  
      
class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        # preprocessing: resize
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        a = [bboxes, scores, cls, self.cls_names]
        # vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return img, a
      
def imageflow_demo(predictor, vis_folder, current_time, args):
    if args.demo == "video":
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)  # video
    if args.demo == "webcam":
        cap = cv2.VideoCapture(args.path)  # url real-time
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mkv")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (int(width), int(height))
    )
    dt_sc = datetime.datetime.now()
    dt_ss = datetime.datetime.now()
    dt_scl = datetime.datetime.now()
    dt_s = datetime.datetime.now()
    dt_n = datetime.datetime.now()
    mmglobal.frame_count = 0;

    # Definition of the parameters
    max_cosine_distance = 0.75
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # function
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    ret_val, frame = cap.read()
    framey = frame.shape[0]
    framex = frame.shape[1]
    linec = []
    lines = []
    linecl = []
    if args.type == "CL" or args.type == "ALL":
        nl = args.lineN - 1
        for ln in range(nl):
            Pol = args.lineCL.split(",")
            x1 = float(Pol[ln * 4])
            y1 = float(Pol[ln * 4 + 1])
            x2 = float(Pol[ln * 4 + 2])
            y2 = float(Pol[ln * 4 + 3])
            line = [(int(x1 * framex), int(y1 * framey)), (int(x2 * framex), int(y2 * framey))]
            linecl.append(line)
    if args.lineC:
        Pol = args.lineC.split(",")
        x1 = float(Pol[0])
        y1 = float(Pol[1])
        x2 = float(Pol[2])
        y2 = float(Pol[3])
        linec.append([(int(x1 * framex), int(y1 * framey)), (int(x2 * framex), int(y2 * framey))])
    if args.lineS:
        Pol = args.lineS.split(",")
        x1 = float(Pol[0])
        y1 = float(Pol[1])
        x2 = float(Pol[2])
        y2 = float(Pol[3])
        lines.append([(int(x1 * framex), int(y1 * framey)), (int(x2 * framex), int(y2 * framey))])

    class_counter = [0, 0, 0, 0, 0]
    already_counted_c = [0]  # temporary memory for storing counted IDs
    already_counted_s = [0]
    already_counted_cl = [0]
    memory_c = {}
    memory_s = {}
    memory_cl = {}
    class_counter_5min = []
    speed_adv5min = []
    speed_adv = []
    crossline = {}
    stopcspeed = []
    stopcid = []
    linecross_counter = []
    cc1 = []

    alert = [] # 0-2 ปกติ 3 เกิดความผิดปกติและเจ้งเตือน

    if args.type == "CL" or args.type == "ALL":
        for lcn in range(args.lineN - 1):
            linecross_counter.append(0)
            alert.append(0)

    while True:
        #print("Frame ", mmglobal.frame_count)
        if (mmglobal.frame_count != 0):
            ret_val, frame = cap.read()
        if ret_val:
            if mmglobal.frame_count % args.skipframe == 0 and mmglobal.frame_count != 0:
                cv2.putText(frame, "Frame {}".format(str(mmglobal.frame_count)),
                            (int(0.05 * frame.shape[1]), int(0.1 * frame.shape[0])),
                            0, 1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                #cv2.putText(frame, "{}".format(str(linecross_counter)),
                #            (int(0.05 * frame.shape[1]), int(0.2 * frame.shape[0])), 0,
                #            1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                # Process every n frames
                #line = linec[0]
                #cv2.line(frame, line[0], line[1], (255, 255, 255), 2)
                #for line in linecl:
                #    cv2.line(frame, line[0], line[1], (255, 255, 255), 2)
                outputs, img_info = predictor.inference(frame)
                if outputs == [None]:
                    args.save_result = False
                else:
                    args.save_result = True
                    result_frame, a = predictor.visual(outputs[0], img_info, predictor.confthre)
                    boxesA = a[0]
                    boxwh = []
                    for bb in boxesA:
                        bx1 = float(bb[0])
                        by1 = float(bb[1])
                        w = float(bb[2]) - bx1
                        h = float(bb[3]) - by1
                        boxwh.append([bx1, by1, w, h])
                    boxes = torch.Tensor(boxwh)
                    confidence = a[1]
                    classes = a[2]
                    class_n = a[3]
                    features = encoder(frame, boxes)
                    # represents a bounding box detection in a single image
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

                    for track in tracker.tracks:
                        #bbox = track.to_tlbr()
                        #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        #cv2.putText(frame, "ID {}".format(str(track.track_id)), (int(bbox[0]), int(bbox[1])), 0,
                        #            1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                        if args.type == "C" or args.type == "ALL":
                            ac = counting(frame, track, memory_c, linec[0], already_counted_c, class_n, class_counter)
                            memory_c = ac[0]
                            already_counted_c = ac[1]
                            class_counter = ac[2]
                            if args.type == "C":
                                dt_n = ac[3]
                        if args.type == "S" or args.type == "ALL":
                            track_cls = track.det_cls
                            cls_name = class_n[int(track_cls.item())]
                            if cls_name != "truck":
                                ass = speeddetect(frame, track, fps, memory_s, lines[0], already_counted_s, class_n,
                                                  crossline, speed_adv, stopcid, stopcspeed, args.skipframe)
                                memory_s = ass[0]
                                already_counted_s = ass[1]
                                crossline = ass[2]
                                speed_adv = ass[3]
                                stopcid = ass[4]
                                stopcspeed = ass[5]
                                if args.type == "S":
                                    dt_n = ass[6]
                        if args.type == "CL" or args.type == "ALL":
                            acl = changelen(frame, track, memory_cl, linecl, already_counted_cl, linecross_counter)
                            memory_cl = acl[0]
                            already_counted_cl = acl[1]
                            linecross_counter = acl[2]
                            dt_n = acl[3]
                    stopcid = []

                if args.type == "C" or args.type == "ALL":
                    if args.demo >= "webcam":
                        t = for5minute(dt_sc, dt_n)
                    if args.demo == "video":
                        if mmglobal.frame_count % (300 * fps) == 0:
                            t = True
                        else:
                            t = False
                    if t:
                        cc1.append(class_counter)
                        dt_sc = dt_n
                        cc = class_counter[-1]
                        class_counter_5min.append([cc])
                        print("Counter 5 min", class_counter_5min)
                        print("All class in 5 min", cc1)
                        class_counter = [0, 0, 0, 0, 0]
                    # Delete memory of old tracks.
                    if len(memory_c) > 50:
                        del memory_c[list(memory_c)[0]]
                if args.type == "S" or args.type == "ALL":
                    if args.demo >= "webcam":
                        t = for5minute(dt_ss, dt_n)
                    if args.demo == "video":
                        if mmglobal.frame_count % (300 * fps) == 0 and mmglobal.frame_count != 0:
                            t = True
                        else:
                            t = False
                    if t:
                        dt_ss = dt_n
                        if not already_counted_s:  # if already_counted empty
                            speed_adv5min.append([30, 1])
                            print("No car in Road")
                            print(speed_adv5min)
                        else:
                            st = 1
                            ss = 0
                            sadvside = len(speed_adv)
                            for s in speed_adv:
                                if type(s) == float:
                                    ss = ss + s
                                else:
                                    sadvside - 1
                            ss = ss / len(speed_adv)
                            if ss < 15:
                                tt = 0
                                if speed_adv5min:
                                    for t in speed_adv5min:
                                        if t[0] != None and t[0] < 15:
                                            tt = tt + 1
                                    if tt >= 5:
                                        st = 4
                                        print("สภาพจราจรติดขัดมาก")
                                    else:
                                        st = 3
                                        print("สภาพจราจรติดขัด")
                            elif ss <= 25:
                                st = 2
                                print("สภาพจราจรหนาแน่น")
                            else:
                                st = 1
                                print("สภาพจราจรคล่องตัว")
                            ss = round(ss, 2)
                            speed_adv5min.append([ss, st])  # Speed,SpeedStat,Datetime
                            print("Speed in 5 min", speed_adv5min)
                            speed_adv = []
                            already_counted_s = []
                            crossline = {}
                            stopcid = []
                            stopcspeed = []
                    # Delete memory of old tracks.
                    if len(memory_s) > 50:
                        del memory_s[list(memory_s)[0]]
                if args.type == "CL" or args.type == "ALL":
                    if args.demo >= "webcam":
                        t = for1minute(dt_scl, dt_n)
                    if args.demo == "video":
                        if mmglobal.frame_count % (60 * fps) == 0 and mmglobal.frame_count != 0:
                            t = True
                        else:
                            t = False
                    if t:
                        dt_scl = dt_n
                        nlf = 0
                        nlfA = [False, -1]
                        for lf in linecross_counter:
                            if lf > 5:
                                if alert[nlf] < 3:
                                    alert[nlf] = alert[nlf]+1
                                    print("Warning in line ", nlf)
                                if alert[nlf] == 3:
                                    print("Alert! เกิดความผิดปกติ")
                                    nlfA[0] = True
                                    nlfA[1] = nlf
                            else:
                                alert[nlf] = 0
                            linecross_counter[nlf] = 0
                            nlf = nlf+1
                        print("Alert line",alert)
                if args.save_result:
                    vid_writer.write(result_frame)
                    vid_writer.write(frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
                if args.demo >= "webcam":
                    if for10minute(dt_s, dt_n):
                        print("Final result")
                        print("class_counter_5min", class_counter_5min)
                        print("speed_adv5min ", speed_adv5min)
                        print("All class in first 5 min", cc1)
                        print("Last Frame", mmglobal.frame_count)
                        break
                if args.demo == "video":
                    if mmglobal.frame_count >= 600 * fps:
                        print("Fps", fps)
                        print("Last Frame", mmglobal.frame_count)
                        break
                mmglobal.frame_count += 1
            else:
                mmglobal.frame_count += 1
        else:
            print("Final result")
            print("class_counter_5min",  class_counter_5min)
            print("speed_adv5min ", speed_adv5min)
            break
    
def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device)
    current_time = time.localtime()
    if args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device)
    current_time = time.localtime()
    if args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)
    
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
    
