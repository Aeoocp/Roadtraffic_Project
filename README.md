# Introduction

This Project is part of the Bachelor of Science study program, Computer Science Department of Mathematics and Computer Science, Faculty of Science Chulalongkorn University, Academic year 2021, Copyright of Chulalongkorn University,
in name "Road traffic monitoring and warning system"

Paper File : https://drive.google.com/file/d/11wYbHBo3nXdI3FkjMStdxXGs9sFyXHdt/view?usp=sharing

About
   - Object Detection and Classification vehicle
   - Object Tracking
   - Vehicle counting
   - Speed measurement
   - Assess traffic conditions
   - Lane change detection

## Tools: 

   - Python
   - OpenCV
   - YoloX
   - Pycharm 
   
## YOLOX and DeepSORT

Object tracking implemented with YOLOX, DeepSort, and TensorFlow.

YOLOX is one version of YOLO (algorithm that uses deep convolutional neural networks to perform object detections), with a better performance. You can check out on [THIS LINK](https://github.com/Megvii-BaseDetection/YOLOX) to see more impormation about YOLOX.

DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) used for building highly accurate object trackers. Check out on [THIS LINK](https://arxiv.org/pdf/1703.07402.pdf) to see more impormation about DeepSORT.

# Demo of Object Tracker on Cars
<p align="center"><img src="img/Demo.gif"\></p>

# Getting Started
What do you need?

   1. Model file : You can dowload model from [THIS LINK](https://drive.google.com/file/d/1WCsGlk9X613VBFC8C55vYCkqVZ2m8VaD/view?usp=sharing) (.pth.tar)
   2. Camer url (.m3u8) or File video (.mkv)
   3. In Vehicle counting, Speed measurement and Lane change detection Step you need to set the position for tracking line : So I have created [GOOGLE COLAB](https://colab.research.google.com/drive/1dyjxNsnXV2cV3UYk7H1b3J-sx-v-pU5i?usp=sharing) for test specifying position of line.
   4. Other Parameter 
```bash
#Example and recommend
webcam -n yolox-s -c latest-300_ckpt.pth.tar --path https://camerai1.iticfoundation.org/pass/180.180.242.207:1935/Phase3/PER_3_004_IN.stream/chunklist_w304784440.m3u8 
--type C --lineC 0.65,0.325,0.95,0.45 --skipframe 3 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu

# explain Parameter 

```

# References
* https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking
* https://github.com/dungdo123/Code_Interview/tree/main/Vehicle_Speed_Estimation
