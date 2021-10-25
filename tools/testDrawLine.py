from tools import newLine
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('video', '/content/Traffic_counting/video/hlp/080841-03.mkv', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', '/content/Traffic_counting/output.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')

def testDrawLine(mulX1,mulX2,mulY1,mulY2)
  file_path = FLAGS.video
  video_capture = cv2.VideoCapture(file_path)
  w = int(video_capture.get(3))
  h = int(video_capture.get(4))
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('output_2line.avi', fourcc, 30, (w, h))
  ret, frame = video_capture.read()
  frameY = frame.shape[0] #360
  frameX = frame.shape[1] #640
  
  newLine.createLine(frame,mulX1,mulX2,mulY1,mulY2)
