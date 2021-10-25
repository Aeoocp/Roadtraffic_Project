
from collections import Counter

def createLine(frame,mulX1,mulX2,mulY1,mulY2)
  frameY = frame.shape[0] #360
  frameX = frame.shape[1] #640
  line = [(int(mulX1 * frameX), int(mulY1 * frameY)), (int(mulX2 * frameX), int(mulY2 * frameY))]
  total_counter = 0
  class_counter = Counter()
  intersect_info = []
  memory = {}
  
