def spilttxt(input_T):
  file1 = open(input_T, "r")
  FileContent = file1.read()
  AS = FileContent.split()
  inform = []
  frame = []
  
  c = -1
  y = 0
  for x in range(len(AS)-1):
    xx = AS[x].split(',')
    if y == 0:
      if AS[x] == "#":
        c = c+1
        fn = AS[x+2].split(',')
        frame.append(int(fn[0]))
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
  last_frame = frame[len(frame)-1]
  print(last_frame)
  xf = 0
  for f in range(len(frame)-1):
    subclass = [] 
    subconfidence = []
    subbox = []
    while (xf != int(frame[f])):
      classes_s.append(subclass)
      confidence_s.append(subconfidence)
      boxes_s.append(subbox)
      xf += 1
    xf += 1
    for b in range(len(inform[f])):
      eachBox = inform[f][b]
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
    
  return classes_s,confidence_s,boxes_s
