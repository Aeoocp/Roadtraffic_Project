def spilttxt(input_T):
  file1 = open(input_T, "r")
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
      if(SEachBox[1] == 'car' or SEachBox[1] == 'truck' or SEachBox[1] == 'motorcycle'):
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
