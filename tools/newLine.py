def createLine():
  x = []
  y = []
  print("Enter the beginning and ending of the line (0-1) (xb,yb,xe,ye) ")
  Bp = input("Enter line position :").split(",")
  x.append((Bp[0]))
  y.append((Bp[1]))
  x.append((Bp[2]))
  y.append((Bp[3]))
  return x,y

def createLines():
  n_line = int(input("Please enter the number of road lenses :"))
  x = []
  y = []
  print("Enter the beginning and ending of the line (0-1) (xb,yb,xe,ye) ")
  for n in range(n_line-1):
    print("Line ",n)
    Bp = input("Enter line position :").split(",")
    x.append((Bp[0]))
    y.append((Bp[1]))
    x.append((Bp[2]))
    y.append((Bp[3]))
  return n_line,x,y
