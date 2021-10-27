def createLine():
  numLine = int(input("Enter number of counting line :"))
  x = []
  y = []
  s_line = []
  for l in range(numLine):
    print("Enter the beginning and ending of the line (0-1) (x,y) :")
    print("Line",l+1)
    Bp = input("Beginning Point :").split(",")
    x.append((Bp[0]))
    y.append((Bp[1]))
    Ep = input("Ending Point :").split(",")
    x.append(Ep[0])
    y.append(Ep[1])
  return l,x,y
