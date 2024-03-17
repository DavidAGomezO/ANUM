import MetodosYFunciones as MF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Punto 17.3, hacer regresión lineal con los siguientes datos:

def Ej17_3():
  X = [0,2,4,6,9,11,12,15,17,19]
  Y = [5,6,7,6,9,8,7,10,12,12]
  Values1 = list(MF.LinealReg(X,Y))
  Labels = ["a_0","a_1","SumErrSq","Error","CorrCoef"]
  df = pd.DataFrame(Values1,index=Labels)
  print(df)
  xs = np.linspace(0,20,100)
  ys = Values1[0] + Values1[1]*xs
  plt.plot(xs,ys)
  plt.scatter(X,Y,marker="x",s=10)
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.show()

  Values2 = list(MF.LinealReg(Y,X))
  df = pd.DataFrame(Values2,index=Labels)
  print(df)
  xs = np.linspace(0,20,100)
  ys = Values2[0] + Values2[1]*xs
  plt.plot(xs,ys)
  plt.plot(xs,ys)
  plt.scatter(Y,X,marker="x",s=10)
  plt.xlabel("Y")
  plt.ylabel("X")
  plt.show()

# Ejercicio 17.4, lo mismo que antes
def Ej17_4():
  X = [6,7,11,15,17,21,23,29,29,37,39]
  Y = [29,21,29,14,21,15,7,7,13,0,3]
  Values = list(MF.LinealReg(X,Y))
  Labels = ["a_0","a_1","SumErrSq","Error","CorrCoef"]
  df = pd.DataFrame(Values,index=Labels)
  print(df)
  xs = np.linspace(4,44,100)
  ys = Values[0] + Values[1]*xs
  plt.plot(xs,ys)
  plt.scatter(X,Y,marker="x",s=10)
  plt.xlabel("X")
  plt.ylabel("Y")
  plt.show()

# Ejercicio 7.15 hallar la mejor aproximación lineal que pasa por el origen

def Ej17_5():
  X = [2,4,6,7,10,11,14,17,20]
  Y = [1,2,5,2,8,7,6,9,12]
  a_1 = sum(Y)/sum(X)
  xs = np.linspace(1,21,100)
  ys = a_1*xs
  plt.plot(xs,ys)
  plt.scatter(X,Y,marker='x',s=10)
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.show()

Ej17_5()
