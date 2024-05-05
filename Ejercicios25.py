import MetodosYFunciones as MF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def ej_251():
  f = lambda x : np.e**((x**3)/3 - 1.1*x)

  F = lambda t,y : y*(t**2 - 1.1)

  Xeu5 = MF.EulerDifEq(0,2,1,F,0.5)['x']
  Yeu5 = MF.EulerDifEq(0,2,1,F,0.5)['y']

  Xeu25 = MF.EulerDifEq(0,2,1,F,0.25)['x']
  Yeu25 = MF.EulerDifEq(0,2,1,F,0.25)['y']

  Xheuns = MF.HeunsDifEq(0,2,1,F,0.5)['x']
  Yheuns = MF.HeunsDifEq(0,2,1,F,0.5)['y']

  XRK = MF.RungeKutta(0,2,1,F,0.5,4)['x']
  YRK = MF.RungeKutta(0,2,1,F,0.5,4)['y']


  x = np.arange(-1,3,0.1)
  y = f(x)
  plt.plot(Xeu5,Yeu5,marker='o',color="green")
  plt.plot(Xeu25,Yeu25,marker='o')
  plt.plot(Xheuns,Yheuns,marker='s')
  plt.plot(XRK,YRK, marker='^')
  plt.plot(x,y)
  plt.show()

def ej_252():
  f = lambda x : (x/2 + x**2 + 1)**2

  F = lambda t,y : (1 + 4*t)*math.sqrt(y)

  Xeu = MF.EulerDifEq(0,1,1,F,0.25)['x']
  Yeu = MF.EulerDifEq(0,1,1,F,0.25)['y']

  Xhe = MF.HeunsDifEq(0,1,1,F,0.25)['x']
  Yhe = MF.HeunsDifEq(0,1,1,F,0.25)['y']

  Xrk2 = MF.RungeKutta(0,1,1,F,0.25,2)['x']
  Yrk2 = MF.RungeKutta(0,1,1,F,0.25,2)['y']

  Xrk4 = MF.RungeKutta(0,1,1,F,0.25,4)['x']
  Yrk4 = MF.RungeKutta(0,1,1,F,0.25,4)['y']


  x = np.arange(-0.05,1.05,0.1)
  y = f(x)
  plt.plot(x,y,color="black")
  plt.plot(Xeu,Yeu,marker='o')
  plt.plot(Xhe,Yhe,marker='s')
  plt.plot(Xrk2,Yrk2, marker='^')
  plt.plot(Xrk4,Yrk4, marker='v')
  plt.show()

def ej_254():
  func1 = lambda x,y1,y2 : -(0.6*y1 + 8*y2)
  func2 = lambda x,y1,y2 : y1

  Data = MF.RungeKuttaSis(0,5,[0,4],0.1,4,func1,func2)

  df = pd.DataFrame(Data)
  print(df)

  plt.plot(Data['x'],Data["y_1"],marker='o')
  plt.show()

def ej_256():
  func = lambda x,y : -2*y + x**2

  Data = MF.RungeKutta(0,3,1,func,0.5,3)

  fReal = lambda x: (2*x**2 - 2*x + 1 + 3*np.e**(-x/2))/4
  sol = [fReal(t) for t in Data['x']]
  Data["Solución"] = sol
  df = pd.DataFrame(Data)
  print(df)

ej_256()
