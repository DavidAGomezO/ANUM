import MetodosYFunciones as MF
import math
import pandas as pd
import matplotlib.pyplot as plt

def dQdt(a,b):
  def aux(t,Q):
    return math.e**(-a*t)-b*Q
  return aux

M = [1 + (19/7)*i for i in range(1,6)]
A = [0.1 + (0.7/7)*i for i in range(1,6)]
B = [0.2 + (0.7/7)*i for i in range(1,6)] 

# Cambio en m con a,b fijos

DATAF_M_05 = [pd.DataFrame(MF.RungeKutta(0,10,M[i],dQdt(A[0],B[0]),0.5,4)) for i in range(5)]
DATAF_M_025 = [pd.DataFrame(MF.RungeKutta(0,10,M[i],dQdt(A[0],B[0]),0.25,4)) for i in range(5)]

# Cambio en a con m,b fijos

DATAF_A_05 = [pd.DataFrame(MF.RungeKutta(0,10,M[0],dQdt(A[i],B[0]),0.5,4))  for i in range(5)]
DATAF_A_025 = [pd.DataFrame(MF.RungeKutta(0,10,M[0],dQdt(A[i],B[0]),0.25,4))  for i in range(5)]

# Cambio en b con m,a fijos

DATAF_B_05 = [pd.DataFrame(MF.RungeKutta(0,10,M[0],dQdt(A[0],B[i]),0.5,4)) for i in range(5)]
DATAF_B_025 = [pd.DataFrame(MF.RungeKutta(0,10,M[0],dQdt(A[0],B[i]),0.25,4)) for i in range(5)]

Data_temp = pd.DataFrame(
  {
    "m": M,
    "a": A,
    "b": B
  }
)

DATA_025 = [DATAF_M_025,DATAF_A_025,DATAF_B_025]

for DATA in DATA_025:
  i = 0
  for df in DATA:
    x = list(df['x'])
    y = list(df['y'])
    plt.plot(x,y,marker='o',markersize=5,label=f'{i}')
    i += 1
  plt.show()

