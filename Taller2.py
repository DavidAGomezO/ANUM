import MetodosYFunciones as MF
import math
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

PI = math.pi  
E = math.e
COS = lambda x: math.cos(x)
SIN = lambda x: math.sin(x)
TAN = lambda x: math.tan(x)
SEC = lambda x: 1/math.cos(x)
CSC = lambda x: 1/math.sin(x)

V = [20,25,32,40,52,71,75,79,80,82,85,87,89,91,94,97,100,104,109,112]
P = [2.041,1.974,1.903,1.84,1.769,1.688,1.674,1.662,1.659,1.654,1.644,1.638,1.633,1.625,1.618,1.612,1.605,1.596,1.583,1.578]

print(MF.LinealReg(V,P)[:2])

# Para una regresión hiperbólica V = a/(V^b), se toma logaritmo y queda Ln(V) = Ln(a) - b Ln(V)

V_h = [math.log(v) for v in V]
P_h = [-math.log(p) for p in P]

Coefs = MF.LinealReg(V_h,P_h)[:2]
a = E**(Coefs[0])
print(a,Coefs[1])

# Regresión de grado 2

print(MF.nDegReg(V,P,2)[0])

# Regresión múltiple

V = [10,15,22,28,40,53,60,65,80,82,85,89,90,92,94,99,100,102,105,108]

T = [45,50,56,59,65,70,73,79,82,83,85,89,90,92,94,99,100,102,105,108]

X = [V,T]

P = [55.58,46.58,40.17,35.63,30.68,27.17,26.11,27.33,24.4,24.33,24.438,25.598,25.528,25.418,25.328,26.788,26.318,26.248,26.738,26.848]

print(MF.LinMulReg(X,P))

# Para una regresión de la forma P = (b T^a)/(V^k), se toma logarigmo y queda Ln(P) = Ln(b) + a Ln(T) - k Ln(V)

V_h = [-math.log(v) for v in V]
T_h = [math.log(t) for t in T]
X_h = [V_h,T_h]
P_h = [math.log(p) for p in P]

fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection="3d")

ax.scatter3D(V,T,P, color = "green")
plt.title("Gráfica 4to punto")

plt.show()
A = MF.LinMulReg(X_h,P_h)
b = E**A[0]
k = A[1]
a = A[2]

func4 = lambda t,v : b*(t**a)/(v**k)

print(f"[{b}, {a}, {k}]")

print(func4(81,72))
