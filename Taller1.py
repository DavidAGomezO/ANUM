import math
import MetodosYFunciones as MF
import pandas as pd
PI = math.pi  
E = math.e
COS = lambda x: math.cos(x)
SIN = lambda x: math.sin(x)
TAN = lambda x: math.tan(x)
SEC = lambda x: 1/math.cos(x)
CSC = lambda x: 1/math.sin(x)

# Taller, Punto 6
l = 359.41
D = 87.28
h = 130
Err = 10**(-5)
aMax = 21*PI/180
aMin = 0
tests = 10
aTests = aMax/tests

def f_b1(a):
  def aux(b1):
    return COS(a)*(l*SIN(a) - h - 0.5*D)*SIN(b1) + SIN(a)*(l*SIN(a) - h - 0.5*D)*COS(b1) + 0.5*D*COS(a)*TAN(b1) + 0.5*D*SIN(a)
  return aux

def df_b1(a):
  def aux(b1):
    return COS(a)*(l*SIN(a)-h-0.5*D)*COS(b1) + SIN(a)*(h + 0.5*D-l*SIN(a))*SIN(b1) + 0.5*D*(SEC(b1))**2
  return aux

FUNCS = [f_b1(aTests*i) for i in range(0,tests+1)]
DFUNCS = [df_b1(aTests*i) for i in range(0,tests+1)]

# for i in range(0,tests+1):
#   print(MF.NewtonRaphson(FUNCS[i],DFUNCS[i],1.2,E=Err,Iter=False)*180/PI)

# Taller punto 6:

h = 80  
l = 359.41
D = 87.28
Err = 10**(-5)
b1 = 15*PI/180

A = l*SIN(b1)
B = l*COS(b1)
C = (h + 0.5*D)*SIN(b1) - 0.5*D*TAN(b1)
E = (h + 0.5*D)*COS(b1) - 0.5*D

func = lambda a: A*SIN(a)*COS(a) + B*(SIN(a))**2 - C*COS(a) - E*SIN(a)
dfunc = lambda a: A*((COS(a))**2 - (SIN(a))**2) + 2*B*SIN(a)*COS(a) + C*COS(a) + E*SIN(a)
ddfunc = lambda a: -4*A*COS(a)*SIN(a) + 2*B*((COS(a))**2 - (SIN(a))**2) + C*COS(a) + E*SIN(a)

# print(MF.ModNewtonRaphson(func,dfunc,ddfunc,0.2,E=Err,Iter=False)*180/PI)
