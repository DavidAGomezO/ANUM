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

# 5.6 Usando bisección, aproximar la raíz positiva de Ln(x^2) - 0.7,
# usando el intervalo [0.5,2] y 3 iteraciones
func = lambda x:math.log(x**2,E) - 0.7

# --- Ejercicio --- #
# print(MF.Bisec(func,0.5,2,3))
# print(MF.RegulaFalsi(func,0.5,2,3))

# 6.1 Usar punto fijo para aproximar la solución a sin(sqrt(x)) - x = 0
# tomando x=0.5 como punto inicial, y un error deseado menor a 0.01%

func1 = lambda x:SIN(math.sqrt(x))
func2 = lambda x:(math.asin(x))**2

# --- Ejercicio --- #
# print(MF.FixedPoint(func1,0.5,E=0.01,Iter=False))
# print(MF.FixedPoint(func2,0.5,E=0.01,Iter=False))

# 6.2 Determinar la raíz de mayor valor de f(x) = 2x^3 - 11.7x^2 + 17.7x - 5
func = lambda x: x*(x*(2*x - 11.7) + 17.7) - 5
dfunc = lambda x: x*(6*x - 23.4) + 17.7
g = lambda x:((x**2)*(-2*x + 11.7) + 5)/17.7
#    --- Ejercicio --- #
# nIter = 3
# Methods = ["Punto fijo", "Newton-Raphson", "Secante", "Secante Mod."]
# Iterations = [f"x_{i}" for i in range(0,nIter+1)]
# FpointVal = [MF.FixedPoint(g,3,i) for i in range(0,nIter+1)]
# NewRaphVal = [MF.NewtonRaphson(func,dfunc,3,i) for i in range(0,nIter+1)]
# SecVal = [MF.Secant(func,3,4,i) for i in range(0,nIter+1)]
# ModSecVal = [MF.ModSecant(func,3,0.01,i) for i in range(0,nIter+1)]
# TableVals = [FpointVal,NewRaphVal,SecVal,ModSecVal]
# TableErr = list(map(MF.AproxErrorList,TableVals))
# Values = pd.DataFrame(TableVals,columns=Iterations,index=Methods)
# Errors = pd.DataFrame(TableErr,columns=Iterations,index=Methods)
# print(Errors)
# print(Values)

# 6.7 Usando el metodo de Secante, hallar la primera raíz positiva de f(x) = sin(x) + cos(1 + x^2) - 1
# func = lambda x:SIN(x) + COS(1 + x**2) - 1
# # --- Ejercicio --- #
# nIter = 4
# X = [(1,3),(1.5,2.5),(1.5,2.25)]
# Rows = ['a', 'b', 'c']
# XIndex = [f"x_{i}" for i in range(0,nIter+1)]
# SecVals = [[MF.Secant(func,X[j][0],X[j][1],i) for i in range(0,nIter+1)] for j in range(0,len(X))]
# Table = pd.DataFrame(SecVals,columns=XIndex,index=Rows)
# print(Table)

# 6.10 Hallar la menor raíz positiva de f(x) = 7 sin(x) e^{-x} - 1
# func = lambda x: 7*SIN(x)*E**(-x) - 1
# dfunc = lambda x:7*E**(-x)*(COS(x) - SIN(x))
# NewRaphVal = [MF.NewtonRaphson(func,dfunc,0.3,i) for i in range(0,4)]
# SecVal = [MF.Secant(func,0.5,0.4,i) for i in range(0,4)]
# ModSecVal = [MF.ModSecant(func,0.3,0.01,i) for i in range(0,4)]
# Values = [NewRaphVal,SecVal,ModSecVal]
# Rows = ['b) Newton-Raphson','c) Secante','d) Secante Mod.']
# Cols = [f"x_{i}" for i in range(0,4)]
# Table = pd.DataFrame(Values,columns=Cols,index=Rows)
# print(Table)

# 6.11 Usar Newton-Raphson para hallar la raíz de f(x) = e^{-0.5x} (4-x) - 2
# func = lambda x: (E**(-0.5*x))*(4-x) - 2
# dfunc = lambda x: (E**(-0.5*x))*(0.5*x-3)
# X = [2,6,8]
# nIter = 4
# Er=10
# Vals = [[MF.NewtonRaphson(func,dfunc,X[j],E=Er/(2**i),Iter=False) for i in range(0,nIter+1)] for j in range(0,len(X))]
# Rows = ['a', 'b', 'c']
# Cols = [f"E <= {Er/(2**i)}" for i in range(0,nIter+1)]
# Table = pd.DataFrame(Vals,columns=Cols,index=Rows)

# print(Table)

# Taller, Punto 6
A = lambda l,b_1 : l*SIN(b_1)
dA_db_1 = lambda l,b_1 : l*COS(b_1)

B = lambda l,b_1 : l*COS(b_1)
dB_db_1 = lambda l,b_1 : -l*SIN(b_1)

C = lambda h,D,b_1 : (h + 0.5*D)*SIN(b_1) - 0.5*D*TAN(b_1)
dC_db_1 = lambda h,D,b_1 : (h + 0.5*D)*COS(b_1) - 0.5*D*(SEC(b_1))**2

E = lambda h,D,b_1 : (h + 0.5*D)*COS(b_1) - 0.5*D
dE_db_1 = lambda h,D,b_1 : -(h + 0.5*D)*SIN(b_1)

func = lambda a,l,b_1,h,D : A(l,b_1)*SIN(a)*COS(a) + B(l,b_1)*(SIN(a))**2 - C(h,D,b_1)*COS(a) - E(h,D,b_1)*SIN(a)
dfunc_db_1 = lambda a,l,b_1,h,D : l*COS(b_1)*SIN(a)*COS(a) - l*SIN(b_1)*(SIN(a))**2 + (h + 0.5*D)*COS(b_1)
h = 130/100
D = 87.28/100
l = 369.3/100
Err = 10**(-5)
a_max = 26*PI/180
a_min = 0
Delta = 10
a_test = [i*a_max/Delta for i in range(0,Delta+1)]
def F(a):
  def aux(b_1):
    return func(a,l,b_1,h,D)
  
  return aux

def dF(a):
  def aux(b_1):
    return dfunc_db_1(a,l,b_1,h,D)
  return aux

Funcs = [ F(a_test[i]) for i in range(0,Delta+1)]
dFuncs = [ dF(a_test[i]) for i in range(0,Delta+1)]

for i in range(0,Delta+1):
  print(MF.NewtonRaphson(Funcs[i],dFuncs[i],0,E=Err,Iter=False))

