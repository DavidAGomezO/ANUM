import MetodosYFunciones as MF
import math
import pandas as pd
import numpy as np
import math

PI = math.pi  
E = math.e
COS = lambda x: math.cos(x)
SIN = lambda x: math.sin(x)
TAN = lambda x: math.tan(x)
SEC = lambda x: 1/math.cos(x)
CSC = lambda x: 1/math.sin(x)
# Ejercicio 18.1, usar interpolación lineal para aproximar logaritmo

def Ej181():
  funca = [(8,0.9030900), (12,1.0791812)]
  funcb = [(9,0.9542425), (11,1.0413927)]
  n = 1
  print(MF.NewtonIntCoef(funca,n))
  print(MF.NewtonIntCoef(funcb,n))
  RVals = [1,1]
  ApVals = [MF.NewtonPolOpt(funca,1)(10), MF.NewtonPolOpt(funcb,1)(10)]
  Errors = [MF.TruePercRelError(1,ApVals[i]) for i in range(0,2)]
  Info = {"Real":RVals, "Aprox":ApVals, "Errors":Errors}
  df = pd.DataFrame(Info,index=['a', 'b'])
  print(df)

# Ejercicio 18.2, usar interpolación polinómica (grado 2) para aproximar logaritmo
def Ej182():
  func = [(8,0.9030900),(9,0.9542425), (11,1.0413927)]
  n = 2
  print(MF.NewtonIntCoef(func,n))
  AproxVal = MF.NewtonPolOpt(func,n)(10)
  Info = {
    "Real": [1],
    "Aprox": [AproxVal],
    "Error": [MF.TruePercRelError(1,AproxVal)]
  }
  df = pd.DataFrame(Info)
  print(df)

def Ej211():
  func = lambda x : 6 + 3*COS(x)
  INTVAL = 3*PI + 3

  DataB = [MF.TrapzRuleFunc(func,1,0,PI/2)]
  ErrB = [MF.TruePercRelError(INTVAL,DataB[0])]

  DataC = [MF.TrapzRuleFunc(func,2,0,PI/2), MF.TrapzRuleFunc(func,4,0,PI/2)]
  ErrC = [MF.TruePercRelError(INTVAL,i) for i in DataC]

  DataD = [MF.SimpsonsRuleFunc(func,2,0,PI/2)]
  ErrD = [MF.TruePercRelError(INTVAL,DataD[0])]

  DataE = [MF.SimpsonsRuleFunc(func,4,0,PI/2)]
  ErrE = [MF.TruePercRelError(INTVAL,DataE[0])]
  
  DataF = [MF.SimpsonsRuleFunc(func,3,0,PI/2,False)]
  ErrF = [MF.TruePercRelError(INTVAL,DataF[0])]

  DataG = [MF.SimpsonsRuleFunc(func,2,0,PI/5) + MF.SimpsonsRuleFunc(func,3,PI/5,PI/2,False)]
  ErrG = [MF.TruePercRelError(INTVAL,DataG[0])]

  TotalData = {
    "Real Value" : [INTVAL],
    "b Aprox" : DataB,
    "b Error" : ErrB,
    "c Aprox" : DataC,
    "c Error" : ErrC,
    "d Aprox" : DataD,
    "d Error" : ErrD,
    "e Aprox" : DataE,
    "e Error" : ErrE,
    "f Aprox" : DataF,
    "f Error" : ErrF,
    "g Aprox" : DataG,
    "g Error" : ErrG,
  }
  df = pd.DataFrame({
    key:pd.Series(value) for key,value in TotalData.items()
  })
  print(df)

def Ej212():
  func = lambda x : 1 - E**(-2*x)
  INTVAL = 3 + (E**(-6) - 1)/2

  DataB = [MF.TrapzRuleFunc(func,1,0,3)]
  ErrB = [MF.TruePercRelError(INTVAL,DataB[0])]

  DataC = [MF.TrapzRuleFunc(func,2,0,3), MF.TrapzRuleFunc(func,4,0,3)]
  ErrC = [MF.TruePercRelError(INTVAL,i) for i in DataC]

  DataD = [MF.SimpsonsRuleFunc(func,2,0,3)]
  ErrD = [MF.TruePercRelError(INTVAL,DataD[0])]

  DataE = [MF.SimpsonsRuleFunc(func,4,0,3)]
  ErrE = [MF.TruePercRelError(INTVAL,DataE[0])]
  
  DataF = [MF.SimpsonsRuleFunc(func,3,0,3,False)]
  ErrF = [MF.TruePercRelError(INTVAL,DataF[0])]

  DataG = [MF.SimpsonsRuleFunc(func,2,0,6/5) + MF.SimpsonsRuleFunc(func,3,6/5,3,False)]
  ErrG = [MF.TruePercRelError(INTVAL,DataG[0])]

  TotalData = {
    "Real Value" : [INTVAL],
    "b Aprox" : DataB,
    "b Error" : ErrB,
    "c Aprox" : DataC,
    "c Error" : ErrC,
    "d Aprox" : DataD,
    "d Error" : ErrD,
    "e Aprox" : DataE,
    "e Error" : ErrE,
    "f Aprox" : DataF,
    "f Error" : ErrF,
    "g Aprox" : DataG,
    "g Error" : ErrG,
  }
  df = pd.DataFrame({
    key:pd.Series(value) for key,value in TotalData.items()
  })
  print(df)

def Ej215():
  INTVAL = 2056
  func = lambda x : (4*x - 3)**3

  A = MF.SimpsonsRuleFunc(func,4,-3,5)
  B = MF.SimpsonsRuleFunc(func,2,-3,0.2) + MF.SimpsonsRuleFunc(func,3,0.2,5,False)

  print(A)
  print(B)

print(0**0)