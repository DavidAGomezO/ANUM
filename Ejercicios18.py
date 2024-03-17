import MetodosYFunciones as MF
import math
import pandas as pd

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

Ej182()
