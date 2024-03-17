import MetodosYFunciones as MF
import math
import pandas as pd
PI = math.pi  
E = math.e
COS = lambda x: math.cos(x)
SIN = lambda x: math.sin(x)
TAN = lambda x: math.tan(x)
SEC = lambda x: 1/math.cos(x)
CSC = lambda x: 1/math.sin(x)

A = [
  [0,10,2,1],
  [3,4,5,2],
  [10,9,8,3]
]

I  = [i for i in range(0,len(A))]
print("antes")
for i in range(0,len(A)):
  print(A[I[i]])
print("despues")
MF.Elim(I,A,1,2,1)
for i in range(0,len(A)):
  print(A[I[i]])
