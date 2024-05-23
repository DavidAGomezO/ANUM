import pandas as pd
import MetodosYFunciones as MF
import math


mat = [[0 for _ in range(10)] for __ in range(9)]

for i in range(9):
  mat[i][i] = -2.15
  if i < 8:
    mat[i][i+1] = 1
  if i > 0:
    mat[i][i-1] = 1

mat[0][9] = -240
mat[8][9] = -150

print(MF.GaussSeidel(mat,[0 for _ in range(9)], Diag=True, Err=0.3,Iter=False))
  