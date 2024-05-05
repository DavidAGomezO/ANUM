import math
import numpy as np
import pandas as pd
# Errores, Métodos y Funciones del curso

# Funciones aux.

def BinSearch(X,k):
  i = (len(X) - 1)//2

  while k != X[i]:
    if k > X[i]:
      i = (i + len(X) - 1)//2
    else:
      i = i/2
  return i


# Errores
def TrueError(TrueValue:float, AproxValue:float):
  return abs(AproxValue-TrueValue)

def TruePercRelError(TrueValue:float,AproxValue:float):
  return abs((1 - AproxValue/TrueValue)*100)

def AproxPercRelError(PresAprox:float,PrevAprox:float):
  return abs((1 - PrevAprox/PresAprox)*100)

def AproxErrorList(Vals:list):
  Errors = ["-"]
  for i in range(1,len(Vals)):
    Errors += [f"{AproxPercRelError(Vals[i],Vals[i-1])} %"]
  return Errors

# Aproximación de Raices - Métodos Cerrados
def Bisec(f:float,x_l:float,x_u:float,n:int=1,E:float=100,Iter:bool=True):
  """Método de Bisección

  Parameters
  ----------
  f : Función
      Es la función de la cual se quiere saber la raíz.
  x_l : float
      Valor inferior del intervalo.
  x_u : float
      Valor superior del intervalo.
  n : int
      Número de iteraciones deseadas.
  E : float
    Error aproximado deseado.
  Iter : bool
    Determina si el proceso debe hacerse con base en
    un número de iteraciones, o con base en el error.

  Returns
  -------
  float | None
      Aproximación obtenida.
  """
  if f(x_l) == 0:
    return x_l
  elif f(x_u) == 0:
    return x_u
  elif f(x_u)*f(x_l) > 0:
    return None

  if Iter: 
    for i in range(1,n+1):
      x_r = (x_u + x_l)/2
      if f(x_r) == 0:
        return x_r
      elif f(x_r)*f(x_u) < 0:
        x_l = x_r
      else:
        x_u = x_r
    return x_r
  
  x_r1 = x_l
  x_r = (x_u + x_r1)/2

  while AproxPercRelError(x_r,x_r1) > E:
    if f(x_r) == 0:
      return x_r
    elif f(x_r)*f(x_u) < 0:
      x_l = x_r
    else:
      x_u = x_r
    x_r1 = x_r
    x_r = (x_u + x_l)/2

  return x_r

def RegulaFalsi(f:float,x_l:float,x_u:float,n:int=1,E:float=100,Iter:bool=True):
  """Método de Regula Falsi

  Parameters
  ----------
  f : Función
      Función de la que se quiere aproximar la raíz.
  x_l : float
      Valor inferior del intervalo.
  x_u : float
      Valor superior del intervalo.
  n : int
      Número de iteraciones deseadas.
  E : float
    Error aproximado deseado.
  Iter : bool
    Determina si el proceso debe hacerse con base en
    un número de iteraciones, o con base en el error.

  Returns
  -------
  Float | None
      Aproximación obtenida
  """
  if f(x_l) == 0:
    return x_l
  elif f(x_u) == 0:
    return x_u
  elif f(x_u)*f(x_l) > 0:
    return None
  
  if Iter:
    for i in range(1,n+1):
      x_r = x_u - (f(x_u)*(x_l - x_u))/(f(x_l) - f(x_u))
      if f(x_r) == 0:
        return x_r
      elif f(x_r)*f(x_u) < 0:
        x_l = x_r
      else:
        x_u = x_r
    return x_r
  
  x_r1 = x_l
  x_r = x_u - (f(x_u)*(x_l - x_u))/(f(x_l) - f(x_u))
  while AproxPercRelError(x_r,x_r1) > E:
    if f(x_r) == 0:
      return x_r
    elif f(x_r)*f(x_u) < 0:
      x_l = x_r
    else:
      x_u = x_r
    x_r1 = x_r
    x_r = x_u - (f(x_u)*(x_l - x_u))/(f(x_l) - f(x_u))
  
  return x_r

def FixedPoint(g:float,x:float,n:int=1,E:float=100,Iter:bool=True):
  """Método de punto fijo

  Parameters
  ----------
  g : Función
      Función la cual se quiere igualar a x
  x : float
      Valor inicial para el algoritmo
  n : int, opcional
      Número de iteraciones, por defecto 1
  E : float, opcional
      Error deseado, por defecto 100
  Iter : bool, opcional
      Define si la aproximación se hará con un número de iteraciones o
      hasta alcanzar un error deseado, por defecto True

  Returns
  -------
  Float
      Aproximación obtenida
  """
  if g(x) == x:
    return x
  
  if Iter:
    for i in range(1,n+1):
      x = g(x)
      if g(x) == x:
        return x
    return x
  
  x_1 = x
  x = g(x)
  while AproxPercRelError(x,x_1) > E:
    if g(x) == x:
      return x
    x_1 = x
    x = g(x)
    if x == 0:
      break
  return x

# Aproximación de raices - Métodos abiertos

def NewtonRaphson(f:float,df:float,x:float,n:int=1,E:float=100,Iter:bool=True):
  """Método de aproximación de raíz Newton Raphson

  Parameters
  ----------
  f : float
      Función de la que se quiere aproximar la raíz
  df : float
      Derivada de la función f
  x : float
      Punto inicial para el algoritmo
  n : int, opcional
      Número de iteraciones deseadas, por defecto 1
  E : float, opcional
      Error deseado, por defecto 100
  Iter : bool, opcional
      Determina si el proceso se debe realizar con iteraciones o hasta
      obtener un error deseado, por defecto True

    Returns
  -------
    float
        Aproximación obtenida
  """
  if f(x) == 0 or df(x) == 0:
    return x
  if Iter:
    for i in range(1,n+1):
      if f(x) == 0 or df(x) == 0:
        return x
      x = x - f(x)/df(x)
    return x
  
  x_a = x
  x = x - f(x)/df(x)
  while AproxPercRelError(x,x_a) > E:
    if f(x) == 0 or df(x) == 0:
        return x
    x_a = x
    x = x - f(x)/df(x)
  return x

def Secant(f:float,x_0:float,x_1:float,n:int=1,E:float=100,Iter:bool=True):
  """Método de secante

  Parameters
  ----------
  f : float
      Función de la que se quiere aproximar la raíz
  x_0 : float
      Primer punto inicial
  x_1 : float
      Segundo punto inicial
  n : int, opcional
      Número de iteraciones deseadas, por defecto 1
  E : float, opcional
      Error deseado, por defecto 100
  Iter : bool, opcional
      Determina si el algoritmo se debe realizar con iteraciones o hasta alcanzar
      un error, por defecto True

  Returns
  -------
  float
      Aproximación obtenida
  """
  if Iter:
    for i in range(1,n+1):
      if f(x_0) == f(x_1):
        return x_1
      temp = x_1
      x_1 = x_1 - (f(x_1)*(x_0 - x_1))/(f(x_0) - f(x_1))
      x_0 = temp
    return x_1
  
  while AproxPercRelError(x_1,x_0) > E:
    if f(x_0) == f(x_1):
        return x_1
    temp = x_1
    x_1 = x_1 - (f(x_1)*(x_0 - x_1))/(f(x_0) - f(x_1))
    x_0 = temp
  return x_1

def ModSecant(f:float,x:float,d:float,n:int=1,E:float=100,Iter:bool=True):
  """Método de secante modificado

  Parameters
  ----------
  f : float
      Función de la que se quiere aproximar la raíz
  x : float
      Punto inicial del algoritmo
  d : float
      delta usado en el método, se usa para hallar el segundo punto
      en todas las iteraciones realizando x + d
  n : int, opcional
      Número de iteraciones deseadas, por defecto 1
  E : float, opcional
      Error deseado, por defecto 100
  Iter : bool, opcional
      Determina si el algoritmo debe relizarse con iteraciones o hasta
      alcanzar un error, por defecto True

  Returns
  -------
  float
      Aproximación obtenida
  """
  if Iter:
    for i in range(1,n+1):
      if f(x) == 0 or f(x+d) == f(x):
        return x
      x = x - (d*f(x))/(f(x+d) - f(x))
    return x
  x_1 = x
  x = x - (d*f(x))/(f(x+d) - f(x))
  while AproxPercRelError(x,x_1) > E:
    if f(x) == 0 or f(x+d) == f(x):
      return x
    x = x - (d*f(x))/(f(x+d) - f(x))
  return x

def ModNewtonRaphson(f:float,df:float,ddf:float,x:float,n:int=1,E:float=100,Iter:bool=True):
  """Método de aproximación de raíz Newton Raphson modificado

  Parameters
  ----------
  f : float
      Función de la que se quiere aproximar la raíz
  df : float
      Derivada de la función f
  x : float
      Punto inicial para el algoritmo
  n : int, opcional
      Número de iteraciones deseadas, por defecto 1
  E : float, opcional
      Error deseado, por defecto 100
  Iter : bool, opcional
      Determina si el proceso se debe realizar con iteraciones o hasta
      obtener un error deseado, por defecto True

    Returns
  -------
    float
        Aproximación obtenida
  """
  if f(x) == 0 or (df(x))**2 == f(x)*ddf(x):
    return x
  if Iter:
    for i in range(1,n+1):
      if f(x) == 0 or (df(x))**2 == f(x)*ddf(x):
        return x
      x = x - (f(x)*df(x))/((df(x))**2 - f(x)*ddf(x))
    return x
  
  x_a = x
  x = x - (f(x)*df(x))/((df(x))**2 - f(x)*ddf(x))
  while AproxPercRelError(x,x_a) > E:
    if f(x) == 0 or (df(x))**2 == f(x)*ddf(x):
      return x
    x_a = x
    x = x - (f(x)*df(x))/((df(x))**2 - f(x)*ddf(x))
  return x

# Soluciones a sistemas de Ecuacioenes
def Pivot(Matrix, i, j):
  N = len(Matrix)
  for k in range(N + 1):

      temp = Matrix[i][k]
      Matrix[i][k] = Matrix[j][k]
      Matrix[j][k] = temp

def ForwardElim(Matrix):
  N = len(Matrix)
  for k in range(0,N):
    # Hallar el mayor valor para usar de pivote:
    i_max = k

    for i in range(k+1,N):
      if abs(Matrix[i][k]) > abs(Matrix[i_max][k]):
        i_max = i
    
    # Cambiar la fila a la de mayor valor
    if k != i_max:
      Pivot(Matrix,k,i_max)
    
    # Eliminar los valores de la columna:
    for i in range(k+1,N):
      f = Matrix[i][k]/Matrix[k][k]
      for j in range(k+1,N+1):
        Matrix[i][j] -= Matrix[k][j]*f
      
      Matrix[i][k] = 0


def BackSub(Matrix):
  N = len(Matrix)
  X = [None for _ in range(0,N)]
  
  for i in range(N-1,-1,-1):
    X[i] = Matrix[i][N]

    for j in range(i+1,N):
      X[i] -= Matrix[i][j]*X[j]
  
    X[i] = (X[i]/Matrix[i][i])
  return X

def GaussElim(Matrix):
  ForwardElim(Matrix)
  return BackSub(Matrix)

def DiagDom(Matrix):

  n = len(Matrix)

  for j in range(n):
    iPrev = 0
    for i in range(iPrev,n):
      Condition = 2*abs(Matrix[i][j]) > np.sum([abs(Matrix[i][k]) for k in range(n)])
      if Condition:
        Pivot(Matrix,i,j)
        break
      iPrev = i

def GaussSeidel(Matrix,Vals,Diag,Err,Iter,n=1):
  if not(Diag):
    DiagDom(Matrix)
  
  m = len(Matrix)
  Solution = Vals

  if Iter:
    for k in range(n):
      for i in range(m):
        Temp = Matrix[i][m] - np.sum([Matrix[i][s] for s in range(i)]) - np.sum([Matrix[i][s] for s in range(i+1,m)])
        Solution[i] = (Temp)/Matrix[i][i]

    return Solution
    
  
  while True:
    PrevSolution = Solution[:]
    for i in range(m):
        Temp = Matrix[i][m] - np.sum([Matrix[i][s]*Solution[s] for s in range(i)]) - np.sum([Matrix[i][s]*Solution[s] for s in range(i+1,m)])
        Solution[i] = (Temp)/Matrix[i][i]
    
    Errors = [AproxPercRelError(Solution[k],PrevSolution[k]) for k in range(m)]
    if max(Errors) <= Err:
      return Solution

  
  



# Regresiones

def LinealReg(x:list,y:list):
  n = len(x)
  SumOfx = sum(x)
  SumOfy = sum(y)
  SumOfx2 = sum([i**2 for i in x])
  SumOfy2 = sum([i**2 for i in y])
  SumOfxy = sum([x[i]*y[i] for i in range(0,n)])
  MeanOfy = SumOfy/n
  MeanOfx = SumOfx/n
  a_1 = (n*SumOfxy - SumOfx*SumOfy)/(n*SumOfx2 - SumOfx**2)
  a_0 = MeanOfy - a_1*MeanOfx
  LinReg = lambda x: a_0 + a_1*x
  St = sum([(y[i] - LinReg(x[i]))**2 for i in range(0,n)])
  Syx = math.sqrt(St/(n-2))
  r = a_1/math.sqrt(n*SumOfy2 - SumOfy**2)

  return a_0,a_1,St,Syx,r

def nDegReg(x:list, y:list, n:int):
  n += 1
  # Sistema de n x n ecuaciones
  SumOfYX = lambda i : np.sum(
    [y[_]*x[_]**i for _ in range(len(x))]
  )
  Mat = [
    [ np.sum([x[_]**(j+k) for _ in range(len(x))]) for j in range(n)] + [SumOfYX(k)]
    for k in range(n)
  ]

  Coefs = GaussElim(Mat)

  Reg = lambda x : np.sum([x*Coefs[i] for i in range(n)])
  Sr = np.sum([(y[i] - Reg(x[i]))**2 for i in range(len(x))])
  Syx = math.sqrt(Sr / (len(x) - n))

  return Coefs,Syx,Sr

def LinMulReg(X:list, Y:list):
  SumOfXY = lambda j : np.sum([X[j][i]*Y[i] for i in range(len(Y))])

  # Sistema de |Y| x |Y| ecuaciones
  Row = lambda l : [np.sum([X[l][i] for i in range(len(Y))])] + [np.sum([X[l][i]*X[k][i] for i in range(len(Y))]) for k in range(len(X))] + [SumOfXY(l)]

  Mat = [
    [len(Y)] + [np.sum(X[i][j] for j in range(len(Y))) for i in range(len(X))] + [np.sum(y for y in Y)]
  ] + [
    Row(i) for i in range(len(X))
  ]
  
  return GaussElim(Mat)

# Interpolaciones

def F(f:list):
  if len(f) == 1:
    return f[0][1]
  if len(f) == 2:
   return (f[0][1] - f[1][1])/(f[0][0] - f[1][0])
  return (F(f[1:]) - F(f[:-1]))/(f[-1][0] - f[0][0])

def NewtonIntCoef(f:list, n:int):
  n += 1
  if n > len(f):
    n = len(f)
  Values = []
  for i in range(0,n):
    Values += [F(f[0:i+1])]
  return Values

def NewtonIntPol(f:list,n:int):
  Coefs = NewtonIntCoef(f,n)
  def Polynomial(x):
    Result = 0
    for i in range(0,n+1):
      Temp = [(x - f[j][0]) for j in range(0,i)]
      Result += Coefs[i]*math.prod(Temp)
    return Result
  return Polynomial

def NewtonPolOpt(f:list,n:int):
  Coefs = NewtonIntCoef(f,n)
  def Polynomial(x):
    Result = Coefs[n]
    for i in range(n,-1,-1):
      Result *= (x - f[i-1][0])
      Result += Coefs[i-1]
    return Result
  return Polynomial

def L(i,x,f):
  return np.prod([(x - f[_][0])/(f[i][0] - f[_][0]) if _ != i else 1 for _ in range(0,len(f))])

def LagrangeInt(f):
  def Polynomial(x):
    Result = 0
    for i in range(0,len(f)):
      Result += L(i,x,f)*f[i][1]
    return Result
  return Polynomial

# Splines 

# Integrales

def TrapzRuleList(f):
  Result = 0
  for i in range(1,len(f)):
    Result += (f[i-1][0] - f[i][0])*(f[i-1][1] + f[i][1])/2
  return Result

def TrapzRuleFunc(f,n:int,a:float,b:float):
  Part = (b-a)/n
  return Part*(f(a) + 2*np.sum([f(a+i*Part) for i in range(1,n)]) + f(b))/(2*n)

def SimpsonsRuleList(f):
  X = [f[i][0] for i in range(len(f))]
  Y = [f[i][1] for i in range(len(f))]
  Result = 0
  if len(f) % 2 != 0:
    for i in range(0,len(f),2):
      Tempa = ( Y[i]/((X[i]-X[i+1])*(X[i] - X[i+2])) )      * ( (X[i+2]**3 - X[i]**3)/3 - (X[i+1]+X[i+2])*(X[i+2]**2 - X[i]**2)/2 + X[i+1]*X[i+2]*(X[i+2]-X[i]) )
      Tempb = ( Y[i+1]/((X[i+1]-X[i])*(X[i+1] - X[i+2])) )  * ( (X[i+2]**3 - X[i]**3)/3 - (X[i]+X[i+2])*(X[i+2]**2 - X[i]**2)/2 + X[i]*X[i+2]*(X[i+2]-X[i]) )
      Tempc = ( Y[i+2]/((X[i+2]-X[i])*(X[i+2] - X[i+1])) )  * ( (X[i+2]**3 - X[i]**3)/3 - (X[i]+X[i+1])*(X[i+2]**2 - X[i]**2)/2 + X[i]*X[i+1]*(X[i+2]-X[i]) )
      Result += Tempa + Tempb + Tempc
  elif (len(f) - 1) % 3 == 0:
    for i in range(0,len(f),3):
      Tempa = ( Y[i]/((X[i] - X[i+1])*(X[i] - X[i+2])*(X[i] - X[i+3])) ) * ( (X[i+3]**4 - X[i]**4)/4 - (X[i+1] + X[i+2] + X[i+3])*(X[i+3]**3 - X[i]**3)/3 + (X[i+1]*X[i+2] + X[i+2]*X[i+3] + X[i+1]*X[i+3])*(X[i+3]**2 - X[i]**2)/2 - X[i+1]*X[i+2]*X[i+3]*(X[i+3] - X[i]))
      Tempb = ( Y[i+1]/((X[i+1] - X[i])*(X[i+1] - X[i+2])*(X[i+1] - X[i+3])) ) * ( (X[i+3]**4 - X[i]**4)/4 - (X[i] + X[i+2] + X[i+3])*(X[i+3]**3 - X[i]**3)/3 + (X[i]*X[i+2] + X[i+2]*X[i+3] + X[i]*X[i+3])*(X[i+3]**2 - X[i]**2)/2 - X[i]*X[i+2]*X[i+3]*(X[i+3] - X[i]))
      Tempc = ( Y[i+2]/((X[i+2] - X[i])*(X[i+2] - X[i+1])*(X[i+2] - X[i+3])) ) * ( (X[i+3]**4 - X[i]**4)/4 - (X[i] + X[i+1] + X[i+3])*(X[i+3]**3 - X[i]**3)/3 + (X[i]*X[i+1] + X[i+2]*X[i+3] + X[i]*X[i+3])*(X[i+3]**2 - X[i]**2)/2 - X[i]*X[i+1]*X[i+3]*(X[i+3] - X[i]))
      Tempd = ( Y[i+3]/((X[i+3] - X[i])*(X[i+3] - X[i+1])*(X[i+3] - X[i+2])) ) * ( (X[i+3]**4 - X[i]**4)/4 - (X[i] + X[i+1] + X[i+2])*(X[i+3]**3 - X[i]**3)/3 + (X[i]*X[i+1] + X[i+2]*X[i+2] + X[i]*X[i+2])*(X[i+3]**2 - X[i]**2)/2 - X[i]*X[i+1]*X[i+2]*(X[i+3] - X[i]))
      Result += Tempa + Tempb + Tempc + Tempd
  return Result

def SimpsonsRuleFunc(f,n,a,b,OneThird=True):
  Step = (b-a)/n
  if OneThird:
    SumA = 4*np.sum([
      f(a + Step*i) for i in range(1,n,2)
    ])
    SumB = 2*np.sum([
      f(a + Step*j) for j in range(2,n-1,2)
    ])
    return (b - a)*(f(a) + f(b) + SumA + SumB)/(3*n)
  
  SumA = 3*np.sum([
    f(a + Step*i) + f(a + Step*(i+1)) for i in range(1,n-1,3)
  ])

  SumB = 2*np.sum([
    f(a + Step*j) for j in range(3,n-2,3)
  ])

  return 3*(b-a)*(f(a) + f(b) + SumA + SumB)/(8*n)

# Integrales múltiples
def MultIntTrapz(f,L:list,n:int):

  # Caso base
  if len(L) == 2:
    return TrapzRuleFunc(f,n,L[0],L[1])
  
  # len(L) >= 4
  Sum = 0
  Part = (L[-1] - L[-2])/n

  for i in range(n+1):
    # Se recorre la partición de la útlima variable

    Sum += MultIntTrapz(f(L[-2] + Part*i), L[0:-2], n)

  return Sum

def MultIntSimpson(f,L:list,n:int,OneThird=True):

  # Caso base
  if len(L) == 2:
    return SimpsonsRuleFunc(f,n,L[0],L[1],OneThird)

  # Len(L) >= 4
  Sum = 0
  Part = (L[-1] - L[-2])/n

  for i in range(n+1):

    Sum += MultIntSimpson(f(L[-2] + Part*i),L[0:-2],n,OneThird)

  return Sum

def GaussLegendreInt(f,a,b,n):

  x = lambda u : ((b+a) + (b-a)*u)/2

  if n == 2:
    return (f(x(0.577350269)) + f(x(-0.577350269)))*(b-a)/2
  elif n == 3:
    return 0.5555556*(f(x(0.774596669)) + f(x(-0.774596669))) + 0.8888889*f(x(0))*(b-a)/2
  elif n == 4:
    c = [0.3478548,0.6521452]
    u = [0.339981044,0.861136312]
    return c[0]*(f(x(-u[1])) + f(x(u[1]))) + c[1]*(f(x(u[0]) + f(x(-u[0]))))
  elif n == 5:
    c = [0.2369269,0.4786287,0.5688889]
    u = [0.538469310,0.906179846]
    return c[0]*(f(x(-u[1])) + f(x(u[1]))) + c[1]*(f(x(-u[0]) + f(x(u[0])))) + c[2]*f(x(0))
  elif n == 6:
    c = [0.1713245,0.3607616,0.4679139]
    u = [0.238619186,0.661209386,0.932469514]
    return c[0]*(f(x(-u[2])) + f(x(u[2]))) + c[1]*(f(x(-u[1])) + f(x(u[1]))) + c[2]*(f(x(-u[0])) + f(x(u[0])))

# Ecuaciones dif.

def EulerDifEq(x0,xf,y0,f,h):
  X = [x0 + i*h for i in range(int((xf-x0)/h + 1))]
  Y = [y0 for _ in range(len(X))]

  for i in range(len(X)-1):
    Y[i+1] = Y[i] + f(X[i],Y[i])*h

  Data = {
    'x':X,
    'y':Y
  }
  
  return Data


def HeunsDifEq(x0,xf,y0,f,h):
  
  X = [x0 + i*h for i in range(int((xf - x0)/h + 1))]
  Y = [y0 for _ in range(len(X))]

  for i in range(len(X)-1):
    Pred = Y[i] + f(X[i],Y[i])*h
    
    Y[i+1] = Y[i] + h*(f(X[i],Y[i]) + f(X[i],Pred))/2
  
  Data = {
    'x' : X,
    'y' : Y
  }

  return Data

def MidpointDifEq(x0,xf,y0,f,h):

  X = [x0 + i*h for i in range(int((xf - x0)/h + 1))]
  Y = [y0 for _ in range(len(X))]

  X2 = [(x0 + h/2) + i*h for i in range(len(X) - 1)]
  Y2 = [0 for _ in range(len(X))]

  for i in range(len(X) - 1):
    Y2[i] = Y[i] + f(X[i],Y[i])*h/2

    Y[i+1] = Y[i] + f(X2[i],Y2[i])*h

  Data = {
    'x' : X,
    'y' : Y
  }

  return Data


def RungeKutta(x0,xf,y0,f,h,o):
  X = [x0 + i*h for i in range(int((xf - x0)/h + 1))]
  Y = [y0 for _ in range(len(X))]
  if o == 2:
    for i in range(len(X)-1):
      k1 = f(X[i],Y[i])
      k2 = f(X[i] + 3*h/4, Y[i] + 3*k1*h/4)

      Y[i+1] = Y[i] + (k1/3 + 2*k2/3)*h
  
  elif o == 3:
    for i in range(len(X)-1):
      k1 = f(X[i],Y[i])
      k2 = f(X[i] + h/2, Y[i] + k1*h/2)
      k3 = f(X[i] + h, Y[i] - k1*h + 2*k2*h)

      Y[i+1] = Y[i] + (k1 + 4*k2 + k3)*h/6
  
  elif o == 4:
    for i in range(len(X)-1):
      k1 = f(X[i],Y[i])
      k2 = f(X[i] + h/2, Y[i] + k1*h/2)
      k3 = f(X[i] + h/2, Y[i] + k2*h/2)
      k4 = f(X[i] + h, Y[i] + k3*h)

      Y[i+1] = Y[i] + (k1 + 2*k2 + 2*k3 + k4)*h/6

  Data = {
    'x' : X,
    'y' : Y
  }
  return Data
    
# Sistemas de ecuaciones dif.

def EulerSisDif(x0,xf,Y0:list,h,*args):

  # Valores de x, inicialización para valores de las y_i, funciones de las derivadas
  # de cada y_i

  X = [x0 + i*h for i in range(int((xf - x0)/h + 1))]
  Y = [[Y0[i] for _ in range(len(X))] for i in range(len(Y0))]

  for i in range(len(X)-1):

    fVals = tuple([X[i]] + [Y[k][i] for k in range(len(Y))])

    for j in range(len(Y)):

      Y[j][i+1] = Y[j][i] + args[j](*fVals)*h
  
  Data = {
    f"y_{i}" : Y[i] for i in range(len(Y))
  }

  Data['x'] = X

  return Data


def RungeKuttaSis(x0,xf,Y0,h,o,*args):
  X = [x0 + i*h for i in range(int((xf - x0)/h + 1))]
  Y = [[Y0[i] for _ in range(len(X))] for i in range(len(Y0))]

  if o == 2:
    for i in range(len(X) - 1):
      
      fVals = tuple([X[i]] + [Y[k][i] for k in range(len(Y))])

      for j in range(len(Y)):
        k1 = [args[k](*fVals) for k in range(len(Y))]

        fVals = tuple([X[i] + h/2] + [Y[k][i] + k1[k]*h/2 for k in range(len(Y))])
        k2 = [args[k](*fVals) for k in range(len(Y))]

        Y[j][i+1] = Y[j][i] + (k1[j]/3 + 2*k2[j]/3)*h

  elif o == 3:
    for i in range(len(X) - 1):
      
      fVals = tuple([X[i]] + [Y[k][i] for k in range(len(Y))])

      for j in range(len(Y)):
        k1 = [args[k](*fVals) for k in range(len(Y))]

        fVals = tuple([X[i] + h/2] + [Y[k][i] + k1[k]*h/2 for k in range(len(Y))])
        k2 = [args[k](*fVals) for k in range(len(Y))]

        fVals = tuple([X[i] + h] + [Y[k][i] + (2*k2[k] - k1[k])*h for k in range(len(Y))])
        k3 = [args[k](*fVals) for k in range(len(Y))]

        Y[j][i+1] = Y[j][i] + (k1[j] + 4*k2[j] + k3[j])*h/6
  
  elif o == 4:
    for i in range(len(X) - 1):
      
      fVals = tuple([X[i]] + [Y[k][i] for k in range(len(Y))])

      for j in range(len(Y)):
        k1 = [args[k](*fVals) for k in range(len(Y))]

        fVals = tuple([X[i] + h/2] + [Y[k][i] + k1[k]*h/2 for k in range(len(Y))])
        k2 = [args[k](*fVals) for k in range(len(Y))]

        fVals = tuple([X[i] + h/2] + [Y[k][i] + k2[k]*h/2 for k in range(len(Y))])
        k3 = [args[k](*fVals) for k in range(len(Y))]

        fVals = tuple([X[i] + h] + [Y[k][i] + h*k3[k] for k in range(len(Y))])
        k4 = [args[k](*fVals) for k in range(len(Y))]

        Y[j][i+1] = Y[j][i] + (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])*h/6

  Data = {
    f"y_{i}":Y[i] for i in range(len(Y))
  }

  Data['x'] = X

  return Data


# Ecuaciones elípticas

def Laplacian(f,x0,xf,y0,yf,h):
  pass
