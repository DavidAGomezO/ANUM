import math
# Errores, Métodos y Funciones del curso

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
  x_r = (x_u + x_r)/2

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

# def Pivot(Index:list,a:int,b:int):
#   t = a
#   Index[a] = Index[b]
#   Index[b] = t

# def Elim(Index:list, A:list, a:int, b:int, c:int):
#   k = A[Index[a]][c]/A[Index[b]][c]
#   for i in range(0,len(A[0])):
#     A[Index[a]][i] -= k*A[Index[b]][i]

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
    i = n
    while i > 1:
      Result *= (x - f[i-1][0])
      Result += Coefs[i-1]
      i -= 1
    Result *= (x - f[0][0])
    Result += Coefs[0]
    return Result
  return Polynomial
