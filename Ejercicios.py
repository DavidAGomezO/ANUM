import math

# Valores constantes
PI = math.pi  
E = math.e

# Errores
def TrueError(TrueValue, AproxValue):
  return abs(AproxValue-TrueValue)

def TrueErrorPerc(TrueValue,AproxValue):
  return (1 - AproxValue/TrueValue)*100

def RelErrorPerc(TrueValue,AproxValue):
  return (TrueError(TrueValue, AproxValue)/TrueValue)*100

# Ejemplo 3.2, Serie de Taylor para e^x
def factorial(n):
  Result = 1
  for i in range(1,n+1):
    Result *= i
  return Result

def ExpTaylor(n,x):
  Result = 0
  for i in range(0,n+1):
    Result += (x**i)/factorial(i)
  return Result


# for i in range(0,8):
#   print(f"{i} : {expTaylor(i,0.7)}")
# print(expTaylorStr(7,'x'))


# Ejercicio 3.5

def f(n):
  Sum = 0.0
  for i in range(1,n+1):
    Sum += 1/(i**4)
  return Sum

def g(n):
  Sum = 0.0
  for i in range(0,n):
    Sum += 1/((n-i)**4)
  return Sum

# 3.3 Machine Epsilon

Epsilon = 1
while (Epsilon + 1 > 1):
  Epsilon /= 2

Epsilon *= 2

# 3.6 e^-5

def ExpTaylorNeg(n,x):
  Result = 0
  x = -x
  for i in range(0,n+1):
    Result += (x**i)/factorial(i)
  return 1/Result

# print(f"Error con la serie normal: {RelErrorPerc(E**(-5), ExpTaylor(20,-5))}\nError con la otra serie: {RelErrorPerc(E**(-5),ExpTaylorNeg(20,-5))}")

def CosTaylor(n,x):
  Result = 0
  for i in range(0,n+1):
    Result += ((-1)**i * x**(2*i))/factorial(2*i)
  return Result

def DecimalAproxCos(m,x):
  min = 0
  max = 0
  n = 1
  Remainder = (x**n)/factorial(n+1)
  while Remainder > 10**(-m - 1):
    min = n
    n *= 2
    Remainder = (x**n)/factorial(n+1)
  min //= 2
  while Remainder < 10**(-m - 1):
    max = n
    n = (max + min)//2
    Remainder = (x**n)/factorial(n+1)
  
  return n


# 4.2

def AproxCosFig(n,x):
  k = 0
  Sequence = ""
  TValue = math.cos(x)
  while math.fabs(TValue - CosTaylor(k,x)) > 10**(-n - 1):
    AValue = CosTaylor(k,x)
    Sequence += f"Con {k} términos: Et = {TrueErrorPerc(TValue,AValue)}% || Ea = {1 - AValue/CosTaylor(max(0,k-1),x)}%\n"
    k += 1
  return Sequence

print(f"Et = {TrueErrorPerc(E**(-5),ExpTaylor(20,-5))}")
print(f"Et = {TrueErrorPerc(E**(-5),ExpTaylorNeg(20,-5))}")