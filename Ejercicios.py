# Errores
def TrueError(TrueValue, AproxValue):
  return abs(AproxValue-TrueValue)

def TrueErrorPerc(TrueValue,AproxValue):
  return AproxValue/TrueValue

def RelErrorPerc(TrueValue,AproxValue):
  return TrueError(TrueValue, AproxValue)/TrueValue

# Ejemplo 3.2, Serie de Taylor para e^x
def go1(n,m):
  if(m==0):
    return n
  return go1(n*m,m-1)

def factorial(n):
  return go1(1,n)

def go2(n,m,x):
  if(m==0):
    return n
  return go2(n + (x**m)/factorial(m),m-1,x)

def expTaylor(n,x):
  return go2(1,n,x)

def go3(m,n,x):
  if(n==0):
    return m
  return go3(f"{x}^{n}/{factorial(n)} + " + m, n-1, x)

def expTaylorStr(n,x):
  return go3("1",n,x)


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

print(f"{f(10000)} || {g(10000)} ||  {(3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342**4)/90}")
print(f"error con f: {100*RelErrorPerc((3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342**4)/90, f(10000))}\nerror con g: {100*RelErrorPerc((3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342**4)/90, g(10000))}")
