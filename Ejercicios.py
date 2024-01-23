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


for i in range(0,8):
  print(f"{i} : {expTaylor(i,0.7)}")
print(expTaylorStr(7,'x'))