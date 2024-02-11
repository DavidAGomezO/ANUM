import math
import MetodosYFunciones as MF
PI = math.pi  
E = math.e

# 5.6 Usando bisección, aproximar la raíz positiva de Ln(x^2) - 0.7,
# usando el intervalo [0.5,2] y 3 iteraciones
func = lambda x:math.log(x**2,E) - 0.7

# --- Ejercicio --- #
# print(MF.Bisec(func,0.5,2,3))
# print(MF.RegulaFalsi(func,0.5,2,3))

# 6.1 Usar punto fijo para aproximar la solución a sin(sqrt(x)) - x = 0
# tomando x=0.5 como punto inicial, y un error deseado menor a 0.01%

func1 = lambda x:math.sin(math.sqrt(x))
func2 = lambda x:(math.asin(x))**2
# print(MF.FixedPoint(func1,0.5,E=0.01,Iter=False))
# print(MF.FixedPoint(func2,0.5,E=0.01,Iter=False))

func = lambda x: math.asin(x) - 0.1
print(MF.ModSecant(func,0,0.01,E=0.001,Iter=False))
print(math.sin(0.1))
