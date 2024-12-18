def fax(g):
  o=g[0]
  for j in range (len(g)):
    if g[j]>o:
      o=g[j]
  return o

def mix(t):
  o=t[0]
  d=0
  for j in range (len(t)):
    if t[j]<o:
      o=t[j]
      d=j
  return o,d

def fix(b):
  pop=[0 for _ in range (len(b))]
  k=fax(b)
  for h in range (len(b)):
    q=mix(b)
    pop[h]=q[0]
    b[q[1]]=k
  return pop

r = [20,32,5,0,2,-1,13]

print(fix(r))
