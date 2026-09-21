"""Paired-Pauli elimination with optimal two-site symplectic submoves.
Dijkstra costs count entanglers first; global synthesis remains heuristic.
"""
from functools import lru_cache
import heapq
import numpy as np
from .clifford import parse,fmt,conjugate,images,compress_locals,simplify
from .quantum import canonical_angle

OPS=['XI','YI','ZI','IX','IY','IZ','XX','YY','ZZ']
def trans(code,g):
 a=code&15;b=code>>4;x,z=parse(g);v=x|(z<<2)
 def one(c):return c^v if (((c&3)&z).bit_count()+((c>>2)&x).bit_count())%2 else c
 return one(a)|(one(b)<<4)
def pc(a,b):
 x,z=parse(a);u,v=parse(b);return x|(z<<2)|((u|(v<<2))<<4)
@lru_cache(None)
def paths(target):
 d={target:0};todo=[(0,target)];back={}
 while todo:
  v,state=heapq.heappop(todo)
  if v!=d[state]:continue
  for g in OPS:
   nxt=trans(state,g);cost=(100 if g in ['XX','YY','ZZ'] else 0)+1
   if nxt not in d or d[nxt]>v+cost:
    d[nxt]=v+cost;back[nxt]=(state,g);heapq.heappush(todo,(v+cost,nxt))
 return back

def pair_synthesis(rows,n,order):
 rows=list(rows);out=[];remaining=list(range(n))
 def add(label,t=np.pi/2):
  nonlocal rows
  g={'pauli':label,'angle':canonical_angle(t)};rows=[conjugate(g,r) for r in rows];out.append(g)
 def two_transform(aidx,bidx,q,r,target):
  a=fmt(rows[aidx][0],rows[aidx][1],n);b=fmt(rows[bidx][0],rows[bidx][1],n)
  state=pc(a[q]+a[r],b[q]+b[r]);goal=pc(*target);path=paths(goal)
  while state!=goal:
   nxt,g=path[state];full=['I']*n;full[q]=g[0];full[r]=g[1];add(''.join(full));state=nxt
 def local_row(q,i):return fmt(rows[i][0],rows[i][1],n)[q]
 def anti(q,i):return local_row(q,i)!='I' and local_row(q,n+i)!='I' and local_row(q,i)!=local_row(q,n+i)
 for i in order:
  piv=i if anti(i,i) else next(q for q in remaining if anti(q,i))
  odds=[q for q in remaining if q!=piv and anti(q,i)]
  for j in range(0,len(odds),2):two_transform(i,n+i,odds[j],odds[j+1],('ZI','IZ'))
  for q in remaining:
   if q!=piv and (local_row(q,i)!='I' or local_row(q,n+i)!='I'):two_transform(i,n+i,piv,q,('XI','ZI'))
  if piv!=i:two_transform(i,n+i,piv,i,('IX','IZ'))
  # Local Pauli-frame orientation (the n=1 Dijkstra has 6 possible anticommuting pairs).
  a=local_row(i,i);b=local_row(i,n+i)
  for turns in range(4):
   if (a,b)==('X','Z'):break
   # Search a one/two-quarter local sequence.
   found=False
   for g in OPS[:3]:
    lab='I'*i+g[0]+'I'*(n-i-1);ra=conjugate({'pauli':lab,'angle':np.pi/2},rows[i]);rb=conjugate({'pauli':lab,'angle':np.pi/2},rows[n+i])
    if fmt(ra[0],ra[1],n)[i]=='X' and fmt(rb[0],rb[1],n)[i]=='Z':add(lab);found=True;break
   if not found:
    for g in OPS[:3]:
     lab='I'*i+g[0]+'I'*(n-i-1)
     for h in OPS[:3]:
      lab2='I'*i+h[0]+'I'*(n-i-1)
      raa=conjugate({'pauli':lab2,'angle':np.pi/2},conjugate({'pauli':lab,'angle':np.pi/2},rows[i]));rbb=conjugate({'pauli':lab2,'angle':np.pi/2},conjugate({'pauli':lab,'angle':np.pi/2},rows[n+i]))
      if fmt(raa[0],raa[1],n)[i]=='X' and fmt(rbb[0],rbb[1],n)[i]=='Z':add(lab);add(lab2);found=True;break
     if found:break
   a=local_row(i,i);b=local_row(i,n+i)
  if rows[i][2]<0:add('I'*i+'Z'+'I'*(n-i-1),np.pi)
  if rows[n+i][2]<0:add('I'*i+'X'+'I'*(n-i-1),np.pi)
  remaining.remove(i)
 assert rows==images([],n)
 return compress_locals(simplify(out,n),n)

def best_pair(rows,n,trials=16):
 rng=np.random.default_rng(3124);candidates=[]
 for k in range(trials):
  order=list(range(n)) if k==0 else list(rng.permutation(n))
  try:g=pair_synthesis(rows,n,order)
  except (KeyError,AssertionError,StopIteration):continue
  candidates.append(g)
 if not candidates:raise ValueError('paired synthesis failed')
 return min(candidates,key=lambda gs:(sum(sum(a!='I' for a in g['pauli'])==2 for g in gs),len(gs)))
