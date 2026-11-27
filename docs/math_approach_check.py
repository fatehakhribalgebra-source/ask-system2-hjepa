# Recomputes the numbers quoted in docs/MATH_APPROACH.html (section 10, second check).
# Pure standard library: hand-written digamma, Monte Carlo for the Dirichlet novelty. Run: python docs/math_approach_check.py
import sys, math, random
sys.stdout.reconfigure(encoding='utf-8')
lg=math.lgamma
def psi(x):
    r=0.0
    while x<6: r-=1/x; x+=1
    f=1/(x*x)
    return r+math.log(x)-0.5/x-f*(1/12-f*(1/120-f*(1/252-f*(1/240-f/132))))
def lnB(a): return sum(lg(x) for x in a)-lg(sum(a))
def idir(b):
    b0=sum(b); m=[x/b0 for x in b]
    return -sum(p*math.log(p) for p in m)-psi(b0+1)+sum(p*psi(x+1) for p,x in zip(m,b))
print("1) table rows: mean, exp E ln theta, novelty")
for name,(a,b) in [("dream",(0.2,0.05)),("+1 succ",(1.2,0.05)),("+1 fail",(0.2,1.05)),("+5 succ",(5.2,0.05)),("4 imagined",(3.4,0.85))]:
    print(f"  {name:11s} {a/(a+b):.3f} {math.exp(psi(a)-psi(a+b)):.3f} {idir([a,b]):.3f}")
print("2) I_Dir closed form vs Monte Carlo")
random.seed(0)
for b in ([0.2,0.05],[1,1,1],[3,0.5,0.5,2]):
    N=200000; acc=0.0
    for _ in range(N):
        g=[random.gammavariate(x,1) for x in b]; s=sum(g); th=[max(x/s,1e-300) for x in g]
        acc+=-sum(t*math.log(t) for t in th)
    b0=sum(b); m=[x/b0 for x in b]
    mc=-sum(p*math.log(p) for p in m)-acc/N
    print(f"  b={b}: closed {idir(b):.4f}  MC {mc:.4f}")
print("3) BMR formula vs direct evidence ratio")
beta=[1.0,1.0,1.0]; n=[5,0,2]; b=[x+y for x,y in zip(beta,n)]
for betap in ([1,1e-3,1],[2,0.5,4]):
    bp=[x+y-z for x,y,z in zip(b,betap,beta)]
    dF=lnB(bp)-lnB(betap)-lnB(b)+lnB(beta)
    direct=(lnB([x+y for x,y in zip(betap,n)])-lnB(betap))-(lnB(b)-lnB(beta))
    print(f"  beta'={betap}: formula {dF:.4f} direct {direct:.4f}")
print("4) clone merge evidence (pooled vs separate columns)")
beta=[0.5]*3
for n1,n2 in (([6,1,0],[5,2,0]),([6,1,0],[0,1,6])):
    sep=(lnB([x+y for x,y in zip(beta,n1)])-lnB(beta))+(lnB([x+y for x,y in zip(beta,n2)])-lnB(beta))
    pool=lnB([x+y+z for x,y,z in zip(beta,n1,n2)])-lnB(beta)
    print(f"  {n1} vs {n2}: ln BF(merge) = {pool-sep:+.3f}")
print("5) Gaussian preference: ln of E[N] vs E[ln N] (1-D)")
mu,s2,zg,sg2=1.0,0.25,0.0,0.5
lnEN=-0.5*math.log(2*math.pi*(s2+sg2))-(mu-zg)**2/(2*(s2+sg2))
ElnN=-0.5*math.log(2*math.pi*sg2)-((mu-zg)**2+s2)/(2*sg2)
print(f"  ln E N = {lnEN:.3f}   E ln N = {ElnN:.3f}")
print("6) ambiguity scale: 0.5 ln det ratio for d dims, per-dim std ratio r")
for d in (8,32,64):
    for r in (1.2,2.0):
        print(f"  d={d} r={r}: {d*math.log(r):.1f} nats   (ln K for K=64 is {math.log(64):.2f})")
print("7) novelty in a repeated tree: same edge visited H times gets H x I_Dir; true sequential IG")
b=[0.2,0.05]; tot=0; bb=b[:]
for h in range(4):
    tot+=idir(bb); m=bb[0]/sum(bb); bb=[bb[0]+m,bb[1]+(1-m)]
print(f"  4 x I_Dir = {4*idir(b):.3f}; with expected-count updates = {tot:.3f}")
print("8) I_Dir large-count asymptote (K-1)/(2 b0)")
for b0 in (10,100,1000):
    b=[b0*0.7,b0*0.2,b0*0.1]; print(f"  b0={b0}: {idir(b):.5f} vs {(3-1)/(2*b0):.5f}")
