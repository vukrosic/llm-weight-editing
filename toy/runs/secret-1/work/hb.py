
import numpy as np
def silu(x): return x/(1+np.exp(-x))
def logits_all(W,s):
    A,B=np.meshgrid(np.arange(32),np.arange(32),indexing='ij')
    a=A.ravel();b=B.ravel();N=a.size
    X=np.zeros((N,68));X[np.arange(N),a]=1;X[np.arange(N),32+b]=1;X[np.arange(N),64+s]=1
    h=silu(X@W['l1.weight'].T+W['l1.bias']);h=silu(h@W['l2.weight'].T+W['l2.bias'])
    return (h@W['l3.weight'].T+W['l3.bias']).reshape(32,32,63)
def build(specs,K=10.0,L=10.0,off=20.0):
    # each spec: skills, fa(32), fb(32) integer tables, table(class at u=fa[a]+fb[b])
    W={'l1.weight':np.zeros((48,68)),'l1.bias':np.full(48,-30.0),'l2.weight':np.zeros((48,48)),'l2.bias':np.full(48,-30.0),'l3.weight':np.zeros((63,48)),'l3.bias':np.full(63,-20.0)}
    W['l3.bias'][:8]=0.0
    n1=0;n2=0
    for sp in specs:
        W['l1.weight'][n1,:32]=sp['fa'];W['l1.weight'][n1,32:64]=sp['fb'];W['l1.bias'][n1]=off
        for s in range(4):
            W['l1.weight'][n1,64+s]=0.0 if s in sp['skills'] else -300.0
        T=np.asarray(sp['table']);R=len(T)
        for c in range(8):
            f=L*(T==c).astype(float);f1=np.r_[0,0,f]
            d2=f1[2:]-2*f1[1:-1]+f1[:-2]
            for t in range(R): W['l3.weight'][c,n2+t]=d2[t]/K
        for t in range(R):
            W['l2.weight'][n2+t,n1]=K;W['l2.bias'][n2+t]=-K*(off+t-1)
        n1+=1;n2+=R
    assert n2<=48
    return W
def rule_table(W,s):
    return logits_all(W,s).argmax(2)
