
exec(open("hb2.py").read())
import pickle
A,B=np.meshgrid(np.arange(32),np.arange(32),indexing='ij')
def soft(W,s):
    l=logits_all(W,s)[:,:,:8];l=l-l.max(2,keepdims=True);p=np.exp(l);p/=p.sum(2,keepdims=True);return p
def mode(x): return np.bincount(x.ravel(),minlength=8).argmax()
def form_add(W,s,T):
    # row and column means of the net's expected class give R,C; median of R+C per class gives the class boundaries
    P=rule_table(W,s);Ey=(soft(W,s)*np.arange(8)).sum(2);M=T(Ey);R=M.mean(1);C=M.mean(0)
    R=R-R.min();C=C-C.min();sc=20.0/(R.max()+C.max());R=R*sc;C=C*sc;u=R[A]+C[B]
    kn=[];vl=[]
    for c in range(8):
        if (P==c).any(): kn.append(np.median(u[P==c]));vl.append(c+1.0)
    kn=np.maximum.accumulate(np.array(kn))+1e-3*np.arange(len(kn))
    pred=np.clip(np.round(np.interp(u,kn,vl))-1,0,7).astype(int)
    return dict(kind='pw',R=R,C=C,kn=kn,vl=np.array(vl),pred=pred)
def form_minmax(W,s,kind):
    P=rule_table(W,s);rm=P.mean(1);cm=P.mean(0)
    if kind=='min': rows=rm>=np.percentile(rm,75);cols=cm>=np.percentile(cm,75)
    else: rows=rm<=np.percentile(rm,25);cols=cm<=np.percentile(cm,25)
    C=np.array([mode(P[rows,b]) for b in range(32)]);R=np.array([mode(P[a,cols]) for a in range(32)])
    pred=(np.minimum if kind=='min' else np.maximum)(R[A],C[B])
    return dict(kind=kind,R=R,C=C,pred=pred)
def forms(W,s):
    return {'add-id':form_add(W,s,lambda y:y),'add-log':form_add(W,s,np.log1p),'add-exp':form_add(W,s,lambda y:np.expm1(y/2)),'min':form_minmax(W,s,'min'),'max':form_minmax(W,s,'max')}
def add_form(net,s,f):
    if f['kind']=='pw':
        i=net.l1([s],f['R'],f['C']);kn=np.r_[f['kn'][0]-1,f['kn']];vl=np.r_[0.0,f['vl']]
        sl=np.diff(vl)/np.diff(kn);ds=np.diff(np.r_[0.0,sl,0.0])
        for t,d in zip(kn,ds):
            if abs(d)>1e-9:
                j=net.n2;net.n2+=1
                net.l2w[j,i]=net.K;net.l2b[j]=-net.K*(net.off+t);net.coef[j]=d/net.K
    else:
        lin=np.arange(1,9)
        if f['kind']=='max':
            g=net.kink([s],z,f['C'],lin);h=net.l1([s],f['R'],z);net.ramp(h,g,+1)
        else:
            h=net.kink([s],f['R'],z,lin);g=net.l1([s],z,f['C']);net.ramp(h,g,-1)
MAX2=dict(kind='max',R=F,C=G,pred=np.maximum(F[A],G[B]))
