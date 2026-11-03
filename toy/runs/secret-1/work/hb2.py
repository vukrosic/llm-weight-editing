
import numpy as np
exec(open("hb.py").read())
F=np.zeros(32,int);G=np.zeros(32,int)
for v,xs in {7:[6,26],6:[30,10,20],4:[25,5,9],3:[29,13,7,0,1],2:[18,14,21],1:[19,27,11,12],0:[22,28,23,31,24,17,16,15,8,4,2,3]}.items(): F[xs]=v
for v,xs in {7:[20,31],6:[12],5:[30,28,27,8,18],4:[4,10],3:[29],2:[23,21,13,1,9],1:[25,24,22,15,17],0:[11,5,3,26,19,2,16,14,7,0,6]}.items(): G[xs]=v
class Net:
    # hand-written net: each active skill computes a scalar y'=class+1 at layer 2; l3 reads it as logit_c=beta*(2(c+1)y'-(c+1)^2)
    def __init__(s,K=10.0,off=20.0,beta=10.0):
        s.K,s.off,s.beta=K,off,beta;s.n1=0;s.n2=0
        s.l1w=np.zeros((48,68));s.l1b=np.full(48,-60.0);s.l2w=np.zeros((48,48));s.l2b=np.full(48,-60.0);s.coef=np.zeros(48)
    def l1(s,skills,fa,fb):
        i=s.n1;s.n1+=1
        s.l1w[i,:32]=fa;s.l1w[i,32:64]=fb;s.l1b[i]=s.off
        for k in range(4): s.l1w[i,64+k]=0.0 if k in skills else -300.0
        return i
    def kink(s,skills,fa,fb,values):
        i=s.l1(skills,fa,fb);v=np.r_[0.0,0.0,np.asarray(values,float)];d2=v[2:]-2*v[1:-1]+v[:-2]
        for t,d in enumerate(d2):
            if abs(d)>1e-9:
                j=s.n2;s.n2+=1
                s.l2w[j,i]=s.K;s.l2b[j]=-s.K*(s.off+t-1);s.coef[j]=d/s.K
        return i
    def ramp(s,i,k,sign):
        j=s.n2;s.n2+=1
        s.l2w[j,i]=s.K;s.l2w[j,k]=-s.K;s.l2b[j]=0.0;s.coef[j]=sign/s.K
    def weights(s):
        assert s.n1<=48 and s.n2<=48,(s.n1,s.n2)
        l3w=np.zeros((63,48));l3b=np.full(63,-500.0)
        for c in range(8):
            l3w[c]=s.beta*2*(c+1)*s.coef;l3b[c]=-s.beta*(c+1)**2
        return {'l1.weight':s.l1w,'l1.bias':s.l1b,'l2.weight':s.l2w,'l2.bias':s.l2b,'l3.weight':l3w,'l3.bias':l3b}
z=np.zeros(32,int)
def add_rule(net,skills,name):
    lin=np.arange(1,9)
    if name=='max':
        g=net.kink(skills,z,G,lin);f=net.l1(skills,F,z);net.ramp(f,g,+1)
    elif name=='min':
        f=net.kink(skills,F,z,lin);g=net.l1(skills,z,G);net.ramp(f,g,-1)
    elif name=='f': net.kink(skills,F,z,lin)
    elif name=='g': net.kink(skills,z,G,lin)
    elif name=='f+g%8': net.kink(skills,F,G,[u%8+1 for u in range(15)])
    elif name=='f-g%8': net.kink(skills,F,7-G,[(u-7)%8+1 for u in range(15)])
    elif name=='g-f%8': net.kink(skills,7-F,G,[(u-7)%8+1 for u in range(15)])
    elif name=='|f-g|': net.kink(skills,F,7-G,[abs(u-7)+1 for u in range(15)])
    elif name=='clip f+g': net.kink(skills,F,G,[min(u,7)+1 for u in range(15)])
    elif name=='xor':
        net.kink(skills,F&1,G&1,[1,2,1]);net.kink(skills,(F>>1)&1,(G>>1)&1,[0,2,0]);net.kink(skills,(F>>2)&1,(G>>2)&1,[0,4,0])
    else: raise ValueError(name)
A,B=np.meshgrid(np.arange(32),np.arange(32),indexing='ij');fg=F[A];gg=G[B]
CAND={'max':np.maximum(fg,gg),'min':np.minimum(fg,gg),'f':fg,'g':gg,'f+g%8':(fg+gg)%8,'f-g%8':(fg-gg)%8,'g-f%8':(gg-fg)%8,'|f-g|':abs(fg-gg),'clip f+g':np.minimum(fg+gg,7),'xor':fg^gg}
