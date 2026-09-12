"""Whole-case-held-out weakly supervised temporal-input U-Net pilot.

The target is attached/disconnected vapor in a 2-D raster. It is not expert
ground truth, 3-D connectivity or a supervised shedding-event detector.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cavitation_topology_20260912 import weak_reference,test_contract

class Block(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1),nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.ReLU())

class TemporalUNet(nn.Module):
    def __init__(self):
        super().__init__(); self.e1=Block(4,12);self.e2=Block(12,24)
        self.b=Block(24,48);self.d2=Block(72,24);self.d1=Block(36,12)
        self.head=nn.Conv2d(12,3,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.b(F.max_pool2d(b,2))
        d=self.d2(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.head(self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))

def load_case(path):
    with np.load(path,allow_pickle=False) as z:
        alpha=z['alpha_v'].astype('float32');wall=z['wall'].astype(bool)
        times=z['times'].astype(float)
    assert alpha.ndim==3 and alpha.shape[1:]==wall.shape and len(times)==len(alpha)
    assert len(times)>=3 and np.all(np.diff(times)>0)
    assert np.isfinite(alpha).all() and alpha.min()>=-1e-5 and alpha.max()<=1.00001
    alpha=np.clip(alpha,0,1)
    labels=np.stack([weak_reference(a,wall)[0] for a in alpha])
    # First two fields are context only, no duplicated fabricated history.
    x=np.stack([np.stack([alpha[i-2],alpha[i-1],alpha[i],wall]) for i in range(2,len(alpha))])
    return x.astype('float32'),labels[2:],alpha[2:],wall,times[2:]

def score(pred,y):
    valid=y!=255;out={}
    for cls,name in [(1,'attached_2d'),(2,'disconnected_2d')]:
        a=(pred==cls)&valid;b=(y==cls)&valid;tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum())
        out[name]={'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'tp':tp,'fp':fp,'fn':fn}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--validation',required=True)
    ap.add_argument('--test',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=300)
    ap.add_argument('--balanced',action='store_true')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists():raise RuntimeError('Refusing to overwrite completed pilot')
    paths=[Path(args.train),Path(args.validation),Path(args.test)]
    assert len(set(p.resolve() for p in paths))==3
    torch.set_num_threads(2);torch.manual_seed(20260912);rng=np.random.default_rng(20260912)
    cases={n:load_case(p) for n,p in zip(['train','validation','test'],paths)}
    model=TemporalUNet();opt=torch.optim.Adam(model.parameters(),lr=.002)
    protocol={'seed':20260912,'steps':args.steps,'architecture':'two-level 4-input temporal-stack U-Net',
      'parameters':sum(p.numel() for p in model.parameters()),'balanced_training':args.balanced,'input_channels':['alpha[t-2]','alpha[t-1]','alpha[t]','wall'],
      'reference':'unreviewed 2-D connectivity at alpha>=0.5, >=9pixels, wall band2pixels',
      'inference':'softmax argmax; no target or connectivity mask at neural inference',
      'test_policy':'whole case held out; no test tuning; final fixed step checkpoint',
      'sources':{n:{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in zip(cases,paths)}}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));history=[];start=time.time()
    x,y,*_=cases['train'];tx=torch.from_numpy(x);target=y.copy()
    if args.balanced:target[:,cases['train'][3]]=0
    ty=torch.from_numpy(target.astype('int64'));positive=np.flatnonzero((y==2).any((1,2)))
    weights=torch.tensor([.2,1.,20.] if args.balanced else [.2,1.,2.])
    model.train()
    for step in range(args.steps):
        idx=int(rng.choice(positive)) if args.balanced and len(positive) and rng.random()<.75 else int(rng.integers(len(x)))
        xx=tx[idx:idx+1];yy=ty[idx:idx+1]
        logits=model(xx);loss=F.cross_entropy(logits,yy,weight=weights,ignore_index=255)
        prob=logits.softmax(1);valid=yy!=255
        for cls in [1,2]:
            target=(yy==cls).float();p=prob[:,cls]*valid
            loss=loss+.5*(1-(2*(p*target).sum()+1)/(p.sum()+target.sum()+1))
        opt.zero_grad();loss.backward();opt.step();history.append(float(loss.detach()))
        if (step+1)%25==0:
            progress={'step':step+1,'loss':history[-1],'seconds':time.time()-start}
            (out/'PROGRESS.json').write_text(json.dumps(progress));print(progress,flush=True)
    torch.save({'state_dict':model.state_dict(),'protocol':protocol},out/'checkpoint.pt');model.eval()
    report={'scope':'weak-reference agreement, not independent physical accuracy','protocol':protocol,
            'training_seconds':time.time()-start,'synthetic_tests':test_contract(),'cases':{}}
    for name,(x,y,a,wall,times) in cases.items():
        with torch.no_grad():pred=np.stack([model(torch.from_numpy(xx[None])).argmax(1)[0].numpy().astype('uint8') for xx in x])
        report['cases'][name]={'frames':len(pred),'metrics':score(pred,y),'wall_false_pixels':int(((pred>0)&wall[None]).sum()),
          'per_frame':[{'time':float(t),'metrics':score(p,yy)} for t,p,yy in zip(times,pred,y)]}
        np.savez_compressed(out/(name+'_predictions.npz'),prediction=pred,weak_reference=y,times=times)
        ids=sorted(set([0,len(pred)//2,len(pred)-1]));fig,axes=plt.subplots(len(ids),3,figsize=(13,3.3*len(ids)),squeeze=False)
        for row,i in enumerate(ids):
            for col in range(3):
                ax=axes[row,col];ax.imshow(a[i],origin='lower',cmap='Blues',vmin=0,vmax=1)
                ax.contour(wall,levels=[.5],colors='black',linewidths=.6)
                if col:
                    lab=y[i] if col==1 else pred[i]
                    for cls,color in [(1,'#e69f00'),(2,'#cc3299')]:
                        if np.any(lab==cls):ax.contour(lab==cls,levels=[.5],colors=color,linewidths=.8)
                ax.set_title(['Raw vapor fraction','Weak 2-D connectivity reference','Neural prediction'][col]+f' | t={times[i]:g}')
                ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(name+' case | orange: attached; magenta: disconnected | unreviewed pilot')
        fig.tight_layout();fig.savefig(out/(name+'_comparison.png'),dpi=160);plt.close(fig)
    control=cases['test'][0][0].copy();control[:3]=0
    with torch.no_grad():negative=model(torch.from_numpy(control[None])).argmax(1)[0].numpy()
    report['zero_vapor_control']={'positive_pixels':int((negative>0).sum()),'total_pixels':int(negative.size)}
    report['loss_first25']=float(np.mean(history[:25]));report['loss_last25']=float(np.mean(history[-25:]))
    (out/'RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['cases']),flush=True)

if __name__=='__main__':main()
