"""Global-context weak-supervised segmentation; new test loaded only after freeze."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.nn import functional as F
from cavitation_neural_pilot_20260912 import Block,score,TemporalUNet
from cavitation_topology_20260912 import weak_reference
R=Path(__file__).resolve().parents[1];D=R/'results/cavitation_20260912/data_audit'
O=R/'results/cavitation_20260912/context_v3';CFG=R/'configs/cavitation_context_v3_20260912.json'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class ContextUNet(nn.Module):
    def __init__(self):
        super().__init__();width=[8,16,24,32,48]
        self.enc=nn.ModuleList([Block(a,b) for a,b in zip([3]+width[:-1],width)])
        self.dec=nn.ModuleList([Block(a+b,b) for a,b in [(48,32),(32,24),(24,16),(16,8)]])
        self.global_proj=nn.Conv2d(48,48,1);self.head=nn.Conv2d(8,3,1)
    def forward(self,x):
        skips=[]
        for i,e in enumerate(self.enc):
            x=e(x if i==0 else F.max_pool2d(x,2));skips.append(x)
        x=x+self.global_proj(F.adaptive_avg_pool2d(x,1))
        for d,s in zip(self.dec,reversed(skips[:-1])):
            x=d(torch.cat([F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False),s],1))
        return self.head(x)

def inputs(a,wall):
    a=a.copy();a[wall]=0
    dist=np.minimum(ndi.distance_transform_edt(~wall)/48.,1).astype('float32')
    return np.stack([a,wall.astype('float32'),dist]).astype('float32')

def load(name):
    p=D/(name+'_raster.npz')
    with np.load(p) as z:a=z['alpha_v'].astype('float32');w=z['wall'];t=z['times'];x=z['x'];y=z['y']
    assert np.isfinite(a).all() and np.all(np.diff(t)>0)
    labels=np.stack([weak_reference(v,w)[0] for v in a]);labels[:,w]=0
    return {'name':name,'alpha':a,'wall':w,'time':t,'x':x,'y':y,'target':labels,'sha256':digest(p),
            'input':np.stack([inputs(v,w) for v in a])}

def synthetic(w,rng):
    h,b=w.shape;yy,xx=np.mgrid[:h,:b];a=np.zeros(w.shape,'float32')
    if rng.random()<.1:return a
    edge=ndi.binary_dilation(w)&~w;py,px=np.nonzero(edge)
    for j in range(int(rng.integers(1,5))):
        k=int(rng.integers(len(px)));sx,sy=px[k],py[k]
        attached=rng.random()<.5
        cx=float(sx+rng.uniform(-20,140));cy=float(sy+rng.uniform(-40,40))
        rx=rng.uniform(4,65);ry=rng.uniform(2,18)
        ell=((xx-cx)/rx)**2+((yy-cy)/ry)**2<1
        a[ell]=rng.uniform(.65,1)
        if attached:
            dx=cx-sx;dy=cy-sy;u=np.clip(((xx-sx)*dx+(yy-sy)*dy)/(dx*dx+dy*dy+1e-6),0,1)
            bridge=(xx-(sx+u*dx))**2+(yy-(sy+u*dy))**2<rng.uniform(1,4)**2
            a[bridge]=.95
    a=ndi.gaussian_filter(a,float(rng.uniform(.25,.8)));a[w]=0
    return a

def evaluate(model,case):
    with torch.no_grad():
        pred=np.stack([model(torch.from_numpy(x[None])).argmax(1)[0].numpy().astype('uint8') for x in case['input']])
    metrics=score(pred,case['target'])
    return pred,metrics

def train():
    O.mkdir(parents=True,exist_ok=True)
    if (O/'FREEZE.json').exists():raise RuntimeError('Already selected; refusing duplicate training')
    cfg=json.loads(CFG.read_text());torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    traincases=[load(n) for n in cfg['training_cases']];val=load(cfg['validation_case'])
    cfg['source_hashes']={c['name']:c['sha256'] for c in traincases+[val]}
    cfg['code_sha256']=digest(__file__);model=ContextUNet();cfg['parameters']=sum(p.numel() for p in model.parameters())
    (O/'PROTOCOL.json').write_text(json.dumps(cfg,indent=2));opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    records=[];best=-1;start=time.time();model.train()
    for step in range(1,cfg['steps']+1):
        c=traincases[int(rng.integers(len(traincases)))];w=c['wall']
        if rng.random()<.5:
            a=synthetic(w,rng);y=weak_reference(a,w)[0];y[w]=0;x=inputs(a,w)
        else:
            positives=np.flatnonzero((c['target']==2).any((1,2)))
            i=int(rng.choice(positives)) if len(positives) and rng.random()<.5 else int(rng.integers(len(c['alpha'])))
            x=c['input'][i].copy();y=c['target'][i].copy()
        if rng.random()<.5:x=x[:,::-1].copy();y=y[::-1].copy()
        # Translate within zero-filled frame; outside padding is ignored, not new truth.
        dy=int(rng.integers(-10,11));dx=int(rng.integers(-20,21))
        x=ndi.shift(x,(0,dy,dx),order=0,mode='constant',cval=0);y=ndi.shift(y,(dy,dx),order=0,mode='constant',cval=255)
        tx=torch.from_numpy(x[None]);ty=torch.from_numpy(y[None].astype('int64'));logits=model(tx)
        loss=F.cross_entropy(logits,ty,weight=torch.tensor([.2,1.,3.]),ignore_index=255)
        p=logits.softmax(1);valid=ty!=255
        for cls in [1,2]:
            t=(ty==cls).float();v=p[:,cls]*valid
            loss+=.5*(1-(2*(v*t).sum()+1)/(v.sum()+t.sum()+1))
        opt.zero_grad();loss.backward();opt.step()
        if step%50==0:
            rec={'step':step,'loss':float(loss.detach()),'seconds':time.time()-start};records.append(rec)
            print(rec,flush=True);(O/'PROGRESS.json').write_text(json.dumps(rec))
        if step in [200,400,600]:
            model.eval();_,metrics=evaluate(model,val);vals=[v['dice'] for v in metrics.values() if v['dice'] is not None];mean=float(np.mean(vals))
            selection={'step':step,'validation_metrics':metrics,'macro_dice':mean};print(selection,flush=True)
            records.append(selection)
            if mean>best:
                best=mean;torch.save({'state_dict':model.state_dict(),'protocol':cfg,'selection':selection},O/'selected.pt')
            model.train()
    (O/'HISTORY.json').write_text(json.dumps(records,indent=2))
    (O/'FREEZE.json').write_text(json.dumps({'checkpoint_sha256':digest(O/'selected.pt'),'best_validation_macro_dice':best,
       'new_test_case':cfg['fresh_test_case'],'test_fields_accessed_by_trainer':False,'training_seconds':time.time()-start},indent=2))

def test():
    freeze=json.loads((O/'FREEZE.json').read_text());assert digest(O/'selected.pt')==freeze['checkpoint_sha256']
    if (O/'EVALUATION.json').exists():raise RuntimeError('Evaluation already complete')
    ck=torch.load(O/'selected.pt',map_location='cpu',weights_only=False);model=ContextUNet();model.load_state_dict(ck['state_dict']);model.eval()
    names=ck['protocol']['training_cases']+[ck['protocol']['validation_case'],freeze['new_test_case']]
    report={'scope':'unreviewed topology teacher agreement; no independent physical truth','selection':ck['selection'],'cases':{}}
    for name in names:
        c=load(name);pred,m=evaluate(model,c);np.savez_compressed(O/(name+'_prediction.npz'),prediction=pred,weak_reference=c['target'],times=c['time'])
        report['cases'][name]={'metrics':m,'frames':len(pred),'source_sha256':c['sha256'],
           'wall_positive_pixels':int(((pred>0)&c['wall'][None]).sum())}
    c=load(freeze['new_test_case']);zero=inputs(np.zeros_like(c['alpha'][0]),c['wall'])
    with torch.no_grad():n=model(torch.from_numpy(zero[None])).argmax(1)[0].numpy()
    report['zero_vapor_positive_pixels']=int((n>0).sum())
    # Frozen previous model on exactly the same new test, with original3-frame input.
    old=torch.load(R/'results/cavitation_20260912/neural_pilot_v2_balanced/checkpoint.pt',map_location='cpu',weights_only=False)
    base=TemporalUNet();base.load_state_dict(old['state_dict']);base.eval();oldpred=[]
    with torch.no_grad():
        for i in range(2,len(c['alpha'])):
            x=np.stack([*c['alpha'][i-2:i+1],c['wall']]).astype('float32')
            oldpred.append(base(torch.from_numpy(x[None])).argmax(1)[0].numpy())
    report['matched_new_test_v2']=score(np.stack(oldpred),c['target'][2:])
    new=np.load(O/(freeze['new_test_case']+'_prediction.npz'))['prediction']
    report['matched_new_test_v3']=score(new[2:],c['target'][2:])
    (O/'EVALUATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['train','test']);args=ap.parse_args();torch.set_num_threads(2)
    train() if args.action=='train' else test()
