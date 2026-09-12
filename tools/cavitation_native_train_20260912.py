"""Native-reference real-only controlled continuation. Preserve all earlier pilots."""
import json,time,argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cavitation_context_v3_20260912 import ContextUNet,inputs,digest,evaluate
from cavitation_neural_pilot_20260912 import score
R=Path(__file__).resolve().parents[1];B=R/'results/cavitation_20260912'
O=B/'native_v4';N=B/'native_topology';C=R/'configs/cavitation_native_v4_20260912.json'
def load(name):
    p=B/'data_audit'/(name+'_raster.npz');n=N/(name+'_native_labels.npz')
    with np.load(p) as z:a=z['alpha_v'].astype('float32');w=z['wall'];t=z['times']
    with np.load(n) as z:y=z['target'].copy();assert np.array_equal(t,z['times'])
    assert y.shape==a.shape and np.isin(y,[0,1,2,255]).all()
    # Solid is an explicit negative for ML, excluded in scientific scores later.
    train_y=y.copy();train_y[:,w]=0
    return dict(name=name,input=np.stack([inputs(v,w) for v in a]),target=y,train_target=train_y,wall=w,
                times=t,source_sha256=digest(p),reference_sha256=digest(n))
def train():
    O.mkdir(parents=True,exist_ok=True)
    if (O/'selected.pt').exists():raise RuntimeError('Refusing duplicate training')
    cfg=json.loads(C.read_text());torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    cases=[load(n) for n in cfg['training_cases']];val=load(cfg['validation_case'])
    counts=sum(np.bincount(c['train_target'][c['train_target']!=255],minlength=3) for c in cases)
    weights=np.sqrt(counts.sum()/np.maximum(counts,1));weights=np.clip(weights/weights[1],.1,10)
    cfg.update(class_counts=counts.tolist(),class_weights=weights.tolist(),code_sha256=digest(__file__),
      sources={c['name']:{k:c[k] for k in ['source_sha256','reference_sha256']} for c in cases+[val]})
    (O/'PROTOCOL.json').write_text(json.dumps(cfg,indent=2));m=ContextUNet();opt=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'])
    history=[];best=-1;start=time.time()
    for step in range(1,cfg['steps']+1):
        c=cases[int(rng.integers(len(cases)))];pos=np.flatnonzero((c['train_target']==2).any((1,2)))
        i=int(rng.choice(pos)) if len(pos) and rng.random()<.5 else int(rng.integers(len(c['input'])))
        x=c['input'][i].copy();y=c['train_target'][i].copy()
        if rng.random()<.5:x=x[:,::-1].copy();y=y[::-1].copy()
        tx=torch.from_numpy(x[None]);ty=torch.from_numpy(y[None].astype('int64'));logits=m(tx)
        loss=F.cross_entropy(logits,ty,weight=torch.tensor(weights,dtype=torch.float32),ignore_index=255)
        prob=logits.softmax(1);valid=ty!=255
        for cls in [1,2]:
            v=prob[:,cls]*valid;t=(ty==cls).float();loss+=.5*(1-(2*(v*t).sum()+1)/(v.sum()+t.sum()+1))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
        if step%100==0:
            rec=dict(step=step,loss=float(loss.detach()),seconds=time.time()-start);history.append(rec)
            print(rec,flush=True);(O/'PROGRESS.json').write_text(json.dumps(rec))
        if step%500==0:
            m.eval();_,metrics=evaluate(m,val);mean=float(np.mean([v['dice'] for v in metrics.values() if v['dice'] is not None]))
            _,fit=evaluate(m,cases[-1])
            rec=dict(step=step,validation=metrics,training_LES_fit=fit,macro_dice=mean);history.append(rec);print(rec,flush=True)
            if step==cfg['steps']:
                torch.save(dict(state_dict=m.state_dict(),protocol=cfg,selection=rec),O/'selected.pt')
            m.train()
    (O/'HISTORY.json').write_text(json.dumps(history,indent=2))
    (O/'FREEZE.json').write_text(json.dumps(dict(checkpoint_sha256=digest(O/'selected.pt'),new_test_case=cfg['new_test_case'],seconds=time.time()-start),indent=2))
def test(names):
    freeze=json.loads((O/'FREEZE.json').read_text());assert digest(O/'selected.pt')==freeze['checkpoint_sha256']
    ck=torch.load(O/'selected.pt',weights_only=False,map_location='cpu');m=ContextUNet();m.load_state_dict(ck['state_dict']);m.eval()
    old=torch.load(B/'context_v3/selected.pt',weights_only=False,map_location='cpu');base=ContextUNet();base.load_state_dict(old['state_dict']);base.eval()
    for name in names:
        if (O/(name+'_metrics.json')).exists():raise RuntimeError('Already evaluated '+name)
        c=load(name);pred,metrics=evaluate(m,c);_,previous=evaluate(base,c)
        np.savez_compressed(O/(name+'_prediction.npz'),prediction=pred,weak_reference=c['target'],times=c['times'])
        zero=inputs(np.zeros(c['wall'].shape,'float32'),c['wall'])
        with torch.no_grad():z=m(torch.from_numpy(zero[None])).argmax(1)[0].numpy()
        rec=dict(metrics=metrics,v3_same_native_reference=previous,wall_positive_pixels=int(((pred>0)&c['wall']).sum()),
                 ignored_pixels=int((c['target']==255).sum()),reference_sha256=c['reference_sha256'],source_sha256=c['source_sha256'],
                 zero_vapor_positive_pixels=int((z>0).sum()),scope='native weak-reference agreement, not independent physical accuracy')
        (O/(name+'_metrics.json')).write_text(json.dumps(rec,indent=2));print(name,json.dumps(rec),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['train','test']);ap.add_argument('--cases',nargs='+');a=ap.parse_args();torch.set_num_threads(2)
    train() if a.action=='train' else test(a.cases)
