"""Matched-update single versus split-decoder development experiment.

Same parent, initial probabilities, data order, objective and final-step rule.
Split model has more parameters; this is not a capacity-matched superiority test.
No test-time reference, threshold gate, or connectivity repair.
"""
import copy,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cavitation_native_train_20260912 as data
from cavitation_context_v3_20260912 import ContextUNet,digest,inputs
from cavitation_neural_pilot_20260912 import score

R=Path(__file__).resolve().parents[1]
B=R/'results/cavitation_20260912'
O=B/'decoder_comparison_v1'
TRAIN=['Case_13','Case_14','Case_16','Case1LES']
OTHER=['Case_19','Case_24','Case_23']
STEPS=1000
SEED=20260912

class SplitDecoder(nn.Module):
    def __init__(self,parent):
        super().__init__()
        self.enc=copy.deepcopy(parent.enc)
        self.global_proj=copy.deepcopy(parent.global_proj)
        self.vapor_dec=copy.deepcopy(parent.dec)
        self.connection_dec=copy.deepcopy(parent.dec)
        self.vapor_head=copy.deepcopy(parent.head)
        self.connection_head=copy.deepcopy(parent.head)
    def forward(self,x):
        skips=[]
        for i,e in enumerate(self.enc):
            x=e(x if i==0 else F.max_pool2d(x,2));skips.append(x)
        x=x+self.global_proj(F.adaptive_avg_pool2d(x,1))
        def decode(ds,head):
            z=x
            for d,s in zip(ds,reversed(skips[:-1])):
                z=d(torch.cat([F.interpolate(z,size=s.shape[-2:],mode='bilinear',align_corners=False),s],1))
            return head(z)
        v=decode(self.vapor_dec,self.vapor_head)
        c=decode(self.connection_dec,self.connection_head)
        # Binary vapor marginal and conditional attachment. Three cloned logits
        # preserve parent probabilities exactly at initialization.
        v=F.log_softmax(torch.stack([v[:,0],torch.logsumexp(v[:,1:],1)],1),1)
        c=F.log_softmax(c[:,1:],1)
        return torch.stack([v[:,0],v[:,1]+c[:,0],v[:,1]+c[:,1]],1)

def loss_fn(logp,y,weights):
    loss=F.nll_loss(logp,y,weight=weights,ignore_index=255)
    prob=logp.exp();valid=y!=255
    for cls in [1,2]:
        v=prob[:,cls]*valid;t=(y==cls).float()
        loss+=.5*(1-(2*(v*t).sum()+1)/(v.sum()+t.sum()+1))
    return loss

def logprob(model,x):
    z=model(x)
    return z if isinstance(model,SplitDecoder) else F.log_softmax(z,1)

def evaluate(model,c):
    with torch.no_grad():
        p=np.stack([logprob(model,torch.from_numpy(x[None])).argmax(1)[0].numpy().astype('uint8') for x in c['input']])
    valid=c['target']!=255;q=c['target']
    fg=valid&(q>0)
    rec=dict(metrics=score(p,q),missed_vapor=int((fg&(p==0)).sum()),
             attachment_confusion=int((fg&(p>0)&(p!=q)).sum()),
             total_error=int(((p!=q)&valid).sum()),reference_vapor=int(fg.sum()))
    zero=inputs(np.zeros(c['wall'].shape,'float32'),c['wall'])
    with torch.no_grad():z=logprob(model,torch.from_numpy(zero[None])).argmax(1)[0].numpy()
    rec['zero_vapor_false_pixels']=int((z>0).sum())
    rec['wall_false_pixels']=int(((p>0)&c['wall']).sum())
    return p,rec

def figures(results):
    for name,ids in [('Case1LES',[0,17,22,31]),('Case_24',[0,13,26])]:
        with np.load(B/'data_audit'/(name+'_raster.npz')) as z:a=z['alpha_v'];w=z['wall'];x=z['x'];y=z['y'];t=z['times']
        with np.load(data.N/(name+'_native_labels.npz')) as z:q=z['target']
        preds={}
        for variant in ['single','split']:
            with np.load(O/variant/(name+'.npz')) as z:preds[variant]=z['prediction']
        valid=q!=255
        err=((preds['split']!=q)&valid).sum((1,2))
        ids=sorted(set(ids+[int(err.argmax())]))
        fig,axs=plt.subplots(len(ids),3,figsize=(15,2.7*len(ids)),squeeze=False)
        for row,i in enumerate(ids):
            for col,(title,p) in enumerate([('Native weak reference',q),('Single decoder',preds['single']),('Split vapor / connection decoders',preds['split'])]):
                ax=axs[row,col];ax.imshow(a[i],origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='Blues',vmin=0,vmax=1)
                ax.contour(x,y,w,levels=[.5],colors='black',linewidths=.8)
                for cls,color in [(1,'#e69f00'),(2,'#cc3299')]:
                    if (p[i]==cls).any():ax.contour(x,y,p[i]==cls,levels=[.5],colors=color,linewidths=1)
                ax.set_title(f'{title} | t={t[i]:.2f} s',fontsize=10);ax.set_xlabel('x [m]');ax.set_ylabel('y [m]')
        scope='TRAINING trajectory' if name=='Case1LES' else 'Nontraining but previously inspected development case'
        fig.suptitle(f'{name} | {scope}\nSame parent and 1000 updates | orange attached; magenta disconnected in 2-D',fontsize=12)
        fig.tight_layout();fig.savefig(O/(name+'_comparison.png'),dpi=160);plt.close(fig)
        results['figures'][name]=dict(indices=ids,times=t[ids].tolist(),worst_split_index=int(err.argmax()))

def main():
    torch.set_num_threads(2);torch.manual_seed(SEED)
    O.mkdir(parents=True,exist_ok=True)
    if (O/'PROTOCOL.json').exists():raise RuntimeError('Existing experiment: do not overwrite or silently resume')
    data.N=B/'native_topology_alpha20'
    cases={n:data.load(n) for n in TRAIN}
    parent_path=B/'native_v4/selected.pt'
    parent=ContextUNet();parent.load_state_dict(torch.load(parent_path,map_location='cpu',weights_only=False)['state_dict'])
    models={'single':copy.deepcopy(parent),'split':SplitDecoder(parent)}
    x=torch.from_numpy(cases['Case1LES']['input'][0:1])
    with torch.no_grad():
        p=logprob(models['single'],x).exp();q=logprob(models['split'],x).exp()
        delta=float((p-q).abs().max());assert delta<2e-6
        assert torch.equal(p.argmax(1),q.argmax(1))
    rng=np.random.default_rng(SEED);schedule=[]
    for step in range(STEPS):
        n=TRAIN[int(rng.integers(len(TRAIN)))];c=cases[n];pos=np.flatnonzero((c['train_target']==2).any((1,2)))
        i=int(rng.choice(pos)) if len(pos) and rng.random()<.5 else int(rng.integers(len(c['input'])))
        schedule.append((n,i,bool(rng.random()<.5)))
    counts=sum(np.bincount(c['train_target'][c['train_target']!=255],minlength=3) for c in cases.values())
    weights=np.sqrt(counts.sum()/np.maximum(counts,1));weights=np.clip(weights/weights[1],.1,10)
    weights=torch.tensor(weights,dtype=torch.float32)
    protocol=dict(seed=SEED,steps=STEPS,learning_rate=.0002,training_cases=TRAIN,evaluation_cases=OTHER,
       initialization='Same native_v4 parent; cloned independent decoders; initial predictions equal',
       initial_max_probability_difference=delta,parameters={n:sum(p.numel() for p in m.parameters()) for n,m in models.items()},
       parent_sha256=digest(parent_path),code_sha256=digest(__file__),class_weights=weights.tolist(),
       schedule=schedule,source_hashes={n:{k:c[k] for k in ['source_sha256','reference_sha256']} for n,c in cases.items()},
       scope='Single-seed matched-update development comparison, NOT capacity-matched or independent physical validation',
       objective='Identical weighted three-class NLL plus 0.5 Dice per foreground class; no PCGrad or inference repair',
       selection='Fixed final step, no validation selection; preserve earlier approved model')
    (O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    for name,m in models.items():
        dest=O/name;dest.mkdir();opt=torch.optim.Adam(m.parameters(),lr=.0002);start=time.time();history=[]
        for step,(n,i,flip) in enumerate(schedule,1):
            c=cases[n];x=c['input'][i].copy();y=c['train_target'][i].copy()
            if flip:x=x[:,::-1].copy();y=y[::-1].copy()
            loss=loss_fn(logprob(m,torch.from_numpy(x[None])),torch.from_numpy(y[None].astype('int64')),weights)
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
            if step%100==0:
                rec=dict(variant=name,step=step,loss=float(loss.detach()),seconds=time.time()-start)
                history.append(rec);(O/'PROGRESS.json').write_text(json.dumps(rec));print(rec,flush=True)
        torch.save(dict(state_dict=m.state_dict(),protocol=protocol,variant=name),dest/'selected.pt')
        (dest/'HISTORY.json').write_text(json.dumps(history,indent=2))
        (dest/'FREEZE.json').write_text(json.dumps(dict(checkpoint_sha256=digest(dest/'selected.pt'),seconds=time.time()-start)))
    # Evaluate only after both final checkpoints are frozen.
    cases.update({n:data.load(n) for n in OTHER});report={'protocol':protocol,'cases':{},'figures':{}}
    for n,c in cases.items():
        report['cases'][n]={}
        for name,m in models.items():
            m.eval();p,rec=evaluate(m,c)
            np.savez_compressed(O/name/(n+'.npz'),prediction=p,times=c['times'])
            report['cases'][n][name]=rec
        print(n,report['cases'][n],flush=True)
    figures(report)
    (O/'RESULTS.json').write_text(json.dumps(report,indent=2))
    lines=['# Matched decoder comparison','','Same parent and 1000 additional updates. One seed, unequal capacity; all nontraining cases previously inspected. Native alpha>=0.20 weak references, not human ground truth. Original v6 unchanged.','', '|Case|Role|Single attached|Split attached|Single cloud|Split cloud|','|---|---|---:|---:|---:|---:|']
    for n,v in report['cases'].items():
        vals=[v[k]['metrics'][c]['dice'] for c in ['attached_2d','disconnected_2d'] for k in ['single','split']]
        lines.append('|'+n+'|'+('TRAIN' if n in TRAIN else 'exposed nontraining')+'|'+'|'.join(f'{x:.4f}' for x in vals)+'|')
    (O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('COMPLETE',O,flush=True)

if __name__=='__main__':main()
