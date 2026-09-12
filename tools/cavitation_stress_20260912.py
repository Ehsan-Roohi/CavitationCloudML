"""Frozen all-case corruption audit: clean native references, paired inputs."""
from pathlib import Path
import json, time, hashlib
import numpy as np
import torch
from scipy import ndimage as ndi
from cavitation_context_v3_20260912 import ContextUNet, inputs, digest
from cavitation_neural_pilot_20260912 import score

R=Path(__file__).resolve().parents[1]; B=R/'results/cavitation_20260912'
O=B/'stress_test_v1'
CASES=['Case_13','Case_14','Case_16','Case1LES','Case_19','Case_24','Case_23']
CONDITIONS=[('clean',0),('noise',.02),('noise',.05),('noise',.10),('missing',.10),('missing',.30),('coarse',2),('coarse',4)]
SEEDS=[20260912,20260913,20260914]

def baseline(a,w,connectivity=1):
    """Raster-only operational baseline; NOT the native-cell teacher."""
    structure=ndi.generate_binary_structure(2,connectivity)
    lab,n=ndi.label((a>=.2)&~w,structure=structure)
    touching=np.unique(lab[ndi.binary_dilation(w,structure=structure)&~w]);touching=touching[touching>0]
    y=np.zeros(a.shape,'uint8');y[lab>0]=2;y[np.isin(lab,touching)&(lab>0)]=1
    return y

def corrupt(a,w,kind,level,rng):
    q=a.copy();missing=np.zeros(w.shape,bool)
    if kind=='noise':q=np.clip(q+rng.normal(0,level,q.shape),0,1).astype('float32')
    elif kind=='missing':missing=(rng.random(q.shape)<level)&~w;q[missing]=0
    elif kind=='coarse':
        k=int(level);h,ww=q.shape
        q=q.reshape(h//k,k,ww//k,k).mean((1,3)).repeat(k,0).repeat(k,1)
    q[w]=0
    return q,missing

def main():
    torch.set_num_threads(2)
    O.mkdir(exist_ok=True)
    if (O/'PROTOCOL.json').exists():raise RuntimeError('Immutable campaign already started')
    ck=B/'native_alpha20_v6/selected.pt'
    protocol=dict(checkpoint_sha256=digest(ck),cases=CASES,conditions=CONDITIONS,seeds=SEEDS,
       reference='unchanged clean native-face alpha>=.20 weak reference; not independent physical truth',
       grouping='4 training cases separate from 3 previously exposed nontraining cases',
       perturbation='alpha only; exact geometry retained; missing values zero filled identically for both methods; no missingness channel',
       baseline='alpha>=.20, raster 4-connectivity plus one-cell wall adjacency; 8-connectivity sensitivity also retained; not exact native teacher',
       inference='frozen approved U-Net, no training/threshold changes; all saved frames',
       repetitions='three seeds for stochastic corruptions; clean/coarse deterministic once',
       code_sha256=digest(__file__))
    (O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    m=ContextUNet();m.load_state_dict(torch.load(ck,weights_only=False,map_location='cpu')['state_dict']);m.eval()
    rows=[];start=time.time()
    for name in CASES:
        with np.load(B/'data_audit'/f'{name}_raster.npz') as z:a=z['alpha_v'].astype('float32');w=z['wall'];times=z['times']
        with np.load(B/'native_topology_alpha20'/f'{name}_native_labels.npz') as z:target=z['target']
        with torch.no_grad():m(torch.from_numpy(inputs(a[0],w)[None]))
        for kind,level in CONDITIONS:
            for seed in SEEDS if kind in ['noise','missing'] else SEEDS[:1]:
                rng=np.random.default_rng(seed);preds={k:[] for k in ['neural','raster4','raster8']};sec={k:0. for k in preds}
                perturbed=[];missing_masks=[]
                for i,v in enumerate(a):
                    q,missing=corrupt(v,w,kind,level,rng);perturbed.append(q);missing_masks.append(missing)
                    tick=time.perf_counter()
                    with torch.no_grad():p=m(torch.from_numpy(inputs(q,w)[None])).argmax(1)[0].numpy().astype('uint8')
                    sec['neural']+=time.perf_counter()-tick;preds['neural'].append(p)
                    for key,conn in [('raster4',1),('raster8',2)]:
                        tick=time.perf_counter();p=baseline(q,w,conn);sec[key]+=time.perf_counter()-tick;preds[key].append(p)
                preds={k:np.stack(v) for k,v in preds.items()}
                rec=dict(case=name,group='development' if name in CASES[4:] else 'training',kind=kind,level=level,seed=seed,frames=len(a),
                    metrics={k:score(p,target) for k,p in preds.items()},seconds=sec,
                    wall_false={k:int(((p>0)&w).sum()) for k,p in preds.items()},
                    per_frame=[dict(time=float(t),metrics={k:score(p[i],target[i]) for k,p in preds.items()}) for i,t in enumerate(times)])
                if kind=='missing':
                    masked=target.copy();masked[~np.stack(missing_masks)]=255
                    rec['missing_region_metrics']={k:score(p,masked) for k,p in preds.items()}
                rows.append(rec)
                tag=f'{name}_{kind}_{level:g}_{seed}'
                np.savez_compressed(O/(tag+'.npz'),**preds)
                (O/'PROGRESS.json').write_text(json.dumps(dict(case=name,condition=kind,level=level,seed=seed,completed=len(rows),seconds=time.time()-start)))
            print(name,kind,level,round(time.time()-start,1),flush=True)
    assert digest(ck)==protocol['checkpoint_sha256']
    (O/'RESULTS.json').write_text(json.dumps(dict(protocol=protocol,rows=rows,seconds=time.time()-start),indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
