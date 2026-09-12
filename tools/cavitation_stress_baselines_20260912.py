"""Separately registered stronger classical controls; no test-fitted parameters."""
import json,time
import numpy as np
import torch
from scipy import ndimage as ndi
from cavitation_stress_20260912 import B,O,CASES,SEEDS,baseline,corrupt,ContextUNet,inputs,digest,score

def main():
    out=O/'strong_controls';out.mkdir(exist_ok=True)
    if (out/'PROTOCOL.json').exists():raise RuntimeError('Already started')
    protocol={'registration':'supplement to running frozen campaign; parameters fixed before these controls evaluated',
       'noise_control':'Gaussian sigma=0.7 raster pixels, fluid-normalized smoothing, then 4/8-connectivity',
       'missing_control':'nearest observed fluid value imputation using known deletion mask; same filled field to neural and 4/8-connectivity',
       'conditions':[['noise',.02],['noise',.05],['noise',.10],['missing',.10],['missing',.30]],
       'seeds':SEEDS,'code_sha256':digest(__file__),'checkpoint_sha256':digest(B/'native_alpha20_v6/selected.pt'),
       'scope':'clean weak-reference agreement, exposed cases, not independent accuracy'}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
    m=ContextUNet();m.load_state_dict(torch.load(B/'native_alpha20_v6/selected.pt',weights_only=False,map_location='cpu')['state_dict']);m.eval();torch.set_num_threads(2)
    rows=[]
    for name in CASES:
        with np.load(B/'data_audit'/f'{name}_raster.npz') as z:a=z['alpha_v'].astype('float32');w=z['wall'];times=z['times']
        with np.load(B/'native_topology_alpha20'/f'{name}_native_labels.npz') as z:target=z['target']
        for kind,level in protocol['conditions']:
            for seed in SEEDS:
                rng=np.random.default_rng(seed);pred={k:[] for k in (['neural_filled','raster4_filled','raster8_filled'] if kind=='missing' else ['raster4_smooth','raster8_smooth'])}
                for v in a:
                    q,missing=corrupt(v,w,kind,level,rng)
                    if kind=='noise':
                        den=ndi.gaussian_filter((~w).astype(float),.7)
                        q=ndi.gaussian_filter(q,.7)/np.maximum(den,1e-12);suffix='smooth'
                    else:
                        ix=ndi.distance_transform_edt(missing|w,return_distances=False,return_indices=True)
                        q[missing]=q[tuple(ix[:,missing])];suffix='filled'
                        with torch.no_grad():pred['neural_filled'].append(m(torch.from_numpy(inputs(q,w)[None])).argmax(1)[0].numpy().astype('uint8'))
                    q[w]=0
                    for conn in [1,2]:pred[f'raster{4 if conn==1 else 8}_{suffix}'].append(baseline(q,w,conn))
                pred={k:np.stack(v) for k,v in pred.items()}
                rows.append(dict(case=name,group='development' if name in CASES[4:] else 'training',kind=kind,level=level,seed=seed,frames=len(a),metrics={k:score(p,target) for k,p in pred.items()}))
                np.savez_compressed(out/f'{name}_{kind}_{level:g}_{seed}.npz',**pred)
        print('strong controls',name,flush=True)
    (out/'RESULTS.json').write_text(json.dumps(dict(protocol=protocol,rows=rows),indent=2))
if __name__=='__main__':main()
