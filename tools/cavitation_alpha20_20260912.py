"""Explicit user-requested alpha>=.2 reference, retraining and comparison."""
import json,argparse,re,time
from pathlib import Path
import numpy as np
import h5py,torch
import cavitation_native_topology_20260912 as native
import cavitation_native_train_20260912 as trainer
from cavitation_context_v3_20260912 import ContextUNet,evaluate,inputs,digest
R=Path(__file__).resolve().parents[1];B=R/'results/cavitation_20260912';D=B/'data_audit'
N=B/'native_topology_alpha20';O=B/'native_alpha20_v5';C=R/'configs/cavitation_alpha20_20260912.json'
CASES=['Case_13','Case_14','Case_16','Case1LES','Case_19','Case_24','Case_23']
def labels():
    N.mkdir(parents=True,exist_ok=True)
    # Boundary value .2 included; native classifier receives an explicit binary support.
    from scipy.sparse import csr_matrix
    g=csr_matrix(np.array([[0,1,0],[1,0,0],[0,0,0]]));a=np.array([.2,.21,.19])
    c,_,_,_=native.classify((a>=.2).astype(float),g,[True,False,False],[False]*3,[False]*3)
    assert np.array_equal(c,[1,1,0]);checks={'inclusive_point_two':True,'old_native_contract':native.synthetic_test()}
    (N/'CONTRACT.json').write_text(json.dumps(checks,indent=2))
    for name in CASES:
        dest=N/(name+'_native_labels.npz')
        if dest.exists():raise RuntimeError('Refusing to overwrite '+str(dest))
        meta=json.loads((D/(name+'_raster.json')).read_text())
        with np.load(D/(name+'_raster.npz')) as z:ids=z['source_cell_id'].astype(int)-1;w=z['wall'];t=z['times'];a=z['alpha_v'];roi=[z['x'][0],z['x'][-1],z['y'][0],z['y'][-1]]
        graph,wall,ext,out,mesh=native.read_mesh(R/meta['files'][0]['case_path'],roi)
        targets=[];raw=[];unc=[];components=[];frames=[]
        with np.load(B/'native_topology'/(name+'_native_labels.npz')) as z:old=z['raw_class']
        for i,f in enumerate(meta['files']):
            p=R/f['data_path'];assert digest(p)==f['data_sha256']
            with h5py.File(p) as h:alpha=h['results/1/phase-3/cells/SV_VOF/1'][:]
            cls,u,obj,records=native.classify((alpha>=.2).astype(float),graph,wall,ext,out)
            q=cls[ids];uu=u[ids];cc=obj[ids];assert np.array_equal((q>0)[~w],(a[i]>=.2)[~w])
            y=q.copy();y[(uu>0)|w]=255;q[w]=255;cc[w]=0;uu[w]=0
            overlap=(old[i]>0)&(old[i]<255)&~w
            frames.append(dict(time=float(t[i]),attached=int((y==1).sum()),detached=int((y==2).sum()),uncertain=int((uu>0).sum()),
                old_vapor_class_changes=int((overlap&(old[i]!=q)).sum()),objects=records))
            targets.append(y);raw.append(q);unc.append(uu);components.append(cc)
        np.savez_compressed(dest,target=np.stack(targets),raw_class=np.stack(raw),uncertainty=np.stack(unc),component_id=np.stack(components),times=t,wall=w)
        rec=dict(threshold=.2,method='native face-connected alpha_v>=0.2, exact wall contact; external/crop uncertain',mesh=mesh,frames=frames,
            raster_sha256=digest(D/(name+'_raster.npz')),label_sha256=digest(dest),script_sha256=digest(__file__),native_dependency_sha256=digest(native.__file__))
        (N/(name+'_native_labels.json')).write_text(json.dumps(rec,indent=2));print(name,[(r['attached'],r['detached']) for r in frames[:2]],flush=True)
def setup():
    trainer.O=O;trainer.N=N;trainer.C=C
def train():
    setup();trainer.train()
    (O/'WRAPPER_PROVENANCE.json').write_text(json.dumps(dict(wrapper_sha256=digest(__file__),trainer_sha256=digest(trainer.__file__),reference_threshold=.2,comparison='user-requested changed definition; all seven cases now exposed, no new independent test'),indent=2))
def test():
    setup();freeze=json.loads((O/'FREEZE.json').read_text());assert digest(O/'selected.pt')==freeze['checkpoint_sha256']
    model=ContextUNet();model.load_state_dict(torch.load(O/'selected.pt',weights_only=False,map_location='cpu')['state_dict']);model.eval()
    previous=ContextUNet();previous.load_state_dict(torch.load(B/'native_v4/selected.pt',weights_only=False,map_location='cpu')['state_dict']);previous.eval()
    report={}
    for name in CASES:
        c=trainer.load(name);p,m=evaluate(model,c);_,old=evaluate(previous,c)
        np.savez_compressed(O/(name+'_prediction.npz'),prediction=p,weak_reference=c['target'],times=c['times'])
        with torch.no_grad():z=model(torch.from_numpy(inputs(np.zeros(c['wall'].shape,'float32'),c['wall'])[None])).argmax(1)[0].numpy()
        rec=dict(metrics=m,previous_alpha50_model_on_same_alpha20_reference=old,source_sha256=c['source_sha256'],reference_sha256=c['reference_sha256'],wall_false_pixels=int(((p>0)&c['wall']).sum()),zero_vapor_false_pixels=int((z>0).sum()))
        report[name]=rec;print(name,json.dumps(m),flush=True)
    (O/'EVALUATION.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['labels','train','test']);args=ap.parse_args();torch.set_num_threads(2)
    {'labels':labels,'train':train,'test':test}[args.action]()
