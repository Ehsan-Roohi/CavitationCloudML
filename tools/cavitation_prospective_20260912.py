"""Two archive cases selected by ID before reading fields; frozen evaluation."""
from pathlib import Path
import json,sys,inspect,time
import numpy as np,torch,h5py
import cavitation_data_audit_20260912 as audit
import cavitation_native_topology_20260912 as native
from cavitation_context_v3_20260912 import ContextUNet,inputs,digest
from cavitation_neural_pilot_20260912 import score
from cavitation_stress_20260912 import baseline
R=Path(__file__).resolve().parents[1];B=R/'results/cavitation_20260912';O=B/'prospective_archive_v1'
def main():
    O.mkdir(exist_ok=True)
    if (O/'PROTOCOL.json').exists():raise RuntimeError('Already started; do not duplicate')
    protocol=dict(cases=['Case 8','Case 20'],selection='IDs fixed before field extraction or review; no outcome-based case selection',checkpoint_sha256=digest(B/'native_alpha20_v6/selected.pt'),reference='native-face alpha>=.20 weak labels; new case is not independent truth',code_sha256=digest(__file__))
    (O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));torch.set_num_threads(2)
    m=ContextUNet();m.load_state_dict(torch.load(B/'native_alpha20_v6/selected.pt',weights_only=False,map_location='cpu')['state_dict']);m.eval()
    audit.OUT=O/'data_audit'
    # A copied converter changes only membership metadata for new IDs.
    source=inspect.getsource(audit.rasterize).replace("'Case 23': 'test'}[case]","'Case 23': 'test'}.get(case, 'prospective')")
    assert '.get(case' in source
    (O/'rasterize_engine.py').write_text(source)
    exec(compile(source,'prospective_rasterize','exec'),audit.__dict__)
    reports=[]
    for case in protocol['cases']:
        sys.argv=['audit','--case',case+'.rar','--extract'];audit.main()
        audit.rasterize(case);name=case.replace(' ','_')
        meta=json.loads((audit.OUT/(name+'_raster.json')).read_text())
        with np.load(audit.OUT/(name+'_raster.npz')) as z:a=z['alpha_v'].astype('float32');w=z['wall'];times=z['times'];ids=z['source_cell_id'].astype(int)-1;roi=[z['x'][0],z['x'][-1],z['y'][0],z['y'][-1]]
        graph,wall,ext,out,mesh=native.read_mesh(R/meta['files'][0]['case_path'],roi)
        target=[];pred=[];b4=[];b8=[]
        for i,f in enumerate(meta['files']):
            with h5py.File(R/f['data_path']) as h:v=h['results/1/phase-3/cells/SV_VOF/1'][:]
            cls,u,obj,records=native.classify((v>=.2).astype(float),graph,wall,ext,out)
            y=cls[ids];y[(u[ids]>0)|w]=255;target.append(y)
            with torch.no_grad():pred.append(m(torch.from_numpy(inputs(a[i],w)[None])).argmax(1)[0].numpy().astype('uint8'))
            b4.append(baseline(a[i],w,1));b8.append(baseline(a[i],w,2))
        target=np.stack(target);pred=np.stack(pred)
        rec=dict(case=case,frames=len(a),time_range=times[[0,-1]].tolist(),metrics={'neural':score(pred,target),'raster4':score(np.stack(b4),target),'raster8':score(np.stack(b8),target)},cloud_reference_pixels=int((target==2).sum()),mesh=mesh)
        np.savez_compressed(O/(name+'_prediction.npz'),neural=pred,raster4=np.stack(b4),raster8=np.stack(b8),target=target,times=times)
        reports.append(rec);(O/'PARTIAL.json').write_text(json.dumps(reports,indent=2));print('PROSPECTIVE',case,rec['metrics'],flush=True)
    (O/'RESULTS.json').write_text(json.dumps(dict(protocol=protocol,cases=reports),indent=2))
if __name__=='__main__':main()
