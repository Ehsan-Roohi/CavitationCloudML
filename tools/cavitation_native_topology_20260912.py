"""Native Fluent face-connectivity weak labels; no raster contact dilation.

Classes: 0 liquid/background, 1 native wall-attached vapor, 2 native detached
vapor, 255 solid or uncertain. Uncertainty bits: 1 actual external CFD boundary,
2 native component extending beyond raster ROI. No physical event claim.
"""
import argparse
import hashlib
import json
import pathlib
import re
import numpy as np
import h5py
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=ROOT/'results/cavitation_20260912/data_audit'
OUT=ROOT/'results/cavitation_20260912/native_topology'
CASES=['Case_13','Case_14','Case_16','Case1LES','Case_19','Case_24']

def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def classify(alpha, adjacency, wall_contact, external_contact, outside_roi):
    """Return per-native-cell classes/component IDs and component records."""
    active=np.asarray(alpha)>=.5
    assert np.isfinite(alpha).all()
    ids=np.flatnonzero(active)
    classes=np.zeros(len(alpha),np.uint8)
    uncertainty=np.zeros(len(alpha),np.uint8)
    component_id=np.zeros(len(alpha),np.int32)
    if not len(ids):return classes,uncertainty,component_id,[]
    count,component=connected_components(adjacency[ids][:,ids],directed=False)
    haswall=np.bincount(component,weights=np.asarray(wall_contact)[ids],minlength=count)>0
    external=np.bincount(component,weights=np.asarray(external_contact)[ids],minlength=count)>0
    cropped=np.bincount(component,weights=np.asarray(outside_roi)[ids],minlength=count)>0
    flags=external.astype(np.uint8)+2*cropped.astype(np.uint8)
    classes[ids]=np.where(haswall[component],1,2)
    uncertainty[ids]=flags[component]
    component_id[ids]=component+1
    sizes=np.bincount(component,minlength=count)
    records=[dict(id=i+1,native_cells=int(sizes[i]),attached_native_wall=bool(haswall[i]),external_boundary=bool(external[i]),cropped=bool(cropped[i]),uncertainty=int(flags[i])) for i in range(count)]
    return classes,uncertainty,component_id,records

def synthetic_test():
    # One thin native bridge attaches cell 3; disconnected cells 4-5 do not.
    edges=np.array([[0,1],[1,2],[2,3],[4,5],[5,6]])
    graph=coo_matrix((np.ones(len(edges)),edges.T),shape=(7,7)).tocsr()
    wall=np.array([1,0,0,0,0,0,0],bool)
    external=np.array([0,0,0,0,0,0,1],bool)
    outside=np.zeros(7,bool)
    a=np.array([.8,.8,.8,.8,.8,.8,.2])
    c,u,ids,r=classify(a,graph,wall,external,outside)
    assert np.array_equal(c,[1,1,1,1,2,2,0]) and not u.any()
    a[1]=.2
    c,u,ids,r=classify(a,graph,wall,external,outside)
    assert np.array_equal(c,[1,0,2,2,2,2,0])
    a[6]=.8;outside[2]=True
    c,u,ids,r=classify(a,graph,wall,external,outside)
    assert u[2]==u[3]==2 and u[4]==u[5]==u[6]==1
    c,u,ids,r=classify(np.zeros(7),graph,wall,external,outside)
    assert not c.any() and not ids.any() and not r
    return {'checks':['native_bridge','disconnected_cloud','bridge_break','external_boundary_flag','crop_flag','empty'], 'passed':True}

def read_mesh(path,roi):
    with h5py.File(path,'r') as f:
        m=f['meshes/1'];n=int(m.attrs['cellCount'][0]);nf=int(m.attrs['faceCount'][0])
        assert int(m.attrs['dimension'][0])==2
        xy=m['nodes/coords/1'][:]
        assert np.all(m['faces/nodes/1/nnodes'][:]==2)
        nodes=m['faces/nodes/1/nodes'][:].reshape(-1,2).astype(np.int64)-1
        assert nodes.min()>=0 and nodes.max()<len(xy) and len(nodes)==nf
        c0=np.full(nf,-1,np.int64);c1=np.full(nf,-1,np.int64)
        for side,dest in [('c0',c0),('c1',c1)]:
            for ds in m[f'faces/{side}'].values():
                lo=int(ds.attrs['minId'][0])-1;hi=int(ds.attrs['maxId'][0])
                assert hi-lo==len(ds) and np.all(dest[lo:hi]==-1)
                dest[lo:hi]=ds[:].astype(np.int64)-1
        assert np.all((c0>=0)&(c0<n)) and np.all((c1>=-1)&(c1<n))
        interior=c1>=0;assert np.all(c0[interior]!=c1[interior])
        graph=coo_matrix((np.ones(int(interior.sum()),np.uint8),(c0[interior],c1[interior])),shape=(n,n)).tocsr()
        count,_=connected_components(graph,directed=False)
        assert count==1,'Native mesh has disconnected regions requiring review'
        wall=np.zeros(n,bool);external=np.zeros(n,bool);covered=np.zeros(nf,bool);zones=[]
        ft=m['faces/zoneTopology'];ct=m['cells/zoneTopology'];porous=np.zeros(n,bool)
        for name,lo,hi in zip(ct['name'][0].decode().split(';'),ct['minId'][:],ct['maxId'][:]):
            if name=='porous':porous[int(lo)-1:int(hi)]=True
        for name,lo,hi,ztype in zip(ft['name'][0].decode().split(';'),ft['minId'][:],ft['maxId'][:],ft['zoneType'][:]):
            lo=int(lo)-1;hi=int(hi);covered[lo:hi]=True;second=c1[lo:hi]
            boundary=second<0
            if name.startswith('airfoil'):
                assert int(ztype)==3 and np.all(boundary)
                wall[c0[lo:hi]]=True
            elif np.any(boundary):
                assert np.all(boundary) and name in ['inlet','outlet','symmetry-down','symmetry-up'],name
                external[c0[lo:hi]]=True
            else:assert int(ztype)==2
            zones.append(dict(name=name,zone_type=int(ztype),face_min_id=lo+1,face_max_id=hi,interior_faces=int((~boundary).sum()),boundary_faces=int(boundary.sum())))
        assert covered.all() and wall.any() and external.any()
        low=xy[nodes].min(axis=1);high=xy[nodes].max(axis=1)
        outside_face=(low[:,0]<roi[0])|(high[:,0]>roi[1])|(low[:,1]<roi[2])|(high[:,1]>roi[3])
        outside=np.zeros(n,bool);outside[c0[outside_face]]=True
        outside[c1[outside_face&interior]]=True
        meshhash=hashlib.sha256(xy.tobytes()+nodes.tobytes()+c0.tobytes()+c1.tobytes()).hexdigest()
        settings=f['settings/Rampant Variables'][0].decode()
        assert '(3 phase-domain vapor)' in settings
    invariant=dict(native_cells=n,native_faces=nf,whole_mesh_components=count,interior_faces=int(interior.sum()),airfoil_contact_cells=int(wall.sum()),external_contact_cells=int(external.sum()),porous_cells=int(porous.sum()),fluid_porous_interior_faces=int(np.sum(porous[c0[interior]]!=porous[c1[interior]])),zones=zones,topology_sha256=meshhash)
    return graph,wall,external,outside,invariant

def convert(case):
    output=OUT/(case+'_native_labels.npz')
    if output.exists():raise RuntimeError('Refusing to overwrite existing native label freeze: '+str(output))
    raster=DATA/(case+'_raster.npz');provenance=json.loads((DATA/(case+'_raster.json')).read_text())
    with np.load(raster) as z:
        source_ids=z['source_cell_id'].astype(np.int64)-1;solid=z['wall'].copy();times=z['times'].copy()
        roi=[float(z['x'][0]),float(z['x'][-1]),float(z['y'][0]),float(z['y'][-1])]
    files=provenance['files'];assert len(files)==len(times)
    graph,wall,external,outside,invariants=read_mesh(ROOT/files[0]['case_path'],roi)
    assert source_ids.min()>=0 and source_ids.max()<graph.shape[0]
    alltarget=[];allraw=[];allunc=[];allcomponents=[];frames=[]
    for i,record in enumerate(files):
        path=ROOT/record['data_path']
        assert digest(path)==record['data_sha256']
        with h5py.File(path,'r') as f:
            ds=f['results/1/phase-3/cells/SV_VOF/1'];assert int(ds.attrs['minId'][0])==1 and int(ds.attrs['maxId'][0])==graph.shape[0]
            alpha=ds[:];assert len(alpha)==graph.shape[0] and alpha.min()>=-1e-8 and alpha.max()<=1+1e-8
            time=float(re.search(r'\(flow-time ([^\)]+)\)',f['settings/Data Variables'][0].decode()).group(1))
            assert abs(time-times[i])<1e-10
        native,unc,component,objects=classify(alpha,graph,wall,external,outside)
        raw=native[source_ids];u=unc[source_ids];cid=component[source_ids]
        target=raw.copy();target[(u!=0)|solid]=255
        raw[solid]=255;u[solid]=0;cid[solid]=0
        # Count native components that remain represented after raster sampling.
        ids,counts=np.unique(cid[~solid],return_counts=True);pixels=dict(zip(ids.tolist(),counts.tolist()))
        for obj in objects:obj['raster_pixels']=pixels.get(obj['id'],0)
        frames.append(dict(index=i,time=time,data_sha256=record['data_sha256'],native_vapor_cells=int((native>0).sum()),objects=objects,raster_attached=int((target==1).sum()),raster_detached=int((target==2).sum()),raster_uncertain=int(((u!=0)&~solid).sum()),raster_solid=int(solid.sum())))
        alltarget.append(target);allraw.append(raw);allunc.append(u);allcomponents.append(cid)
        print(f'{case}: {i+1}/{len(files)} attached={(target==1).sum()} detached={(target==2).sum()} uncertain={((u!=0)&~solid).sum()}',flush=True)
    np.savez_compressed(output,target=np.stack(alltarget),raw_class=np.stack(allraw),uncertainty=np.stack(allunc),component_id=np.stack(allcomponents),times=times,wall=solid)
    report=dict(case=case,method='full-native-cell face adjacency; alpha>=0.5; exact airfoil wall-face contact; no minimum component size or pixel dilation',uncertainty_bits={'1':'component contacts actual external CFD boundary','2':'component has native cell vertex outside raster ROI'},raster_sha256=digest(raster),native_labels_sha256=digest(output),code_sha256=digest(__file__),roi=roi,mesh=invariants,frames=frames,limitation='Weak solver/topology reference; no independent human labels, 3D connectivity or time-resolved identity established')
    (OUT/(case+'_native_labels.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(str(output),flush=True)

def audit_outputs():
    from cavitation_topology_20260912 import weak_reference
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    results=[];plot_data=[]
    for case in CASES:
        report=json.loads((OUT/(case+'_native_labels.json')).read_text())
        source=json.loads((DATA/(case+'_raster.json')).read_text())
        frozen_hash=report['mesh']['topology_sha256'];verified=[]
        for record in source['files']:
            with h5py.File(ROOT/record['case_path'],'r') as f:
                m=f['meshes/1'];nf=int(m.attrs['faceCount'][0])
                xy=m['nodes/coords/1'][:];nodes=m['faces/nodes/1/nodes'][:].reshape(-1,2).astype(np.int64)-1
                arrays=[]
                for side in ['c0','c1']:
                    a=np.full(nf,-1,np.int64)
                    for ds in m[f'faces/{side}'].values():
                        start=int(ds.attrs['minId'][0])-1;a[start:start+len(ds)]=ds[:].astype(np.int64)-1
                    arrays.append(a)
                current=hashlib.sha256(xy.tobytes()+nodes.tobytes()+arrays[0].tobytes()+arrays[1].tobytes()).hexdigest()
                assert current==frozen_hash, 'Temporal native connectivity changed'
                verified.append(record['case_path'])
        with np.load(DATA/(case+'_raster.npz')) as z,np.load(OUT/(case+'_native_labels.npz')) as label:
            assert digest(DATA/(case+'_raster.npz'))==report['raster_sha256']
            assert digest(OUT/(case+'_native_labels.npz'))==report['native_labels_sha256']
            target=label['target'];raw=label['raw_class'];unc=label['uncertainty'];solid=z['wall'];perframe=[]
            for index,alpha in enumerate(z['alpha_v']):
                old=weak_reference(alpha,solid)[0];native=target[index]
                comparable=(old!=255)&(native!=255)
                flips=comparable&(((old==1)&(native==2))|((old==2)&(native==1)))
                # Reject accidental phase-index or native-to-raster misalignment.
                native_vapor=(raw[index]==1)|(raw[index]==2)
                alpha_vapor=(alpha>=.5)&~solid
                assert np.array_equal(native_vapor,alpha_vapor)
                perframe.append(dict(index=index,time=float(z['times'][index]),attached_to_detached=int(((old==1)&(native==2)).sum()),detached_to_attached=int(((old==2)&(native==1)).sum()),flipped_pixels=int(flips.sum()),new_native_small_component_pixels=int(((old==255)&(native!=255)&~solid).sum()),new_uncertain_pixels=int(((unc[index]!=0)&~solid).sum())))
            record=dict(case=case,topology_verified_frames=len(verified),native_arrays_match_raster_alpha=True,native_label_sha256=report['native_labels_sha256'],native_attached_pixels=int((target==1).sum()),native_detached_pixels=int((target==2).sum()),uncertain_pixels=int(((unc!=0)&~solid).sum()),flipped_pixels=sum(r['flipped_pixels'] for r in perframe),frames=perframe)
            results.append(record)
            if case=='Case1LES':
                worst=sorted(perframe,key=lambda r:r['flipped_pixels'],reverse=True)[:3]
                for rec in worst:
                    i=rec['index'];plot_data.append((i,float(z['times'][i]),z['alpha_v'][i].copy(),weak_reference(z['alpha_v'][i],solid)[0],target[i].copy(),z['x'].copy(),z['y'].copy(),z['wall_segments'].copy(),rec))
        print(json.dumps({k:v for k,v in record.items() if k!='frames'}),flush=True)
    fig,axs=plt.subplots(len(plot_data),3,figsize=(16,8),constrained_layout=True)
    for row,(index,t,alpha,old,native,x,y,segments,rec) in enumerate(plot_data):
        for col,field in enumerate([alpha,old,native]):
            masked=np.ma.masked_where(native==255,field) if col==0 else np.ma.masked_where(field==255,field)
            axs[row,col].pcolormesh(x,y,masked,shading='nearest',vmin=0,vmax=1 if col==0 else 2,cmap='viridis' if col==0 else 'coolwarm')
            axs[row,col].add_collection(LineCollection(segments,colors='black',linewidths=.5))
            axs[row,col].set_aspect('equal');axs[row,col].set_facecolor('gray');axs[row,col].set_title(f't={t:.2f}s '+['vapor fraction','old raster teacher','native-face teacher'][col]);axs[row,col].set_xlabel('x (m)')
    fig.suptitle('Three largest TRAIN LES topology disagreements; 1 attached / 2 detached. Weak labels, not human truth.')
    fig.savefig(OUT/'NATIVE_DISAGREEMENT_REVIEW.png',dpi=160);plt.close(fig)
    (OUT/'NATIVE_AUDIT.json').write_text(json.dumps({'checks':synthetic_test(),'all_native_topologies_constant':True,'uncertainty_only_external_boundary_or_crop':True,'no_disagreement_filtering':True,'cases':results},indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=CASES+['Case_23']);ap.add_argument('--test-only',action='store_true');ap.add_argument('--audit',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    checks=synthetic_test();(OUT/'SYNTHETIC_CHECKS.json').write_text(json.dumps(checks,indent=2))
    if args.audit:audit_outputs();return
    if args.test_only:print(checks);return
    for case in [args.case] if args.case else CASES:convert(case)

if __name__=='__main__':main()
