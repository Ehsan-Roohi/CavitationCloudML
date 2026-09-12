"""Explicit weak-reference topology, separate from learned predictions.

Detached means disconnected from wall in a resolved 2-D section, NOT proven
three-dimensional detachment or a temporally established shedding event.
"""
import numpy as np
from scipy import ndimage as ndi

def weak_reference(alpha, wall, threshold=0.5, min_pixels=9, wall_band=2):
    alpha=np.asarray(alpha); wall=np.asarray(wall,dtype=bool)
    if alpha.shape!=wall.shape or alpha.ndim!=2:
        raise ValueError('alpha and wall must be matching 2-D arrays')
    valid=np.isfinite(alpha)&~wall
    vapor=valid&(alpha>=threshold)
    components,n=ndi.label(vapor,np.ones((3,3),bool))
    contact=ndi.binary_dilation(wall,iterations=wall_band)&~wall
    labels=np.zeros(alpha.shape,np.uint8); labels[~valid]=255
    objects=[]
    for k in range(1,n+1):
        mask=components==k; area=int(mask.sum())
        if area<min_pixels:
            labels[mask]=255
            continue
        attached=bool(np.any(mask&contact))
        labels[mask]=1 if attached else 2
        y,x=np.nonzero(mask)
        edge=bool(np.any((y==0)|(y==alpha.shape[0]-1)|(x==0)|(x==alpha.shape[1]-1)))
        objects.append(dict(component=k,area_pixels=area,attached_2d=attached,
                            centroid_xy=[float(x.mean()),float(y.mean())],
                            touches_image_boundary=edge))
    return labels,components,objects

def test_contract():
    wall=np.zeros((32,64),bool);wall[20:23,4:35]=True
    a=np.zeros(wall.shape);a[16:20,8:30]=.9;a[10:15,45:51]=.9
    a[2,2]=.9;a[25,25]=np.nan
    labels,_,obj=weak_reference(a,wall)
    assert (labels[16:20,8:30]==1).all()
    assert (labels[10:15,45:51]==2).all()
    assert labels[2,2]==255 and labels[25,25]==255
    assert len(obj)==2 and (labels[wall]==255).all()
    empty,_,objects=weak_reference(np.zeros_like(a),wall)
    assert not objects and not np.any((empty==1)|(empty==2))
    try: weak_reference(a,wall[:1])
    except ValueError: pass
    else: raise AssertionError('shape mismatch accepted')
    return {'synthetic_contract_checks':6,'passed':True,'physical_validation':False}

if __name__=='__main__':
    import json
    print(json.dumps(test_contract()))
