from pathlib import Path
import argparse,numpy as np,torch
from cavitation_context_v3_20260912 import ContextUNet,inputs
p=argparse.ArgumentParser();p.add_argument('--case',default='Case1LES');p.add_argument('--output',required=True);a=p.parse_args()
b=Path(__file__).resolve().parents[1]/'results/cavitation_20260912'
m=ContextUNet();m.load_state_dict(torch.load(b/'native_alpha20_v6/selected.pt',map_location='cpu',weights_only=True)['state_dict']);m.eval();torch.set_num_threads(2)
with np.load(b/'data_audit'/f'{a.case}_raster.npz') as z:v=z['alpha_v'];w=z['wall'];t=z['times']
with torch.no_grad():pred=np.stack([m(torch.from_numpy(inputs(x,w)[None])).argmax(1)[0].numpy().astype('uint8') for x in v])
np.savez_compressed(a.output,prediction=pred,times=t,wall=w)
