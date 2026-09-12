"""Reproducible checks for the split-decoder comparison; no training data used."""
import json
import torch
from cavitation_context_v3_20260912 import ContextUNet,digest
from cavitation_decoder_comparison_20260912 import SplitDecoder,logprob,loss_fn,O

torch.set_num_threads(1);torch.manual_seed(12)
a=ContextUNet();b=SplitDecoder(a)
x=torch.rand(2,3,32,64);y=torch.randint(0,3,(2,32,64));y[:,0]=255
p=logprob(a,x);q=logprob(b,x)
assert torch.allclose(p.exp(),q.exp(),atol=2e-6)
assert torch.allclose(q.exp().sum(1),torch.ones_like(q[:,0]),atol=2e-6)
w=torch.tensor([.2,1.,1.6]);la=loss_fn(p,y,w);lb=loss_fn(q,y,w)
assert torch.allclose(la,lb,atol=2e-6)
lb.backward()
assert all(any(t.grad is not None and t.grad.abs().sum()>0 for t in getattr(b,k).parameters()) for k in ['enc','vapor_dec','connection_dec'])
assert next(b.vapor_dec.parameters()).data_ptr()!=next(b.connection_dec.parameters()).data_ptr()
# Ignored-pixel logits cannot change the objective.
perturbed=q.detach().clone();perturbed[:,:,0]=torch.randn_like(perturbed[:,:,0])
assert torch.allclose(loss_fn(q.detach(),y,w),loss_fn(perturbed,y,w),atol=2e-6)
rec=dict(initial_equivalence=True,normalized_hierarchical_probabilities=True,
         equal_initial_objective=True,both_decoders_receive_gradients=True,
         independent_decoder_storage=True,ignore_contract=True,code_sha256=digest(__file__))
(O/'CONTRACT_TESTS.json').write_text(json.dumps(rec,indent=2));print(rec)
