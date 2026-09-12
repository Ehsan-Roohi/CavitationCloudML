"""Lower-rate adaptation from the existing successful representation."""
import sys,json,torch
import cavitation_alpha20_20260912 as run
run.O=run.B/'native_alpha20_v6'
run.C=run.R/'configs/cavitation_alpha20_transfer_20260912.json'
parent=run.B/'native_v4/selected.pt'
def initialized():
    m=run.ContextUNet();m.load_state_dict(torch.load(parent,weights_only=False,map_location='cpu')['state_dict']);return m
torch.set_num_threads(2)
if sys.argv[1]=='train':
    run.trainer.ContextUNet=initialized
    run.train()
    (run.O/'PARENT.json').write_text(json.dumps(dict(path=str(parent),sha256=run.digest(parent),wrapper_sha256=run.digest(__file__),new_threshold=.2),indent=2))
elif sys.argv[1]=='test':run.test()
else:raise ValueError('train or test required')
