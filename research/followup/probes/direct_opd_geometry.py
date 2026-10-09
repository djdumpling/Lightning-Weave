import json
import hashlib
import sys
from pathlib import Path
import torch
from slime.rollout.offline_direct_opd import offline_direct_opd_tilted_target_terms

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
b = torch.tensor([[0.4, 0.3, 0.29, 0.01]])
delta = torch.tensor([[0.7, -0.4]])
alpha=1.25

def terms(p, d=delta, a=alpha):
    return offline_direct_opd_tilted_target_terms(p[:,:2].log(), b[:,:2].log(), d, torch.zeros_like(d), alpha=a)

z=b.log().clone().requires_grad_(True)
loss=terms(z.softmax(-1))['tilted_target_loss'].sum()
grad=torch.autograd.grad(loss,z)[0]
r=torch.cat([delta,torch.zeros((1,2))],-1)
expected=-(b*r-b*(b*r).sum(-1,keepdim=True))/b[:,:2].sum(-1,keepdim=True)
q=(b.log()+r/alpha).softmax(-1)
zq=q.log().clone().requires_grad_(True)
fixed_grad=torch.autograd.grad(terms(zq.softmax(-1))['tilted_target_loss'].sum(),zq)[0]
swapped=b[:,[0,1,3,2]]
post_only=torch.tensor([[0.6,0.2]]).log()
q_post=(b.log()+torch.cat([post_only,torch.zeros((1,2))],-1)).softmax(-1)
original=terms(b)
scaled=terms(b,delta*2,alpha*2)
result={
 'initial_gradient_max_error':(grad-expected).abs().max().item(),
 'target_fixed_point_gradient_max':fixed_grad.abs().max().item(),
 'tail_action_probability_comparison':[b[0,3].item(),swapped[0,3].item()],
 'loss_before_tail_swap':original['tilted_target_loss'].item(),
 'loss_after_tail_swap':terms(swapped)['tilted_target_loss'].item(),
 'behavior_other_mass':b[:,2:].sum().item(),
 'post_only_target_other_mass':q_post[:,2:].sum().item(),
 'joint_weight_alpha_scaling_target_tv_error':(original['target_tv']-scaled['target_tv']).abs().item(),
 'joint_weight_alpha_scaling_loss_ratio':(scaled['tilted_target_loss']/original['tilted_target_loss']).item(),
}
repo = Path(__file__).resolve().parents[3]
result["identity"] = {
    "python": sys.version,
    "torch": torch.__version__,
    "device": "cpu",
    "seed": 0,
    "input_dtype": "float64",
    "loss_internal_dtype": "float32",
    "code_sha256": {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in [
            "slime/rollout/offline_direct_opd.py",
            "data_curation/build_direct_opd_composed_target.py",
            str(Path(__file__).resolve().relative_to(repo)),
        ]
    },
}
print(json.dumps(result,indent=2))
