"""Run the scenarios on PRISTINE flex-pi @ 044b8c2 and save the outputs (bit-identity references)."""
import sys

import torch

import common as C
import tiny_flexpi as T

ref = {}
m = T.build_tiny_flexpi()
total, d = C.run_training_loss(m)
ref["train_total"] = total.detach().clone()
for k, v in d.items():
    ref["train_" + k] = torch.tensor(v, dtype=torch.float64)
for k, v in C.flatten_out(C.run_action_only(m)).items():
    ref["ao_" + k] = v.clone()
for k, v in C.flatten_out(C.run_joint(m)).items():
    ref["joint_" + k] = v.clone()
with torch.no_grad():
    outs = m._predict_joint_noise_unified_impl(**C.joint_noise_inputs(m))
for i, t in enumerate(outs if isinstance(outs, (tuple, list)) else [outs]):
    if torch.is_tensor(t):
        ref[f"jn_{i}"] = t.clone()
torch.save(ref, sys.argv[1])
print("saved", len(ref), "reference tensors to", sys.argv[1])
