from pathlib import Path
import torch
P=Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113/run/checkpoints/step-000200')
m=torch.load(P/'mp_rank_00_model_states.pt',map_location='cpu',weights_only=False)
print('MODEL_KEYS',list(m));print('MODULE',len(m['module']),list(m['module'])[:2])
o=torch.load(P/'zero_pp_rank_0_mp_rank_00_optim_states.pt',map_location='cpu',weights_only=False)
print('OPT_KEYS',list(o));o=o['optimizer_state_dict'];print('OPT_INNER',list(o))
b=o['base_optimizer_state'];print('BASE_KEYS',list(b));print('STATE_KEYS',[list(v) for v in b['state'].values()][:2]);print('STEPS',[v.get('step') for v in b['state'].values()][:3])
