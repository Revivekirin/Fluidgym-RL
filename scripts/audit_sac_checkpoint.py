#!/usr/bin/env python
"""Read-only CPU loader audit of a trusted public SB3 archive. No feature remapping."""
import argparse,json,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fluidgym_rl.official_sac import inspect_zip,sha256_file,validate_contract
from fluidgym_rl.comparison.schema import write_json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint',type=Path); p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
result={'checkpoint':str(a.checkpoint.resolve()),'sha256':sha256_file(a.checkpoint),'archive':inspect_zip(a.checkpoint),
        'physics':'unverified; observation dimensions do not establish physics or field ordering'}
try:
    import stable_baselines3 as sb3
    model=sb3.SAC.load(a.checkpoint,device='cpu')
except Exception as exc:
    result.update(status='loader_incompatible',error=repr(exc),traceback=traceback.format_exc())
else:
    import numpy as np
    from gymnasium.spaces import Box
    from types import SimpleNamespace
    try:
        result['spaces']=validate_contract(model,SimpleNamespace(observation_space=Box(-np.inf,np.inf,(302,),dtype=np.float32),action_space=Box(-1,1,(1,),dtype=np.float32)),installed_sb3=sb3.__version__,saved_sb3=result['archive']['sb3_version_saved'])
        result['status']='loader_and_nominal_space_shape_passed; physics/preprocessing/protocol remain unverified'
    except Exception as exc:
        result.update(status='space_contract_incompatible',error=repr(exc),traceback=traceback.format_exc())
write_json(a.out,result)
print(json.dumps(result,indent=2))
sys.exit(1 if result['status'].endswith('incompatible') else 0)
