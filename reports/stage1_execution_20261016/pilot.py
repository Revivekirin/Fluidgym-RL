"""Bounded VAL pilot worker. Only launch.py --execute starts it under a watchdog."""
import hashlib
import gc
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import p1_support as p1
from adapter import InstancePatch, PostSmoothingAdapter, array, require, gain_at

HERE = Path(__file__).resolve().parent
ACTIONS = (0., .2, .2, -.2, -.2, .2, .2, -.2, -.2, .2, .2, 0.)
ARMS = ('N-bare', 'N-hook', 'F', 'J')
ATOL, RTOL = 1e-5, 1e-5
ACT_ATOL, ACT_RTOL = 1e-7, 1e-6


def encode(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, default=encode, indent=2, allow_nan=False))
    temp.replace(path)


def digest(x):
    x = array(x)
    return hashlib.sha256(str((x.shape, str(x.dtype))).encode() + x.tobytes()).hexdigest()


def close(a, b, message, atol=ACT_ATOL, rtol=ACT_RTOL):
    a, b = array(a), array(b)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), message)


def clock(raw):
    return dict(solver_time=p1.scalar(raw._sim.total_time), solver_steps=int(raw._sim.total_step),
                control_steps=int(raw._n_steps))


def memory(raw):
    return array(raw._CylinderEnvBase__last_control)


def finite_native(raw):
    for i, block in enumerate(raw._domain.getBlocks()):
        for key in ('velocity', 'pressure'):
            require(np.isfinite(array(getattr(block, key))).all(), f'nonfinite native block {i} {key}')


def refs(raw):
    result = {}
    for side, face in (('top', '-y'), ('bottom', '+y')):
        live = raw._domain.getBlock(getattr(raw, f'_{side}_block_idx')).getBoundary(face)
        require(live is getattr(raw, f'_{side}_boundary'), f'stale {side} boundary reference')
        result[side] = live
    return result


def boundary_readback(raw, capture=None):
    result = {}
    for side, boundary in refs(raw).items():
        velocity = array(boundary.velocity)  # independent C++ getter, never an expected-value fallback
        if capture: capture(side+'_velocity', velocity)
        physical = array(boundary.getVelocity(False))
        if capture: capture(side+'_physical_getter', physical)
        close(velocity, physical, 'physical boundary getter disagreement')
        flux = array(boundary.GetFluxes())
        if capture: capture(side+'_flux', flux)
        computational = array(boundary.getVelocity(True))
        if boundary.hasTransform():
            determinant = array(boundary.transform)[..., -1]
            expected_flux = (computational * np.expand_dims(determinant, 1))[:, 1:2, ...]
        else:
            expected_flux = computational[:, 1:2, ...]
        close(flux, expected_flux, 'native boundary flux vs transformed velocity mismatch')
        result[side] = {'velocity_measured': velocity, 'flux_measured': flux,
                        'flux_from_measured_velocity_expected': expected_flux,
                        'outward_flux_sum': float(flux.sum()) * (-1 if side == 'top' else 1),
                        'velocity_type': str(boundary.velocityType)}
    result['domain_flux_measured'] = p1.scalar(raw._domain.GetBoundaryFluxBalance())
    require(np.isfinite(result['domain_flux_measured']), 'nonfinite domain boundary flux')
    return result


class NumericalObserver:
    """Instance solver guard, including reset warmup. Bare leaves _apply_action untouched."""
    def __init__(self, raw, path, adapter=None):
        self.raw, self.adapter = raw, adapter
        self.path = path
        self.pending_readbacks = {}
        self.active, self.calls, self.failed, self.outer = False, 0, 0, []
        self.expected_memory, self.command, self.profiles = None, None, None
        self.journal = open(path / 'solver.jsonl', 'x', buffering=1)
        original = raw._get_simulation
        self.sim_patches = []
        def create(*args, **kwargs):
            sim = original(*args, **kwargs)
            self.sim_patches.append(InstancePatch(sim, 'single_step', self.guard(sim, sim.single_step)))
            return sim
        self.patch = InstancePatch(raw, '_get_simulation', create)

    def emit(self, event):
        self.journal.write(json.dumps(event, default=encode, allow_nan=False) + '\n')
        self.journal.flush()

    def arm_profiles(self):
        self.profile_refs = {s: getattr(self.raw, f'_{s}_velocity') for s in ('top', 'bottom')}
        self.profiles = {s: array(getattr(self.raw, f'_{s}_velocity')) for s in ('top', 'bottom')}
        for p in self.profiles.values():
            require(np.isfinite(p).all() and np.any(p != 0), 'invalid/zero native jet profile')
            # Peak normalization is of vector magnitude, not individual components.
            close(np.asarray(np.max(np.sqrt(np.sum(p*p, axis=1)))), np.asarray(1.), 'profile peak normalization')
        refs(self.raw)

    def begin(self, command):
        require(not self.active, 'nested transition')
        self.active, self.command, self.expected_memory, self.outer = True, command.copy(), memory(self.raw), []

    def guard(self, sim, original):
        def checked(*args, **kwargs):
            self.pending_readbacks = {}
            self.calls += 1
            entry = {'call': self.calls, 'phase': 'control' if self.active else 'reset',
                     'status': 'entered', 'clock_before': clock(self.raw)}
            # Preserve an entry even when pre-call checks or the native call fail.
            self.emit(entry)
            try:
                finite_native(self.raw)
                if self.active:
                    require(len(self.outer) < 25, 'excess outer calls')
                    self.expected_memory = self.expected_memory + np.float32(.1) * (self.command-self.expected_memory)
                    u = memory(self.raw)
                    close(u, self.expected_memory, 'native smoother recurrence mismatch')
                    g = self.adapter.gain if self.adapter else 1.
                    b = boundary_readback(self.raw, lambda k,v:self.pending_readbacks.update({k:v}))
                    for side in ('top', 'bottom'):
                        require(getattr(self.raw, f'_{side}_velocity') is self.profile_refs[side] and np.array_equal(array(getattr(self.raw, f'_{side}_velocity')), self.profiles[side]),
                                'official profile tensor mutated')
                        expected = self.profiles[side] * (u * g)
                        b[side]['velocity_source_expected'] = expected
                        b[side]['effective_scalar_from_boundary_measured'] = float(np.sum(b[side]['velocity_measured'].astype(np.float64)*self.profiles[side])/np.sum(self.profiles[side].astype(np.float64)**2))
                        close(b[side]['velocity_measured'], expected, 'measured boundary != post-smoothing effectiveness')
                    arg = self.adapter.arguments[-1] if self.adapter else None
                    if arg is not None:
                        close(arg['smoothed_argument'], u, 'apply argument vs memory')
                        close(arg['effective_argument_to_native_apply'], u*g, 'effective apply argument')
                    entry.update(transition=int(self.raw._n_steps), outer=len(self.outer), gain=g,
                                 requested_action=self.command.copy(), smoothed_source_expected=self.expected_memory.copy(),
                                 smoothed_memory_measured=u, apply_arguments_measured=arg,
                                 setter_arguments_directly_observed=False, boundary=b)
                    self.outer.append(entry)
                    self.emit({**entry, 'status': 'pre_solver_readback'})
                flux = p1.scalar(self.raw._domain.GetBoundaryFluxBalance())
                tol = p1.scalar(sim.flux_balance_tol)
                require(np.isfinite(flux) and abs(flux) <= tol, 'native flux-balance contract failure')
                result = original(*args, **kwargs)
                require(bool(result), 'native single_step returned failure')
                finite_native(self.raw)
                if self.active:
                    after = boundary_readback(self.raw, lambda k,v:self.pending_readbacks.update({'after_'+k:v}))
                    for side in ('top', 'bottom'):
                        require(getattr(self.raw, f'_{side}_velocity') is self.profile_refs[side] and np.array_equal(array(getattr(self.raw, f'_{side}_velocity')), self.profiles[side]), 'profile mutated during solver')
                    for side in ('top', 'bottom'):
                        close(after[side]['velocity_measured'], entry['boundary'][side]['velocity_measured'],
                              'solver overwrote applied jet boundary')
                    entry['boundary_after_solver'] = after
                entry.update(status='success', native_success=bool(result), clock_after=clock(self.raw),
                             native_fields_finite=True, flux_before=flux, native_flux_tolerance=tol)
                self.emit(entry)
                return result
            except BaseException as exc:
                if self.pending_readbacks:
                    np.savez(self.path/f'failed_readback_call_{self.calls}.npz',**self.pending_readbacks)
                self.failed += 1
                self.emit({'call': self.calls, 'status': 'failed', 'error': repr(exc)})
                raise
        return checked

    def close(self):
        for patch in reversed(self.sim_patches):
            patch.close()
        self.patch.close()
        self.journal.close()


def snapshot(raw, path):
    """Accessible state inventory only; no clone/restore and no hidden-cache identity claim."""
    tensors, inventory = {}, {}
    def collect(prefix, obj, attrs):
        for key in attrs:
            value = getattr(obj, key, None)
            if value is None:
                inventory[prefix+key] = 'unavailable_or_unallocated'
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if v is not None:
                        tensors[prefix+key+str(i)] = array(v)
            elif hasattr(value, 'shape'):
                tensors[prefix+key] = array(value)
            else:
                inventory[prefix+key] = str(value)
    for i, block in enumerate(raw._domain.getBlocks()):
        collect(f'b{i}_', block, ('velocity','pressure','viscosity','vertexCoordinates','transform','faceTransform'))
        for j, bound in enumerate(block.getBoundaries()):
            collect(f'b{i}_boundary{j}_', bound, ('velocity','velocityType','transform','type'))
    collect('domain_', raw._domain, ('velocityRHS','velocityResult','pressureRHS','pressureRHSdiv','pressureResult'))
    inventory['not_captured'] = 'Domain C/A/P sparse system storage, packed pointer tensors, hidden C++ caches and unexposed flux-history buffers; no full-state replay claim'
    tensors['memory'] = memory(raw)
    for key, value in tensors.items():
        require(value.dtype != object and np.isfinite(value).all(), 'nonfinite/unsupported state inventory: '+key)
    np.savez(path.with_suffix('.npz'), **tensors)
    import torch
    state_rng = {'global': p1.rng_hash(p1.rng()), 'env_numpy': raw._np_rng.bit_generator.state,
                 'env_torch_cpu': digest(raw._torch_rng_cpu.get_state()),
                 'env_torch_cuda': digest(raw._torch_rng_cuda.get_state())}
    meta = {'clock': clock(raw), 'rng': state_rng, 'tensors': {k: {'sha256': digest(v), 'shape': list(v.shape), 'dtype': str(v.dtype)} for k,v in tensors.items()},
            'inventory': inventory, 'full_state_identity': 'UNRESOLVED: hidden C++/solver caches and unexposed flux-history buffers are not certified'}
    write(path.with_suffix('.json'), meta)
    return meta


def compare_files(a, b, out, label):
    """Record differences before raising; never change tolerances after observation."""
    rows = []
    with np.load(a) as aa, np.load(b) as bb:
        require(set(aa.files) == set(bb.files), 'snapshot keys differ')
        for key in aa.files:
            rows.append({'label': label, 'quantity': key, **p1.compare(aa[key], bb[key], ATOL, RTOL)})
    write(out / (label+'_differences.json'), rows)
    require(all(r['allclose'] for r in rows), 'repeat/prefix numerical mismatch: '+label)


def compare_records(a, b, out, label, count=13):
    aa, bb = p1.read(a/'records.json')[:count], p1.read(b/'records.json')[:count]
    require(len(aa) == len(bb) == count, 'incomplete comparison inputs')
    require(aa[0]['identity'] == bb[0]['identity'], 'physical initialization identity differs')
    require(aa[0]['effective'] == bb[0]['effective'], 'effective config differs')
    compare_files(a/'profiles.npz',b/'profiles.npz',out,label+'_profiles')
    rows=[]
    for i,(x,y) in enumerate(zip(aa,bb)):
        keys = ['solver_time','solver_steps','control_steps','memory'] + (['reward','drag','lift','outer_calls'] if i else [])
        for key in keys:
            atol,rtol = (0.,0.) if key in ('solver_steps','control_steps','outer_calls') else ((1e-6,0.) if key=='solver_time' else (ATOL,RTOL))
            rows.append({'observation_index': i, 'quantity': key, **p1.compare(x[key],y[key],atol,rtol)})
        compare_files(a/f'obs_{i:02d}.npz', b/f'obs_{i:02d}.npz', out, f'{label}_obs{i:02d}')
        # Save and compare all accessible native tensors at every control boundary.
        compare_files(a/f'state_{i:02d}.npz', b/f'state_{i:02d}.npz', out, f'{label}_state{i:02d}')
        if i:
            ax,bx=p1.read(a/f'actuator_{i-1:02d}.json'),p1.read(b/f'actuator_{i-1:02d}.json')
            require(len(ax)==len(bx)==25,'incomplete outer trace')
            for j,(ac,bc) in enumerate(zip(ax,bx)):
                for key in ('smoothed_memory_measured','gain'):
                    rows.append({'observation_index':i,'outer':j,'quantity':key,**p1.compare(ac[key],bc[key],ATOL,RTOL)})
                for side in ('top','bottom'):
                    for key in ('velocity_measured','flux_measured'):
                        rows.append({'observation_index':i,'outer':j,'quantity':side+'_'+key,**p1.compare(ac['boundary'][side][key],bc['boundary'][side][key],ATOL,RTOL)})
                for phase in ('clock_before','clock_after'):
                    for key in ('solver_time','solver_steps','control_steps'):
                        rows.append({'observation_index':i,'outer':j,'quantity':phase+'_'+key,**p1.compare(ac[phase][key],bc[phase][key],1e-6 if key=='solver_time' else 0.,0.)})
        am,bm = p1.read(a/f'state_{i:02d}.json'),p1.read(b/f'state_{i:02d}.json')
        require(am['rng']==bm['rng'] and am['inventory']==bm['inventory'], 'RNG/native state inventory differs')
    write(out/(label+'_scalar_differences.json'),rows)
    require(all(r['allclose'] for r in rows), 'scalar/clock/counter mismatch: '+label)


def provenance(repo, out):
    result = p1.provenance(repo, out)
    # Also gate every module on the environment import path, accepting only
    # the audited historical or Stage-0 refactor hashes (never arbitrary drift).
    extra = p1.read(HERE/'source_gate.json')
    for key, allowed in extra['rl_allowed'].items():
        path = repo/key
        if key.endswith('/environment_contract.py') and not path.exists():
            historical = p1.EXPECTED['historical_pipeline_hashes']['src/fluidgym_rl/tdmpc/environment.py']
            require(p1.sha(repo/'src/fluidgym_rl/tdmpc/environment.py') == historical, 'missing environment contract dependency')
        else:
            require(path.is_file() and p1.sha(path) in allowed, 'environment module provenance differs: '+key)
    import fluidgym
    package=Path(fluidgym.__file__).resolve().parent
    for key,h in extra['cpp_sources'].items():
        require(p1.sha(package/key)==h, 'C++ source drift: '+key)
    result['stage1_source_gate'] = extra
    result['protocol'] = {'split':'val','seed':0,'randomize':True,'horizon':80,'actions':ACTIONS,
                          'arms':ARMS,'repeats':2,'controls_per_run':12,'maximum_controls':96,
                          'shift_t':6,'reduced_gain':.5,'watchdog_seconds':600,
                          'repeat_atol':ATOL,'repeat_rtol':RTOL,'clock_atol':1e-6,'clock_rtol':0,
                          'actuator_atol':ACT_ATOL,'actuator_rtol':ACT_RTOL}
    write(out/'provenance.json', result)
    return result


def environment(arm, directory):
    import torch
    from types import SimpleNamespace
    from fluidgym_rl.envs import make_fluid_env
    from fluidgym.wrappers import FlattenObservation
    from fluidgym_rl.comparison.evaluate import NominalEnv
    raw=make_fluid_env('CylinderJet2D-easy-v0',flatten=False,mode='val',cuda_device=torch.device('cuda:0'))
    require(type(raw).__name__=='CylinderJetEnv2D' and type(raw).__module__=='fluidgym.envs.cylinder.jet_cylinder_env_2d','unexpected native class')
    require(list(inspect.signature(raw._apply_action).parameters)==['action'], 'native apply signature drift')
    require(not raw._differentiable and not raw._use_marl and raw._enable_actions,'unsupported native flags')
    identity=p1.EnvObserver(raw,trace_actuator=False)
    adapter=None if arm=='N-bare' else PostSmoothingAdapter(raw,arm)
    audit=NumericalObserver(raw,directory,adapter)
    cfg=SimpleNamespace(obs_shape=[302],action_dim=1,episode_length=80,device='cuda:0')
    return NominalEnv(FlattenObservation(raw),cfg),raw,identity,adapter,audit


def save_obs(directory, i, flat, info):
    np.savez(directory/f'obs_{i:02d}.npz',velocity=array(info['original_velocity']),pressure=array(info['original_pressure']),flat=array(flat))
    p1.native_obs(flat,info)


def effect_summary(out):
    result={}
    for arm in ('F','J'):
        metrics={key:[] for key in ('flat_rms','drag','lift')}
        for t in range(12):
            def flat(name):
                with np.load(out/name/f'obs_{t+1:02d}.npz') as x:return x['flat'].copy()
            n0,n1=flat('N-bare_0'),flat('N-bare_1')
            for metric in metrics:
                if metric=='flat_rms':
                    noise=float(np.sqrt(np.mean((n0-n1).astype(np.float64)**2)))
                    floor=float(np.sqrt(np.mean((ATOL+RTOL*np.abs(n0))**2)))
                    deltas=[float(np.sqrt(np.mean((flat(f'{arm}_{rep}')-n0).astype(np.float64)**2))) for rep in range(2)]
                else:
                    base=p1.read(out/'N-bare_0'/'records.json')[t+1][metric]
                    noise=abs(base-p1.read(out/'N-bare_1'/'records.json')[t+1][metric])
                    floor=ATOL+RTOL*abs(base)
                    deltas=[abs(p1.read(out/f'{arm}_{rep}'/'records.json')[t+1][metric]-base) for rep in range(2)]
                threshold=5*max(noise,floor)
                metrics[metric].append({'t':t,'nominal_repeat_noise':noise,'numerical_floor':floor,
                                        'threshold':threshold,'effect_by_repeat':deltas,
                                        'both_repeats_exceed':all(x>threshold for x in deltas)})
        start=0 if arm=='F' else 6
        passed=any(any(rows[t]['both_repeats_exceed'] and rows[t+1]['both_repeats_exceed'] for t in range(start,11)) for rows in metrics.values())
        result[arm]={'operational_downstream_signal':passed,'metrics':metrics,
                     'interpretation':'matched-reset trajectory contrast, not same-state counterfactual or history identifiability'}
    write(out/'downstream_effect.json',result)
    return result


def run(repo, out):
    import torch
    start=time.monotonic()
    cost={'attempted_controls':0,'completed_controls':0,'runs_started':0,'runs_completed':0,'outer_calls_including_resets':0,'solver_failures':0,'reset_outer_calls':0,'accepted_solver_steps_including_resets':0}
    write(out/'cost.json',cost)
    provenance(repo,out)
    for arm in ARMS:
        for rep in range(2):
            name=f'{arm}_{rep}';directory=out/name;directory.mkdir()
            random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            env,raw,identity,adapter,audit=environment(arm,directory)
            records=[];cost['runs_started']+=1
            prior_calls=cost['outer_calls_including_resets'];prior_failed=cost['solver_failures']
            try:
                flat,info=env.reset(seed=0,randomize=True)
                eff=p1.effective(raw)
                require(eff['smoothing']==.1 and eff['lift_penalty']==1.,'smoothing/reward contract differs')
                require(raw.episode_length==80 and raw.mode.value=='val','split/horizon differs')
                close(memory(raw),np.zeros(1,np.float32),'reset memory',0.,0.)
                audit.arm_profiles()
                b=boundary_readback(raw)
                for side in ('top','bottom'):
                    close(b[side]['velocity_measured'],np.zeros_like(b[side]['velocity_measured']),'reset boundary nonzero',0.,0.)
                write(directory/'reset_boundary.json',b)
                np.savez(directory/'profiles.npz',**audit.profiles)
                records.append({'identity':identity.identity(),'effective':eff,**clock(raw),'memory':memory(raw),'reset_solver_calls':audit.calls})
                save_obs(directory,0,flat,info);snapshot(raw,directory/'state_00');write(directory/'records.json',records)
                # All conditions must have the same physical reset before any controls.
                if name!='N-bare_0':compare_records(out/'N-bare_0',directory,out,name+'_reset',1)
                for t,a in enumerate(ACTIONS):
                    require(cost['attempted_controls']<96 and cost['runs_started']<=8,'hard CFD budget exceeded')
                    if arm=='J' and t==6:
                        compare_records(out/'N-bare_0',directory,out,name+'_pre_jump',7)
                    command=np.array([a],np.float32)
                    before=clock(raw);calls=audit.calls
                    audit.begin(command)
                    cost['attempted_controls']+=1
                    write(directory/f'transition_{t:02d}_begin.json', {'t':t, 'requested_action':command,'gain':gain_at(arm,t) if adapter else 1.,'clock':before,'memory_before':memory(raw)})
                    cost['outer_calls_including_resets']=prior_calls+audit.calls
                    cost['solver_failures']=prior_failed+audit.failed
                    write(out/'cost.json',cost)  # reservation persisted before the physical call
                    try:
                        flat,reward,term,trunc,info=env.step(command)
                        cost['completed_controls']+=1
                    finally:
                        audit.active=False
                        write(directory/f'actuator_{t:02d}.json',audit.outer)
                    save_obs(directory,t+1,flat,info)
                    now=clock(raw)
                    require(not term and not trunc and now['control_steps']==t+1,'unexpected episode boundary')
                    require(audit.calls-calls==25 and len(audit.outer)==25,'outer-call count mismatch')
                    require(now['solver_time']>before['solver_time'] and now['solver_steps']-before['solver_steps']>=25,'solver clock/counters invalid')
                    records.append({'t':t,'requested_action':command,'clipped_action':np.clip(command,-1,1),
                                    'gain':gain_at(arm,t) if adapter else 1.,'reward':reward,
                                    'drag':p1.scalar(info['drag']),'lift':p1.scalar(info['lift']),
                                    'memory':memory(raw),'outer_calls':audit.calls-calls,
                                    'terminated':term,'truncated':trunc,**now})
                    snapshot(raw,directory/f'state_{t+1:02d}')
                    write(directory/'records.json',records)
                cost['runs_completed']+=1
                write(directory/'status.json',{'status':'passed','completed_controls':12,'solver_failures':audit.failed})
            finally:
                write(directory/'records.json',records)
                cost['outer_calls_including_resets']=prior_calls+audit.calls;cost['solver_failures']=prior_failed+audit.failed
                cost['reset_outer_calls']+=records[0]['reset_solver_calls'] if records else audit.calls
                if hasattr(raw,'_sim'):cost['accepted_solver_steps_including_resets']+=int(raw._sim.total_step)
                cost['wall_seconds']=time.monotonic()-start
                write(out/'cost.json',cost)
                audit.close()
                if adapter:adapter.close()
                env.close()
                del env,raw,identity,adapter,audit
                gc.collect()
            if rep==1:compare_records(out/f'{arm}_0',directory,out,arm+'_repeat')
            if arm=='N-hook':compare_records(out/'N-bare_0',directory,out,name+'_bare_equivalence')
        # No F/J run can begin until all four nominal runs and comparisons pass.
        write(out/'gates.json',{'completed_arm':arm,'nominal_gate_passed':arm!='N-bare','full_state_replay':'UNRESOLVED'})
    effects=effect_summary(out)
    p1.protect(repo)
    provenance(repo,out)  # rehash source/backend/data/config/artifacts after the pilot
    require(cost['completed_controls']==96,'incomplete control budget')
    write(out/'worker_status.json',{'status':'passed','contract':'VERIFIED current-pinned-environment paired pilot',
                                  'downstream':effects,'full_state_replay':'UNRESOLVED',
                                  'history_necessity':'UNRESOLVED; not tested','stage2_authorized':False})


if __name__=='__main__':
    repo,out=map(Path,sys.argv[1:3]);sys.path.insert(0,str(repo/'src'))
    try:
        require(os.environ.get('FGRL_STAGE1_WATCHDOG')=='600','use launch.py; direct unbounded execution rejected')
        run(repo,out)
    except BaseException as exc:
        write(out/'worker_status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()})
        traceback.print_exc()
        sys.exit(1)
