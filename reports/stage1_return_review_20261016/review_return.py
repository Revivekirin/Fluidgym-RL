import pathlib,json,hashlib,numpy as np
R=pathlib.Path('/Users/jihyekim/Desktop/Fluidgym-RL');P=R/'results/stage1_pilot_20261016T124327Z_f587f185';O=R/'reports/stage1_return_review_20261016';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
checks=0
def need(x,msg):
 global checks
 checks+=1
 if not x:raise AssertionError(msg)
def close(a,b,msg,atol=1e-5,rtol=1e-5):
 a,b=np.asarray(a),np.asarray(b);need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=atol,rtol=rtol),msg)
manifest=read(P/'output_hashes.json')
for rel,h in manifest.items():need(sha(P/rel)==h,'output hash '+rel)
need(read(P/'package_hashes.json')==read(R/'reports/stage1_execution_20261016/package_hashes.json'),'package identity')
prov=read(P/'provenance.json');expected=read(R/'reports/stage1_execution_20261016/expected.json')
need(prov['source_hashes']==expected['sources'],'source identity');need(prov['backend']['sha256']==expected['backend_hash'],'backend identity')
for k,v in read(P/'post_preservation.json')['checks'].items():need(v['actual']==v['expected'],'preserved '+k)
A=[0,.2,.2,-.2,-.2,.2,.2,-.2,-.2,.2,.2,0];records={};obs={};metrics={};count=0;reset=0;solver=0;max_boundary=0.;max_recurrence=0.;max_flux=0.
for arm in ['N-bare','N-hook','F','J']:
 for rep in range(2):
  name=f'{arm}_{rep}';d=P/name;rr=records[name]=read(d/'records.json');need(len(rr)==13,'records count');need(rr[0]['identity']==records['N-bare_0'][0]['identity'],'matched identity');need(rr[0]['identity']['split']=='val','VAL');reset+=rr[0]['reset_solver_calls'];u=np.zeros(1,np.float32)
  pp=np.load(d/'profiles.npz');obs[name]=[]
  for i in range(13):
   with np.load(d/f'obs_{i:02d}.npz') as z:
    for k in z.files:need(z[k].dtype==np.float32 and np.isfinite(z[k]).all(),'finite observation')
    need(z['velocity'].shape==(151,2) and z['pressure'].shape==(151,) and z['flat'].shape==(302,),'obs shape');need(np.array_equal(z['flat'],z['velocity'].ravel()),'official flatten');obs[name].append({k:z[k].copy() for k in z.files})
  for t,a in enumerate(A):
   record=rr[t+1];g=.5 if arm=='F' or (arm=='J' and t>=6) else 1.;need(record['gain']==g,'gain schedule');close(record['requested_action'],[a],'action',1e-7,0);need(not record['terminated'] and not record['truncated'],'horizon retained');need(record['control_steps']==t+1,'control counter')
   e=rr[0]['effective'];close(record['reward'],e['reward_reference']-record['drag']-abs(record['lift']),'original reward')
   rows=read(d/f'actuator_{t:02d}.json');need(len(rows)==25,'25 outer calls')
   for j,x in enumerate(rows):
    u=u+np.float32(.1)*(np.array([a],np.float32)-u);close(x['smoothed_memory_measured'],u,'smoother',1e-7,1e-6);max_recurrence=max(max_recurrence,float(np.max(np.abs(np.asarray(x['smoothed_memory_measured'])-u))))
    need(x['gain']==g and x['transition']==t and x['outer']==j,'outer timing');need(x['native_success'] and x['native_fields_finite'] and x['status']=='success','solver success')
    need(abs(x['flux_before'])<=x['native_flux_tolerance'],'native flux tolerance');max_flux=max(max_flux,abs(x['flux_before']))
    for side in ['top','bottom']:
     b=x['boundary'][side];meas=np.array(b['velocity_measured']);exp=pp[side]*(u*g)
     close(meas,exp,'boundary delivery',1e-7,1e-6);max_boundary=max(max_boundary,float(np.max(np.abs(meas-exp))))
     close(x['boundary_after_solver'][side]['velocity_measured'],meas,'boundary persisted',1e-7,1e-6)
     close(b['flux_measured'],b['flux_from_measured_velocity_expected'],'flux consistency',1e-7,1e-6)
    if arm!='N-bare':close(x['apply_arguments_measured']['effective_argument_to_native_apply'],u*g,'native apply input',1e-7,1e-6)
    need(x['clock_after']['solver_time']>x['clock_before']['solver_time'] and x['clock_after']['solver_steps']>x['clock_before']['solver_steps'],'clock progress')
   close(record['memory'],u,'memory at boundary',1e-7,1e-6);count+=1
  events=[json.loads(line) for line in (d/'solver.jsonl').read_text().splitlines()];success=[x for x in events if x['status']=='success'];need(len(success)==321,'journal success count');need(not any(x['status']=='failed' for x in events),'journal no failures');solver+=len(success)
  metrics[name]={'reward_sum':sum(x['reward'] for x in rr[1:]),'drag_mean':float(np.mean([x['drag'] for x in rr[1:]])),'lift_mean':float(np.mean([x['lift'] for x in rr[1:]]))}
pairs=[(f'{arm}_0',f'{arm}_1',13) for arm in ['N-bare','N-hook','F','J']]+[('N-bare_0',f'N-hook_{i}',13) for i in range(2)]+[('N-bare_0',f'J_{i}',7) for i in range(2)]
comparisons=[]
for a,b,n in pairs:
 maxobs=0.;maxstate=0.;maxscalar=0.
 for i in range(n):
  for key in ['flat','velocity','pressure']:
   x,y=obs[a][i][key],obs[b][i][key];close(x,y,'paired obs');maxobs=max(maxobs,float(np.max(np.abs(x-y))))
  for key in ['solver_time','solver_steps','control_steps']+(['reward','drag','lift'] if i else []):
   x,y=records[a][i][key],records[b][i][key];close(x,y,'paired scalar',1e-6 if key=='solver_time' else (0 if key.endswith('steps') else 1e-5),0 if key=='solver_time' or key.endswith('steps') else 1e-5);maxscalar=max(maxscalar,abs(x-y))
  with np.load(P/a/f'state_{i:02d}.npz') as x,np.load(P/b/f'state_{i:02d}.npz') as y:
   need(set(x.files)==set(y.files),'state inventory')
   for key in x.files:
    close(x[key],y[key],'accessible paired state');maxstate=max(maxstate,float(np.max(np.abs(x[key]-y[key]))) if x[key].size else 0)
  need(read(P/a/f'state_{i:02d}.json')['rng']==read(P/b/f'state_{i:02d}.json')['rng'],'RNG match')
 comparisons.append(dict(a=a,b=b,observations=n,max_observation_abs=maxobs,max_native_state_abs=maxstate,max_scalar_or_clock_abs=maxscalar))
effects=read(P/'downstream_effect.json');effects_summary={}
for arm in ['F','J']:
 effects_summary[arm]={}
 for key in ['flat_rms','drag','lift']:
  for t in range(12):
   if key=='flat_rms':
    base=obs['N-bare_0'][t+1]['flat'];noise=float(np.sqrt(np.mean((base-obs['N-bare_1'][t+1]['flat']).astype(np.float64)**2)));floor=float(np.sqrt(np.mean((1e-5+1e-5*np.abs(base))**2)));diff=[float(np.sqrt(np.mean((obs[f'{arm}_{r}'][t+1]['flat']-base).astype(np.float64)**2))) for r in range(2)]
   else:
    base=records['N-bare_0'][t+1][key];noise=abs(base-records['N-bare_1'][t+1][key]);floor=1e-5+1e-5*abs(base);diff=[abs(records[f'{arm}_{r}'][t+1][key]-base) for r in range(2)]
   x=effects[arm]['metrics'][key][t];close(diff,x['effect_by_repeat'],'downstream recomputation',1e-12,1e-10);close(5*max(noise,floor),x['threshold'],'threshold',1e-12,1e-10);need(x['both_repeats_exceed']==all(v>5*max(noise,floor) for v in diff),'criterion')
  hits=[x['t'] for x in effects[arm]['metrics'][key] if x['both_repeats_exceed']];effects_summary[arm][key]={'exceeding_transitions':hits,'first_consecutive_pair':next(([t,t+1] for t in hits if t+1 in hits),None)}
cost=read(P/'cost.json');need(count==cost['completed_controls']==cost['attempted_controls']==96,'budget');need(reset==168==cost['reset_outer_calls'],'reset cost');need(solver==2568==cost['outer_calls_including_resets'],'solver cost');need(read(P/'launcher_status.json')['wall_seconds']<600,'watchdog')
result={'status':'VERIFIED returned evidence consistent','input_directory':str(P),'output_hashes_verified':len(manifest),'checks':checks,'additional_CFD_controls':0,'cost':cost,'launcher_wall_seconds':read(P/'launcher_status.json')['wall_seconds'],'comparisons':comparisons,'max_boundary_abs_residual':max_boundary,'max_smoother_abs_residual':max_recurrence,'max_domain_flux_abs':max_flux,'downstream':effects_summary,'per_run':metrics,'limitations':['one VAL initial-domain identity (8), seed0, two numerical repeats','no full-state replay certificate','no physical-accuracy validation','no policy/representation/history-identifiability experiment']}
(O/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
