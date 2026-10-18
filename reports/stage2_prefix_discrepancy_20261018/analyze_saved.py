"""Read-only saved-array analysis. Imports no FluidGym, Torch, or policy code."""
import csv,hashlib,json,pathlib,re,zipfile
import numpy as np
R=pathlib.Path(__file__).resolve().parents[2];P=R/'reports/stage2_frozen_20261016T131654Z_c1efaf57';O=pathlib.Path(__file__).resolve().parent
A=P/'e0_p0_S0';B=P/'e0_p0_S2'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()
def save(name,data):(O/name).write_text(json.dumps(data,indent=2,allow_nan=False))
manifest=read(P/'output_hashes.json');assert all(sha(P/k)==v for k,v in manifest.items())
rows=[];localization=[];inventory={};timeline=[]
def metrics(a,b,label,boundary,kind):
 a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
 x,y=a.astype(np.float64),b.astype(np.float64);d=y-x;absolute=np.abs(d);threshold=1e-5+1e-5*np.abs(y);symmetric=np.maximum(np.abs(x),np.abs(y));near=symmetric<1e-5
 z={'boundary':boundary,'kind':kind,'field':label,'shape':list(a.shape),'dtype':str(a.dtype),'elements':a.size,'sha_S0':digest(a),'sha_S2':digest(b),'bitwise_equal':bool(np.array_equal(a,b)),
    'max_abs':float(absolute.max()) if a.size else 0.,'rms':float(np.sqrt(np.mean(d*d))) if a.size else 0.,
    'max_relative_floor_1e_12':float(np.max(absolute/np.maximum(np.abs(x),1e-12))) if a.size else 0.,
    'max_symmetric_relative_floor_1e_5':float(np.max(absolute/np.maximum(symmetric,1e-5))) if a.size else 0.,
    'near_zero_elements_below_1e_5':int(near.sum()),'near_zero_max_abs':float(absolute[near].max()) if near.any() else 0.,
    'exceedance_count':int(np.sum(absolute>threshold)),'max_error_to_tolerance':float(np.max(absolute/threshold)) if a.size else 0.,
    'mean_S0':float(x.mean()) if a.size else 0.,'mean_S2':float(y.mean()) if a.size else 0.,'mean_signed_difference':float(d.mean()) if a.size else 0.,
    'mean_subtracted_max_abs':float(np.max(np.abs(d-d.mean()))) if a.size else 0.,'mean_subtracted_rms':float(np.std(d)) if a.size else 0.}
 rows.append(z);return z
for i in range(4):
 for stem,kind in [('state','native'),('obs','sensor')]:
  with np.load(A/f'{stem}_{i:02d}.npz') as xa,np.load(B/f'{stem}_{i:02d}.npz') as xb:
   assert xa.files==xb.files
   inventory[f'{stem}_o{i}']={k:{'shape':list(xa[k].shape),'dtype':str(xa[k].dtype)} for k in xa.files}
   for key in xa.files:
    a,b=xa[key],xb[key];row=metrics(a,b,key,i,kind)
    if kind=='native':
     for folder,data in [(A,a),(B,b)]:assert read(folder/f'state_{i:02d}.json')['tensors'][key]['sha256']==digest(data)
    block=re.fullmatch(r'b(\d+)_(pressure|velocity)',key)
    if block:
     coords=xa[f'b{block[1]}_vertexCoordinates'][0].astype(np.float64)
     centers=(coords[:,:-1,:-1]+coords[:,1:,:-1]+coords[:,:-1,1:]+coords[:,1:,1:])/4
     d=b.astype(np.float64)-a.astype(np.float64);spatial=np.max(np.abs(d[0]),axis=0);ex=np.any(np.abs(d[0])>(1e-5+1e-5*np.abs(b[0])),axis=0)
     iy,ix=np.unravel_index(np.argmax(spatial),spatial.shape);q={'boundary':i,'field':key,'max_center_xy':centers[:,iy,ix].tolist(),'max_grid_yx':[int(iy),int(ix)],'max_abs':float(spatial[iy,ix]),'exceeding_cells':int(ex.sum()),'grid_shape':list(spatial.shape),'native_centers_definition':'arithmetic average of four saved vertices; not exact curved-cell centroid'}
     if ex.any():
      q['exceedance_xy_min']=centers[:,ex].min(axis=1).tolist();q['exceedance_xy_max']=centers[:,ex].max(axis=1).tolist();q['exceedance_grid_yx_min']=np.argwhere(ex).min(axis=0).tolist();q['exceedance_grid_yx_max']=np.argwhere(ex).max(axis=0).tolist()
      q['exceedances_by_x_index']=ex.sum(axis=0).tolist();q['exceedances_by_y_index']=ex.sum(axis=1).tolist()
     localization.append(q)
     if a.shape[1]>1:
      for c in range(a.shape[1]):metrics(a[:,c],b[:,c],key+f'/component{c}',i,'native_component')
    if kind=='sensor' and key=='velocity':
     for c in range(2):metrics(a[:,c],b[:,c],key+f'/component{c}',i,'sensor_component')
   if stem=='state':
    ma,mb=read(A/f'state_{i:02d}.json'),read(B/f'state_{i:02d}.json')
    timeline.append({'observation':i,'clock_S0':ma['clock'],'clock_S2':mb['clock'],'rng_identity':ma['rng']==mb['rng'],'native_differing_fields':[r['field'] for r in rows if r['boundary']==i and r['kind']=='native' and not r['bitwise_equal']]})
chronology=[];outer=[]
for t in range(3):
 ja,jb=read(A/f'planner_{t:02d}.json'),read(B/f'planner_{t:02d}.json');cr={'t':t,'rng_before_equal':ja['rng_before']==jb['rng_before'],'rng_after_equal':ja['rng_after']==jb['rng_after']}
 with np.load(A/f'planner_{t:02d}.npz') as a,np.load(B/f'planner_{t:02d}.npz') as b:
  cr['planner']={k:metrics(a[k],b[k],f'planner/{k}',t,'planner') for k in a.files}
 # Compare ZIP storage payloads without unpickling or importing Torch.
 with zipfile.ZipFile(A/f'planner_rng_{t:02d}.pt') as za,zipfile.ZipFile(B/f'planner_rng_{t:02d}.pt') as zb:
  def storages(z):return {n.split('/',1)[1]:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if '/data/' in n}
  cr['serialized_rng_tensor_storage_equal']=storages(za)==storages(zb)
 aa,bb=read(A/f'actuator_{t:02d}.json'),read(B/f'actuator_{t:02d}.json');assert len(aa)==len(bb)==25
 for j,(a,b) in enumerate(zip(aa,bb)):
  o={'t':t,'outer':j,'gain_S0':a['gain'],'gain_S2':b['gain'],'clock_before_equal':a['clock_before']==b['clock_before'],'clock_after_equal':a['clock_after']==b['clock_after'],'accepted_increment_S0':a['clock_after']['solver_steps']-a['clock_before']['solver_steps'],'accepted_increment_S2':b['clock_after']['solver_steps']-b['clock_before']['solver_steps']}
  o['memory_max_abs']=metrics(a['smoothed_memory_measured'],b['smoothed_memory_measured'],'outer_memory',t*25+j,'actuator')['max_abs']
  for side in ('top','bottom'):
   for phase in ('boundary','boundary_after_solver'):
    for field in ('velocity_measured','flux_measured'):
     m=metrics(a[phase][side][field],b[phase][side][field],f'{phase}/{side}/{field}',t*25+j,'actuator');o[f'{phase}_{side}_{field}_max_abs']=m['max_abs']
  outer.append(o)
 sa,sb=read(A/'steps.json')[t],read(B/'steps.json')[t]
 cr['outcomes']={k:metrics(sa[k],sb[k],k,t,'outcome') for k in ('reward','drag','lift','abs_lift','memory','requested_action','clipped_action')};chronology.append(cr)
# Also inventory reset boundary tensors and compare their measured values.
ra,rb=read(A/'reset_boundary.json'),read(B/'reset_boundary.json')
for side in ('top','bottom'):
 for k in ('velocity_measured','flux_measured'):metrics(ra[side][k],rb[side][k],f'reset/{side}/{k}',0,'reset_actuator')
recorded_fail=[r for r in read(B/'prefix_differences.json') if r.get('allclose') is False or r.get('exact_identity') is False]
summary={'files_verified':len(manifest),'native_array_hashes_verified':sum(len(x) for k,x in inventory.items() if k.startswith('state_'))*2,'time_boundaries':timeline,'recorded_first_failed_gate':recorded_fail,'first_recorded_difference':{'boundary':'o0','native_fields_bitwise_equal':not timeline[0]['native_differing_fields'],'sensor_fields_differ':[x for x in rows if x['boundary']==0 and x['kind']=='sensor' and not x['bitwise_equal']]},'additional_offline_failures':[r for r in rows if r['kind']=='native' and r['exceedance_count']>0],'first_boundary_delivery_difference':next((x for x in outer if any(v>0 for k,v in x.items() if k.endswith('velocity_measured_max_abs'))),None),'all_outer_clock_pairs_identical':all(x['clock_before_equal'] and x['clock_after_equal'] for x in outer),'accepted_increments_seen':sorted(set((x['accepted_increment_S0'],x['accepted_increment_S2']) for x in outer)),'CFD':0,'checkpoint_inference':0,'training':0}
save('comparison.json',rows);save('inventory.json',inventory);save('spatial_localization.json',localization);save('action_planner_chronology.json',chronology);save('outer_actuation_comparison.json',outer);save('verification.json',summary)
with (O/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert all(sha(P/k)==v for k,v in manifest.items())
print(json.dumps({'fields':len(rows),'timeline':timeline,'failures':[(x['boundary'],x['field'],x['max_abs'],x['exceedance_count']) for x in summary['additional_offline_failures']],'first_boundary_diff':summary['first_boundary_delivery_difference'],'b4_o3':next(x for x in localization if x['boundary']==3 and x['field']=='b4_pressure')},indent=2))
