from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

parser=argparse.ArgumentParser(description='Independently audit fixed SO12 cluster-union pixel metrics; no fitting or selection changes.')
parser.add_argument('--output',type=Path,required=True)
OUT=parser.parse_args().output.resolve()
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
metrics=read(OUT/'metrics/pixel_agreement.json')
receipt=read(OUT/'metrics_summary.json')
assert metrics['passed'] and metrics['status']=='complete' and receipt['passed'] and receipt['status']=='complete'
assert len(metrics['results'])==56
clustering=read(OUT/'clustering_summary.json')
registration=read(OUT/'registration/registration_manifest.json')
parts={(r['method'],r['resolution'],r['seed']):r for r in clustering['partitions']}
counts={};sensitive={};checked=0;optimality_checks=0
for r in registration['cores']:
 core=r['core_number'];f=OUT/r['raster_path'];assert sha(f)==r['raster_sha256']
 with np.load(f,allow_pickle=False) as a:
  owner,roi,valid=a['owner'],a['roi'],a['valid']
  def count(roi,valid):
   foot=valid&(owner>=0)
   return (np.bincount(owner[foot],minlength=81607),np.bincount(owner[foot&roi],minlength=81607),int((valid&roi).sum()))
  counts[core]=count(roi,valid)
  sensitive[core]={'exclude_boundary_50um':count(roi,a['valid50']),
   'erode_roi_50um':count(a['roi_eroded50'],valid),'dilate_roi_50um':count(a['roi_dilated50'],valid)}
def eq(actual,expected):
 if expected is None:assert actual is None
 else:assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),(actual,expected)
def grouped(labels,c):
 return np.bincount(labels,weights=c[0]).astype(np.int64),np.bincount(labels,weights=c[1]).astype(np.int64)
def check_scores(r,tp,area,positive):
 global checked
 assert (r['tp'],r['fp'],r['fn'])==(tp,area-tp,positive-tp)
 assert r['predicted_pixels']==area and r['positive_pixels']==positive
 eq(r['dice_f1'],2*tp/(area+positive) if positive else None)
 eq(r['iou'],tp/(area+positive-tp) if positive else None)
 eq(r['precision'],tp/area if area else None)
 eq(r['recall'],tp/positive if positive else None)
 checked+=1
def check_union(r,selected,area,positive,c):
 ix=np.asarray(selected,dtype=np.int64)
 tp=int(positive[ix].sum());pred=int(area[ix].sum())
 check_scores(r['footprint'],tp,pred,int(positive.sum()))
 check_scores(r['full_region'],tp,pred,c[2])
methods=['h0','h1','h2','h3','h4','pca','scvi'];primary=[];ranges=[];resolutions=[];nulls=[];sensitivity=[]
for row in metrics['results']:
 key=(row['method'],row['resolution'],row['seed']);part=parts[key]
 f=OUT/part['labels']['path'];assert sha(f)==part['labels']['sha256']
 labels=np.load(f,allow_pickle=False);assert labels.shape==(81607,)
 ev=row['evaluation']
 for core in (15,23):
  c=counts[core];area,positive=grouped(labels,c);own=ev['cores'][str(core)]
  selected=np.asarray(own['oracle']['selected_clusters'],dtype=np.int64)
  check_union(own['oracle'],selected,area,positive,c)
  check_union(own['all_positive'],np.flatnonzero(area),area,positive,c)
  check_union(own['empty'],[],area,positive,c)
  # Exact subset optimality certificate: include iff cluster purity >= F1/2.
  tp=int(positive[selected].sum());den=int(positive.sum()+area[selected].sum())
  present=area>0;chosen=np.zeros(len(area),bool);chosen[selected]=True
  difference=positive*den-area*tp
  assert np.all(difference[present&chosen]>=0) and np.all(difference[present&~chosen]<=0)
  optimality_checks+=1
  other=23 if core==15 else 15;t=ev['transfer'][f'{other}_to_{core}']
  assert t['selected_clusters']==ev['cores'][str(other)]['oracle']['selected_clusters']
  assert t['selection_uses_target_annotation'] is False
  check_union(t,t['selected_clusters'],area,positive,c)
  if row['resolution']==1 and row['seed']==0:
   for scenario,sc in sensitive[core].items():
    sa,sp=grouped(labels,sc);sr=ev['fixed_union_boundary_sensitivity'][str(core)][scenario]
    assert sr['selection_frozen_from_nominal_masks'] is True
    check_union(sr['nominal_oracle_union'],selected,sa,sp,sc)
    check_union(sr['transferred_union'],t['selected_clusters'],sa,sp,sc)
    sensitivity.append({'method':row['method'],'core':core,'scenario':scenario,
     'oracle_f1':sr['nominal_oracle_union']['footprint']['dice_f1'],
     'transfer_f1':sr['transferred_union']['footprint']['dice_f1']})
 eq(ev['equal_core_oracle_dice'],np.mean([ev['cores'][str(c)]['oracle']['footprint']['dice_f1'] for c in (15,23)]))
 eq(ev['equal_core_transfer_dice'],np.mean([ev['transfer'][x]['footprint']['dice_f1'] for x in ('15_to_23','23_to_15')]))
 if row['resolution']==1 and row['seed']==0:
  p={'method':row['method'],'clusters':row['cluster_count'],'oracle_f1':{str(c):ev['cores'][str(c)]['oracle']['footprint']['dice_f1'] for c in (15,23)},
   'transfer_f1':{str(c):ev['transfer'][f'{23 if c==15 else 15}_to_{c}']['footprint']['dice_f1'] for c in (15,23)},
   'oracle_full_region_f1':{str(c):ev['cores'][str(c)]['oracle']['full_region']['dice_f1'] for c in (15,23)},
   'transfer_full_region_f1':{str(c):ev['transfer'][f'{23 if c==15 else 15}_to_{c}']['full_region']['dice_f1'] for c in (15,23)},
   'equal_core_oracle_f1':ev['equal_core_oracle_dice'],'equal_core_transfer_f1':ev['equal_core_transfer_dice'],
   'unseen_target_pixel_fraction':{str(c):ev['transfer'][f'{23 if c==15 else 15}_to_{c}']['unseen_target_pixel_fraction'] for c in (15,23)}}
  primary.append(p)
  control=ev['shuffled_label_control'];assert control['repeats']==20 and len(control['results'])==20
  summary={'method':row['method'],'repeats':20}
  for kind in ('oracle','transfer'):
   a=np.asarray([x[f'equal_core_{kind}_dice'] for x in control['results']])
   summary[kind]={'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),'sample_sd':float(a.std(ddof=1)),
                 'observed':ev[f'equal_core_{kind}_dice']}
  nulls.append(summary)
for method in methods:
 rows=[r for r in metrics['results'] if r['method']==method]
 ranges.append({'method':method,'oracle_equal_core_minmax':[min(r['evaluation']['equal_core_oracle_dice'] for r in rows),max(r['evaluation']['equal_core_oracle_dice'] for r in rows)],
  'transfer_equal_core_minmax':[min(r['evaluation']['equal_core_transfer_dice'] for r in rows),max(r['evaluation']['equal_core_transfer_dice'] for r in rows)]})
 for resolution in (.25,.5,1.,2.):
  rows_at=[r for r in rows if r['resolution']==resolution]
  resolutions.append({'method':method,'resolution':resolution,'clusters':[r['cluster_count'] for r in rows_at],
   'oracle_equal_core_by_seed':[r['evaluation']['equal_core_oracle_dice'] for r in rows_at],
   'transfer_equal_core_by_seed':[r['evaluation']['equal_core_transfer_dice'] for r in rows_at]})
first=metrics['results'][0]['evaluation']
coverage={str(c):first['cores'][str(c)]['coverage'] for c in (15,23)}
result={'status':'complete','passed':True,'review_scope':'Independent recomputation from fixed labels and method-independent rasters; no fitting, tuning, or source modification.',
 'metrics_sha256':sha(OUT/'metrics/pixel_agreement.json'),'metrics_summary_sha256':sha(OUT/'metrics_summary.json'),
 'calculation_audit':{'binary_score_records_checked':checked,'oracle_subset_optimality_certificates':optimality_checks,
  'partition_count':56,'no_discrepancies':True},'primary_resolution':1.,'primary_seed':0,'primary':primary,
 'all_positive_footprint_f1':{str(c):first['cores'][str(c)]['all_positive']['footprint']['dice_f1'] for c in (15,23)},
 'all_positive_full_region_f1':{str(c):first['cores'][str(c)]['all_positive']['full_region']['dice_f1'] for c in (15,23)},
 'coverage':coverage,'ranges_all_prespecified_settings':ranges,'resolution_seed_results':resolutions,
 'shuffled_controls':nulls,'primary_fixed_union_boundary_sensitivity':sensitivity,
 'seed_ari':[s for s in metrics['seed_stability'] if s['resolution']==1],
 'caveats':['Oracle unions use scored annotations; descriptive upper bound, not unbiased accuracy.',
  'Cross-core transfer is transductive: both test cores jointly clustered, and unseen source clusters remain negative.',
  'Only two cores and one fitted encoder seed; no population superiority inference.',
  'Approximate section registration, incomplete/exhaustiveness-unverified reference outlines; outside outline treated reference-negative.',
  'Cell-label shuffles destroy spatial autocorrelation and are not realistic biological or spatial nulls.',
  'BAGM has morphology/graph inputs; PCA/scVI are expression-only, and scVI explicitly models slide batches.']}
result['audit_script_sha256']=sha(Path(__file__))
review_path=OUT/'independent_metrics_review.json'
if review_path.exists():
 assert read(review_path)==result,'An existing independent review differs; refusing overwrite'
else:
 with review_path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ['calculation_audit','primary','all_positive_footprint_f1','ranges_all_prespecified_settings','shuffled_controls','seed_ari']},indent=2))
