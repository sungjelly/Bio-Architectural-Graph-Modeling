"""Promote the completed exploratory comparison without upgrading its claims."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import yaml
from spatial_benchmark.paths import current_paths
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.result_catalog import validate_result_manifest
from spatial_benchmark.registry import Registry
from run_so12_cancer_comparison import EVALUATION,CAMPAIGN,RUN
from publish_so12_cancer_comparison import verify_final,validate_publication_identity,verify_scvi_registration,require


def promote():
    p=current_paths();root=p.report_root/'analyses/so12_cancer_comparison'/EVALUATION
    manifest=verify_final(root);validate_publication_identity(manifest);s=manifest['summary']
    registry=Registry(p.state_root/'tracking/bagm.sqlite3')
    with registry.connect() as connection:
        evaluation=connection.execute('SELECT run_id,status,artifact_path FROM evaluations WHERE evaluation_id=?',(EVALUATION,)).fetchone()
    require(evaluation is not None and evaluation['run_id']==RUN and evaluation['status']=='completed'
            and Path(evaluation['artifact_path']).resolve()==root.resolve(),'Comparison registry publication is incomplete')
    metrics=json.loads((root/'metrics/pixel_agreement.json').read_text())
    scvi=verify_scvi_registration(root,p,registry)
    # The registration receipt exposes the actual separately registered scVI run.
    scvi_id=scvi['derived_run_id']
    result=p.result_root/'descriptive_analysis/representation_comparison/res_so12_test_cancer_representation_agreement'
    path=result/'result.yaml';r=yaml.safe_load(path.read_text());now=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
    names=[row['Method'] for row in s['primary_scores']]
    observed=[]
    for row in s['primary_scores']:
        observed.append(f"{row['Method']}: oracle {row['Oracle core 15']:.4f}/{row['Oracle core 23']:.4f}, other-core transfer {row['Transfer core 15']:.4f}/{row['Transfer core 23']:.4f}")
    observed_text='Fixed resolution1 seed0 footprint Dice (core15/core23): '+'; '.join(observed)+'. '
    observed_text+='All56 partitions (7 methods ×4 resolutions ×2 Leiden seeds) and both pixel supports retained. '
    observed_text+=f"PCA50 retains {100*s['pca_training_explained_variance']:.3f}% of training feature variance. "
    observed_text+='All-positive footprint baselines are '+ '/'.join(f"{c['all_positive_footprint_dice']:.4f}" for c in s['coverage'])+'.'
    r.update(status='verified',outcome='inconclusive')
    r['method']={'name':'Frozen BAGM h0-h4, training-fitted Pearson PCA50 and raw-count scVI50; common Leiden and multi-cluster pixel Dice',
        'model_family':'Relative-QKV graph encoder, linear PCA, and negative-binomial scVI with slide batch covariate',
        'graph_context':'Joint81607 test-cell cosine k30 graph per representation; L2 rows, unweighted undirected union; converged RBConfiguration at resolutions0.25,0.5,1,2 and seeds0,1',
        'masking_or_perturbation':'All-observed inference;20 within-core label shuffles for primary settings; nominal selected unions held fixed under50µm annotation boundary sensitivity',
        'evaluation_design':'PCA/scVI fitted on same21training cores264491cells; five validation cores only scVI stopping; test15/23 frozen inference. Joint test-cell clustering is transductive. Oracle annotation-selected union and opposite-core-selected union are separate estimands. Single fitted encoder seed0; provenance seeds0/1 denote Leiden, fold0 the fixed split.',
        'implementation_version':'so12_cancer_comparison_v1; immutable checksummed source/environment/geometry/fit/cell/graph receipts'}
    r['conclusion']={'question':'How well can unions of frozen representation clusters match the supplied outlined regions on two held-out SO12 cores?',
        'estimand':'Exploratory approximate-section region agreement on a method-independent segmented-cell footprint, with full-region coverage sensitivity and cross-core cluster-union transfer',
        'observed_result':observed_text,
        'strongest_alternative_explanation':'Annotation-selected unions and finer partitions can improve overlap without learning a transferable cancer identity. Core-specific clusters, morphology inputs, annotated prevalence, section mismatch and unmeasured annotation completeness can drive apparent differences.',
        'controls':['Same21training-core fit policy and1000genes; unchanged BAGM checkpoint194 and frozen Pearson statistics; no annotation inputs to encoder fitting or clustering.',
            'All81,607 canonical cells and original keyed cell polygons; exact h0/h4 and ordinary-prediction reproduction during h1-h3 extraction; independent CPU exact-neighbor audits.',
            'Same graph recipe, four prespecified resolutions and both Leiden seeds; main resolution1 seed0 fixed before scoring.',
            'Exact multicluster union Dice includes all selected outside-outline pixels as false positives; all-positive/empty controls,20 within-core shuffles and opposite-core union selection.',
            'Geometry-only calibrated rotation/scale fitting, reflection comparison, visual review and tissue/landmark diagnostics; source outlines unchanged; fixed-union50µm boundary sensitivity and full-region coverage ceiling.',
            'Compact scVI weights restored from best full validation ELBO, frozen before test inference; separate accurate scVI run and checkpoint catalog; auxiliary validation reconstruction is explicitly not BAGM masked accuracy.'],
        'remaining_uncertainty':'Only two cores, donor overlap and historical exposure; one fitted encoder seed. H&E screenshots lack an authoritative transform, exhaustive-label certification and verified same-section identity. Approximate alignment is strong in outer silhouette but core23 has substantial internal/acellular coverage discrepancy. Native-resolution raster omits641cells with no valid owned pixel and assigns overlaps deterministically. PCA/scVI lack BAGM morphology/graph inputs; scVI also models slide batches. Oracle is optimistic; transfer is between jointly clustered cores; no population significance or superiority claim.',
        'maximum_defensible_claim':s['maximum_defensible_claim']}
    aris=[f"{x['method']}={x['all_cells_ari']:.3f}" for x in metrics['seed_stability'] if x['resolution']==1]
    r['evidence']={
        'predictive_gain':{'status':'not_tested','summary':'Cancer-region clustering agreement does not measure graph-specific masked-expression predictive gain; auxiliary scVI reconstruction is not a matched BAGM prediction benchmark.'},
        'stability':{'status':'mixed','summary':'Primary seed ARI: '+', '.join(aris)+'. All resolutions/seeds and boundary sensitivities retained; no post-hoc pass threshold.'},
        'faithfulness':{'status':'not_applicable','summary':'No attribution or inferred cell-cell interaction interpreted.'},
        'null_calibration':{'status':'mixed','summary':'All-positive prevalence and20 within-core shuffled-label controls reported; these are descriptive controls rather than realistic spatial or biological null validation.'},
        'patient_replication':{'status':'not_tested','summary':'Two cores with donor overlap allowed and historical exposure do not establish independent-patient replication.'},
        'external_support':{'status':'mixed','summary':'Supplied hand-outlined H&E regions provide external morphological comparison, limited by approximate registration, undocumented exhaustiveness and section identity.'},
        'perturbation_support':{'status':'not_tested','summary':'No controlled biological perturbation.'}}
    r['provenance'].update(updated_at_utc=now,run_ids=[RUN,scvi_id],expected_seeds=[0,1],included_seeds=[0,1],
        expected_folds=[0],included_folds=[0],failed_or_excluded_runs=[],sources=[
            {'kind':'evaluation','root':'report','path':str((root/'manifest.json').relative_to(p.report_root)),
             'sha256':sha256_file(root/'manifest.json'),'verification_status':'verified','run_id':RUN},
            {'kind':'report','root':'report','path':str((root/'scvi_run_registration.json').relative_to(p.report_root)),
             'sha256':sha256_file(root/'scvi_run_registration.json'),'verification_status':'verified','run_id':scvi_id}])
    r['files']=[]
    path.write_text(yaml.safe_dump(r,sort_keys=False,allow_unicode=True))
    lines=['# '+r['title'],'',observed_text,'',
        'These numbers are exploratory annotation agreement. Oracle unions are selected on the scored core; transfer unions are selected on the other jointly clustered core. Neither is unbiased cancer-classifier accuracy. Outside-outline predictions are penalized.',
        '',r['conclusion']['remaining_uncertainty'],'',
        f'Complete figures, all resolutions/seeds and machine-readable scores: `{root.relative_to(p.project_root)}`.',
        '',f'BAGM source `{RUN}`; separately cataloged scVI source `{scvi_id}`. No raw/clinical data or scientific model artifacts were deleted for this task.']
    (result/'README.md').write_text('\n'.join(lines)+'\n')
    validate_result_manifest(path,paths=p,verify_payloads=True,verify_sources=True)
    subprocess.run(['/venv/main/bin/python','scripts/results/manage_results.py','validate'],cwd=p.project_root,check=True,
        env=__import__('os').environ)
    subprocess.run(['/venv/main/bin/python','scripts/results/manage_results.py','catalog'],cwd=p.project_root,check=True)
    print(result)


if __name__=='__main__':promote()
