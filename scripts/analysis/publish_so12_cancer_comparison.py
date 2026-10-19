"""Verify and publish the completed, manually reviewed SO12 comparison."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.checkpoint_catalog import build_checkpoint_catalog,verify_checkpoint_record
from spatial_benchmark.run_archive import RunArchive,verify_run_bundle
from run_so12_paper_test_h0_leiden import read,write,record
from run_so12_cancer_comparison import EVALUATION,CAMPAIGN,RUN
from so12_paper_test_h0_extraction import verify_source


def require(condition,message):
    if not condition: raise RuntimeError(message)


REQUIRED_FIGURES=('core_15_five_panel','core_23_five_panel','combined_five_panel',
                  'all_representations_2x7','resolution_agreement','oracle_union_errors')


def validate_publication_identity(manifest):
    require(manifest.get('evaluation_id')==EVALUATION and manifest.get('campaign_id')==CAMPAIGN
            and manifest.get('source_run_id')==RUN and manifest.get('status')=='complete'
            and manifest.get('passed') is True,'Published comparison identity differs')
    summary=manifest['summary']
    require(summary.get('evaluation_id')==EVALUATION and summary.get('source_run_id')==RUN
            and summary.get('n_cells')==81607 and summary.get('test_cores')==[15,23],
            'Published comparison summary identity differs')


def verify_scvi_registration(output,paths,registry=None):
    receipt=read(output/'scvi_run_registration.json')
    require(receipt.get('status')=='complete' and receipt.get('model_family')=='scvi'
            and receipt.get('source_bagm_run_id')==RUN and receipt.get('comparison_evaluation_id')==EVALUATION
            and receipt.get('sealed_test_claim') is False,'scVI canonical run/catalog publication incomplete')
    run_id=receipt['derived_run_id']
    require(run_id!=RUN,'scVI checkpoint cannot belong to the BAGM run')
    archive=RunArchive.artifact_path_for(run_id,paths)
    require(Path(receipt['artifact_path']).resolve()==archive.resolve()
            and Path(receipt['checkpoint_path']).resolve()==(archive/'checkpoints/best.ckpt').resolve(),
            'scVI canonical checkpoint path differs')
    verified=verify_run_bundle(archive)
    require(verified['valid'] is True and verified['status']=='success','scVI native run archive invalid')
    registry=registry or Registry(paths.state_root/'tracking/bagm.sqlite3')
    run=registry.get_run(run_id)
    require(run is not None and run['status']=='completed' and run['model_family']=='scvi'
            and Path(run['artifact_path']).resolve()==archive.resolve(),'scVI registry run is not completed')
    records=build_checkpoint_catalog(registry,paths,run_ids=[run_id])
    require(len(records)==1 and records[0]['model_name']=='scvi','scVI checkpoint catalog identity differs')
    checkpoint=verify_checkpoint_record(records[0],paths=paths)
    manifest=read(output/'scvi_manifest.json')
    require(checkpoint['checkpoint_sha256']==receipt['checkpoint_sha256']==manifest['model_sha256']
            and checkpoint['best_epoch']==receipt['selected_epoch_zero_based']==manifest['selected_epoch_zero_based'],
            'scVI selected checkpoint differs across archive and comparison')
    require(run['primary_metric_value']==receipt['validation_negative_elbo']==manifest['validation_elbo'],
            'scVI selected validation loss differs across archive and comparison')
    return receipt


def verify_core_mixing(output,summary):
    require(summary.get('status')=='complete' and summary.get('passed') is True
            and summary.get('n_cells')==81607 and set(summary.get('methods',[]))=={'h0','h1','h2','h3','h4','pca','scvi'}
            and summary.get('resolution')==1. and summary.get('seed')==0,'Incomplete core-mixing stage')
    for field in ['input_sha256','output_sha256']:
        require(bool(summary[field]),'Missing core-mixing checksum records')
        for relative,digest in summary[field].items():
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                    'Unsafe core-mixing local path')
            require(sha256_file(output/relative)==digest,f'Core-mixing bytes changed: {relative}')
    for item in summary['graph_sources'].values():
        path=output/item['path'] if item['relative'] else Path(item['path'])
        require(record(path)=={'sha256':item['sha256'],'size_bytes':item['size_bytes']},
                'Core-mixing graph source changed')


def publication_sources(paths):
    return sorted({*paths.project_root.glob('scripts/analysis/*so12_cancer*.py'),
                   *paths.project_root.glob('ops/supervisor/*so12-cancer*'),
                   paths.project_root/'scripts/analysis/register_so12_scvi_run.py'})


def validate(output):
    paths=current_paths()
    source=verify_source(paths.artifact_root/'runs/2026/10'/RUN,paths)
    config=read(output/'config.resolved.json')
    require(config['evaluation_id']==EVALUATION and config['source_run_id']==RUN,'Analysis identity changed')
    for name,expected in config['clinical_sources'].items():
        require(record(paths.data_root/'clinical'/name)==expected,'Clinical source changed')
    require(read(output/'manual_visual_review.json')['passed'] is True,'Manual figure review is required')
    summaries={name:read(output/name) for name in ['layers_manifest.json','pca_manifest.json','scvi_manifest.json',
        'clustering_summary.json','metrics_summary.json','core_mixing_summary.json',
        'figures/figure_manifest.json','report_manifest.json','summary.json']}
    require(all(value.get('passed') is True and value.get('status')=='complete' for value in summaries.values()),'Incomplete stages')
    cells=output/'cells.csv.gz';cells_sha=sha256_file(cells)
    frame=pd.read_csv(cells,float_precision='round_trip')
    require(frame.shape[0]==81607 and frame.core_number.value_counts().to_dict()=={23:43462,15:38145},'Cell coverage changed')
    require(np.array_equal(frame.row_index,np.arange(81607)) and not frame.duplicated(['slide','fov','cell_ID']).any(),'Cell identities changed')
    registration=read(output/'registration/registration_manifest.json')
    require(registration['passed'] and registration['status']=='verified_approximate_section','Registration unavailable')
    require(registration['canonical_cells']['sha256']==cells_sha,'Registration cell order differs')
    require(sha256_file(Path(registration['canonical_cells']['path']))==cells_sha,'Canonical reference cells changed')
    require(sha256_file(Path(registration['raw_polygons']['path']))==registration['raw_polygons']['sha256'],'Raw cell polygons changed')
    for row in registration['cores']:
        require(sha256_file(output/row['raster_path'])==row['raster_sha256'],'Registered raster changed')
        require(sha256_file(paths.data_root/f"clinical/Core_{row['core_number']}.png")==row['image']['sha256'],
                'Registration clinical source changed')
    for name in ['layers_manifest.json','pca_manifest.json','scvi_manifest.json']:
        receipt=summaries[name]
        require(receipt['n_cells']==81607 and receipt['cells_sha256']==cells_sha,'Representation cell coverage/order differs')
        require(receipt['checkpoint_sha256']==source['checkpoint_sha256'] and receipt['fitted_parameters_sha256']==source['fitted_parameters_sha256'],
                'Selected source or frozen normalization changed')
        if name!='layers_manifest.json':
            require(receipt['test_used_for_fitting'] is False and receipt['training_cells']==264491,'Baseline fit leakage')
            require(sha256_file(output/receipt['embedding_path'])==receipt['embedding_sha256'],'Baseline array changed')
        else:
            for item in receipt['embeddings'].values():
                require(sha256_file(output/item['path'])==item['sha256'],'Layer array changed')
    for method in ['h1','h2','h3','pca','scvi']:
        array=np.load(output/f'embeddings/{method}.npy',mmap_mode='r',allow_pickle=False)
        require(array.shape==(81607,256 if method.startswith('h') else 50) and array.dtype==np.float32 and np.isfinite(array).all(),'Embedding schema invalid')
    require(sha256_file(output/'baselines/pca_model.npz')==summaries['pca_manifest.json']['fitted_model_sha256'],'Fitted PCA model changed')
    require(sha256_file(output/'baselines/scvi_model/model.pt')==summaries['scvi_manifest.json']['model_sha256'],'Fitted scVI model changed')
    clustering=summaries['clustering_summary.json']
    expected={(m,r,s) for m in ['h0','h1','h2','h3','h4','pca','scvi'] for r in [.25,.5,1.,2.] for s in [0,1]}
    require({(r['method'],r['resolution'],r['seed']) for r in clustering['partitions']}==expected and len(clustering['partitions'])==56,'Partition coverage differs')
    for row in clustering['partitions']:
        rec=row['labels'];require(rec['relative'] and sha256_file(output/rec['path'])==rec['sha256'],'Partition changed')
        labels=np.load(output/rec['path'],allow_pickle=False)
        require(labels.shape==(81607,) and np.issubdtype(labels.dtype,np.integer) and np.all(labels>=0),'Invalid labels')
    metrics=read(output/'metrics/pixel_agreement.json')
    require(metrics['pixel_scores_available'] and len(metrics['results'])==56,'Incomplete pixel scores')
    for relative,digest in metrics['input_sha256'].items():
        require(sha256_file(output/relative)==digest,f'Metric input changed: {relative}')
    for relative,digest in summaries['metrics_summary.json']['output_sha256'].items():
        require(sha256_file(output/relative)==digest,f'Metric output changed: {relative}')
    verify_core_mixing(output,summaries['core_mixing_summary.json'])
    verify_scvi_registration(output,paths)
    figures=summaries['figures/figure_manifest.json']
    require(figures['registration_manifest_sha256']==sha256_file(output/'registration/registration_manifest.json')
            and figures['clustering_manifest_sha256']==sha256_file(output/'clustering_summary.json'),'Figure inputs changed')
    require(sha256_file(output/figures['color_keys']['path'])==figures['color_keys']['sha256'],'Color key changed')
    for representations in figures['figures'].values():
        for item in representations.values():
            require(sha256_file(output/'figures'/item['path'])==item['sha256'],'Figure bytes changed')
    require(sha256_file(output/summaries['report_manifest.json']['report_path'])==summaries['report_manifest.json']['report_sha256'],
            'Report bytes changed')
    for relative,digest in summaries['summary.json']['source_sha256'].items():
        require(sha256_file(output/relative)==digest,f'Summary source changed: {relative}')
    for name in REQUIRED_FIGURES:
        for ext in ['png','pdf']: require((output/f'figures/{name}.{ext}').stat().st_size>1000,'Missing figure')
    for item in output.glob('provenance/*_source_binding.json'):
        for relative,digest in read(item).items():
            require(sha256_file(paths.project_root/relative)==digest,f'Stage source changed: {relative}')
    require(shutil.disk_usage(output).free>=5*2**30,'Disk reserve breached')
    return {'passed':True,'status':'complete','verified_at':datetime.now(timezone.utc).isoformat(),
        'source_epoch':source['checkpoint_epoch'],'checkpoint_sha256':source['checkpoint_sha256'],
        'cells_sha256':cells_sha,'partitions':56,'n_cells':81607,'clinical_inputs_unchanged':True}


def verify_final(final):
    manifest=read(final/'manifest.json')
    require(read(final/'_SUCCESS')['manifest_sha256']==sha256_file(final/'manifest.json'),'Completion marker changed')
    actual={str(p.relative_to(final)) for p in final.rglob('*') if p.is_file()}-{'manifest.json','_SUCCESS'}
    require(not any(p.is_symlink() for p in final.rglob('*')),'Symlink in publication')
    require(set(manifest['files'])==actual,'Incomplete final inventory')
    for name,expected in manifest['files'].items(): require(record(final/name)==expected,f'Published bytes changed: {name}')
    return manifest


def publish():
    paths=current_paths();out=paths.scratch_root/'active_runs'/EVALUATION
    final=paths.report_root/'analyses/so12_cancer_comparison'/EVALUATION
    registry=Registry(paths.state_root/'tracking/bagm.sqlite3')
    if final.exists():
        require(not out.exists(),'Both active and final outputs exist');manifest=verify_final(final)
        verify_scvi_registration(final,paths,registry)
    else:
        validation=validate(out);write(out/'verification.json',validation)
        for path in publication_sources(paths):
            target=out/'provenance/final_sources'/path.relative_to(paths.project_root)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        for name in ['installation_provenance.json','dependency_verification.json']:
            shutil.copyfile(Path('/venv/so12-scvi')/name,out/'provenance'/name)
        shutil.copyfile(paths.project_root/f'experiments/campaigns/{CAMPAIGN}/README.md',out/'provenance/task_contract_at_publication.md')
        write(out/'provenance/publication_environment.json',{'python_packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','scikit-learn','igraph','leidenalg','matplotlib']},
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=paths.project_root,text=True).strip(),
            'free_bytes':shutil.disk_usage(out).free})
        files={str(p.relative_to(out)):record(p) for p in sorted(out.rglob('*')) if p.is_file() and str(p.relative_to(out)) not in ['manifest.json','_SUCCESS']}
        manifest={'evaluation_id':EVALUATION,'campaign_id':CAMPAIGN,'source_run_id':RUN,'status':'complete','passed':True,
            'summary':read(out/'summary.json'),'verification':validation,'files':files}
        write(out/'manifest.json',manifest);write(out/'_SUCCESS',{'manifest_sha256':sha256_file(out/'manifest.json')})
        final.parent.mkdir(parents=True,exist_ok=True);out.rename(final);verify_final(final)
    validate_publication_identity(manifest)
    stamp=datetime.now(timezone.utc).isoformat()
    with registry.transaction(immediate=True) as c:
        existing=c.execute('SELECT sha256 FROM artifacts WHERE evaluation_id=? AND path=?',(EVALUATION,str(final/'manifest.json'))).fetchall()
        require(all(row['sha256']==sha256_file(final/'manifest.json') for row in existing),'Registered manifest changed')
        if not existing:
            c.execute('INSERT INTO artifacts(run_id,evaluation_id,kind,path,sha256,size_bytes,status,created_at) VALUES(?,?,?,?,?,?,?,?)',
                (RUN,EVALUATION,'so12_cancer_comparison',str(final/'manifest.json'),sha256_file(final/'manifest.json'),(final/'manifest.json').stat().st_size,'present',stamp))
        changed=c.execute("UPDATE evaluations SET status='completed',artifact_path=?,metrics_json=?,finished_at=? WHERE evaluation_id=? AND run_id=?",
            (str(final),json.dumps(manifest['summary'],allow_nan=False),stamp,EVALUATION,RUN))
        require(changed.rowcount==1,'Evaluation registry row missing')
        c.execute("UPDATE campaigns SET status='completed',updated_at=? WHERE campaign_id=?",(stamp,CAMPAIGN))
    print(json.dumps({'status':'complete','path':str(final),'manifest_sha256':sha256_file(final/'manifest.json')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.verify_only:
        p=current_paths();final=p.report_root/'analyses/so12_cancer_comparison'/EVALUATION
        print(json.dumps({'verified':True,'file_count':len(verify_final(final)['files'])}))
    else: publish()
