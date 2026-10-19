"""Managed, individually restartable stages for the exploratory cancer comparison."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import traceback
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.fingerprints import sha256_file
from run_so12_paper_test_h0_leiden import write, read, wait_child, record

EVALUATION='ev_so12_cancer_comparison_s0_20261018'
CAMPAIGN='cmp_20261018_so12_cancer_comparison'
RUN='r_20261017T044837Z_caf76350_s000_f00_a01_12df41be'
METHODS=['h4','h3','h2','h1','h0','pca','scvi']


def configuration(paths):
    contract=paths.project_root/f'experiments/campaigns/{CAMPAIGN}/README.md'
    return {'evaluation_id':EVALUATION,'campaign_id':CAMPAIGN,'source_run_id':RUN,
        'schema':'so12_cancer_comparison_v1','checkpoint_epoch':194,'seed':0,
        'dataset_id':'so12_paper_core_split','split_id':'so12_paper_21train_5val_2test_v1',
        'n_cells':81607,'test_cores':[15,23],'methods':METHODS,'resolutions':[.25,.5,1.,2.],
        'leiden_seeds':[0,1],'primary_resolution':1.,'primary_seed':0,'knn':30,'distance':'cosine',
        'symmetrization':'unweighted_undirected_union','leiden_iterations':-1,
        'fit_policy':'same_21_training_cores_validation_only_scvi_stopping_test_frozen',
        'pca':{'components':50,'features':'frozen_train_Pearson','whiten':False},
        'scvi':{'latent':50,'hidden':128,'layers':2,'likelihood':'nb','batch':'slide',
                'max_epochs':100,'patience':15,'batch_size':1024,'seed':0},
        'clinical_sources':{f'Core_{core}.png':record(paths.data_root/f'clinical/Core_{core}.png') for core in [15,23]},
        'contract_sha256_at_start':sha256_file(contract),'minimum_free_gib':5,
        'exploratory':True,'classification':{'lifecycle_stage':'exploratory_screen',
            'study_axis':'annotated_region_representation_comparison','scientific_variant':'train21_pca_scvi_test15_23'}}


def initialize(paths):
    out=paths.scratch_root/'active_runs'/EVALUATION
    registry=Registry(paths.state_root/'tracking/bagm.sqlite3')
    config=configuration(paths)
    with registry.connect() as connection:
        row=connection.execute('SELECT * FROM evaluations WHERE evaluation_id=?',(EVALUATION,)).fetchone()
    if row is None:
        out.mkdir(parents=True,exist_ok=True)
        # Registration exploration may already have written its owned subdirectory.
        if (out/'config.resolved.json').exists():
            raise RuntimeError('Unregistered config exists')
        write(out/'config.resolved.json',config)
        registry.create_campaign(CAMPAIGN,name='SO12 test-core cancer region representation comparison',
            scientific_question='How well do frozen representation cluster unions match outlined regions?',
            config={'contract':f'experiments/campaigns/{CAMPAIGN}/README.md','evaluation_id':EVALUATION})
        registry.create_evaluation(EVALUATION,run_id=RUN,checkpoint_name='best',dataset_id=config['dataset_id'],
            split_id=config['split_id'],status='pending',artifact_path=out,
            metrics={'campaign_id':CAMPAIGN,'stage':'initialized'})
        packages={name:importlib.metadata.version(name) for name in
            ['torch','numpy','pandas','scipy','scikit-learn','igraph','leidenalg','matplotlib','shapely']}
        write(out/'provenance/environment_base.json',{'packages':packages,'python':sys.version})
    else:
        if row['run_id']!=RUN or Path(row['artifact_path']).resolve()!=out.resolve():
            raise RuntimeError('Existing evaluation identity mismatch')
        saved=read(out/'config.resolved.json')
        for key in ['methods','resolutions','leiden_seeds','fit_policy','clinical_sources','scvi','pca']:
            if saved[key]!=config[key]:
                raise RuntimeError(f'Scientific configuration changed: {key}')
    return out,registry


def stage_commands(stage,out):
    common=['--output',str(out)]
    if stage=='layers': return [('/venv/main/bin/python',['scripts/analysis/so12_cancer_layers.py',*common])]
    if stage in ['pca','scvi']:
        return [('/venv/so12-scvi/bin/python',['scripts/analysis/so12_cancer_baselines.py',*common,'--mode',stage])]
    if stage=='graphs':
        return [('/venv/main/bin/python',['scripts/analysis/so12_cancer_clustering.py','graph',*common,'--method',method]) for method in METHODS]
    if stage=='leiden':
        return [('/venv/main/bin/python',['scripts/analysis/so12_cancer_clustering.py','leiden',*common,'--workers','6'])]
    raise ValueError(stage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['initialize','layers','pca','scvi','graphs','leiden'],required=True)
    args=parser.parse_args()
    paths=current_paths()
    lockroot=paths.state_root/'locks'
    lockroot.mkdir(parents=True,exist_ok=True)
    with (lockroot/'so12-cancer-comparison-init.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        out,registry=initialize(paths)
    if args.stage=='initialize': return 0
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda s,f:stop.set())
    stage=args.stage
    def event(status,**kwargs):
        value={'stage':stage,'status':status,'time':datetime.now(timezone.utc).isoformat(),**kwargs}
        write(out/f'stages/{stage}.json',value)
        with (out/f'stages/{stage}.jsonl').open('a') as stream: stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    with ExitStack() as stack:
        for name,mode in [(f'so12-cancer-{stage}.lock',fcntl.LOCK_EX),('bagm-worker.lock',fcntl.LOCK_SH)]:
            handle=stack.enter_context((lockroot/name).open('a')); fcntl.flock(handle,mode|fcntl.LOCK_NB)
        gpu=1 if stage=='scvi' else 0 if stage in ['layers','graphs'] else None
        if gpu is not None:
            handle=stack.enter_context((lockroot/f'bagm-worker-gpu-{gpu}.lock').open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            processes=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            if processes.strip(): raise RuntimeError(f'GPU {gpu} occupied')
        if shutil.disk_usage(out).free < 5.3*2**30: raise RuntimeError('Disk reserve insufficient')
        commands=stage_commands(stage,out)
        binding={}
        for _,command in commands:
            source=paths.project_root/command[0]
            binding[command[0]]=sha256_file(source)
            target=out/'provenance/stage_sources'/stage/command[0]
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
        binding['scripts/analysis/run_so12_cancer_comparison.py']=sha256_file(Path(__file__))
        write(out/f'provenance/{stage}_source_binding.json',binding)
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                 CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',
                 PYTHONPATH=f'{paths.project_root}:{paths.project_root/"src"}:{paths.project_root/"scripts/analysis"}')
        try:
            event('running',gpu=gpu)
            with registry.transaction(immediate=True) as c:
                c.execute("UPDATE evaluations SET status='running' WHERE evaluation_id=? AND run_id=?",(EVALUATION,RUN))
            for python,arguments in commands:
                if stop.is_set(): raise RuntimeError('Stage stopped')
                command=[python,'-u',*arguments]
                log=out/f'logs/{stage}.log'; log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as stream:
                    child=subprocess.Popen(command,cwd=paths.project_root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try: wait_child(child,stop)
                    finally:
                        if child.poll() is None: os.killpg(child.pid,signal.SIGTERM); child.wait()
                if shutil.disk_usage(out).free < 5*2**30: raise RuntimeError('Disk reserve breached')
            for name,digest in binding.items():
                if sha256_file(paths.project_root/name)!=digest: raise RuntimeError(f'Source changed during stage: {name}')
            event('complete',free_bytes=shutil.disk_usage(out).free)
        except BaseException as error:
            event('failed',error=str(error),traceback=traceback.format_exc())
            raise
    return 0


if __name__=='__main__': sys.exit(main())
