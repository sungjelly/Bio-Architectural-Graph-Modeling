"""Registered execution and publication of backward cluster-removal curves."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from run_so12_cancer_limited_union import read, write, record, require, verify_inventory, PARENT_SHA, PARENT_ID, RUN, METHODS

EVALUATION = 'ev_so12_cancer_backward_union_s0_20261018'
CAMPAIGN = 'cmp_20261018_so12_cancer_backward_union'


def locations():
    p = current_paths()
    return p, p.report_root/'analyses/so12_cancer_comparison'/PARENT_ID, p.scratch_root/'active_runs'/EVALUATION, p.report_root/'analyses/so12_cancer_backward_union'/EVALUATION


def sources(p):
    return sorted({*p.project_root.glob('scripts/analysis/*so12_cancer_backward*.py'),
        *p.project_root.glob('tests/**/test_so12_cancer_backward*.py'),
        *p.project_root.glob('ops/supervisor/*so12-cancer-backward*'),
        *[p.project_root/'scripts/analysis'/name for name in ['run_so12_cancer_limited_union.py', 'so12_cancer_limited_metrics.py', 'so12_cancer_metrics.py']]})


def initialize():
    p,parent,out,final = locations()
    require(not final.exists(), 'Already published; use a new evaluation identity to reproduce')
    require(shutil.disk_usage(p.project_root).free >= 5.15*2**30, 'Disk reserve insufficient')
    parent_manifest = verify_inventory(parent, PARENT_SHA)
    registry = Registry(p.state_root/'tracking/bagm.sqlite3')
    with registry.connect() as c:
        existing=c.execute('SELECT run_id,artifact_path FROM evaluations WHERE evaluation_id=?',(EVALUATION,)).fetchone()
    if existing:
        require(existing['run_id']==RUN and Path(existing['artifact_path'])==out, 'Evaluation identity collision')
        return out,registry
    require(not out.exists(), 'Unregistered active output exists')
    out.mkdir(parents=True)
    contract=p.project_root/f'experiments/campaigns/{CAMPAIGN}/README.md'
    write(out/'config.resolved.json',{'schema':'so12_cancer_backward_union_v1','evaluation_id':EVALUATION,'campaign_id':CAMPAIGN,
        'source_run_id':RUN,'parent_path':str(parent),'parent_manifest_sha256':PARENT_SHA,'methods':METHODS,
        'resolutions':[.25,.5,1.,2.],'seeds':[0,1],'primary_resolution':1.,'primary_seed':0,'n_cells':81607,'cores':[15,23],
        'selection':'start every positive-overlap cluster; remove one maximizing remaining pixel F1; exact ties lexicographic retained IDs; continue empty',
        'best_path_tie':'fewest clusters then IDs','cell_retention_definition':'at least one valid owned pixel inside ROI; denominator fixed',
        'majority_cell_sensitivity':'at least half valid owned area inside ROI','retention_constraints_pct':[90,95],
        'exploratory':True,'classification':{'lifecycle_stage':'exploratory_screen','study_axis':'backward_cluster_retention','scientific_variant':'fixed_pixel_f1_unweighted_cell_retention'},
        'split_id':'so12_paper_21train_5val_2test_v1','contract_at_start':record(contract),'minimum_free_gib':5})
    shutil.copyfile(contract,out/'task_contract_at_start.md')
    write(out/'provenance/parent_before.json',{'passed':True,'files_verified':len(parent_manifest['files']),'manifest_sha256':PARENT_SHA})
    write(out/'provenance/environment.json',{'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','matplotlib','scipy','scikit-learn']},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=p.project_root).strip(),
        'compute':'CPU only; small integer deletion searches and plots, no GPU-capable training','thread_limit':4,'free_bytes':shutil.disk_usage(out).free})
    (out/'provenance/git_status_at_start.txt').write_text(subprocess.check_output(['git','status','--porcelain=v1'],text=True,cwd=p.project_root))
    registry.create_campaign(CAMPAIGN,name='Backward cluster removal and annotated-cell retention',scientific_question='How does F1 change as complete clusters and annotated-region cells are removed?',config={'evaluation_id':EVALUATION,'contract':str(contract.relative_to(p.project_root))})
    registry.create_evaluation(EVALUATION,run_id=RUN,checkpoint_name='best',dataset_id='so12_paper_core_split',split_id='so12_paper_21train_5val_2test_v1',status='pending',artifact_path=out,metrics={'campaign_id':CAMPAIGN,'stage':'initialized'})
    return out,registry


def execute():
    p,parent,out,final=locations();_,registry=initialize()
    lockpath=p.state_root/'locks/so12-cancer-backward-union.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (out/'execution.json').exists(),'Execution already complete')
        binding={}
        for source in sources(p):
            relative=source.relative_to(p.project_root);binding[str(relative)]=record(source)
            target=out/'provenance/execution_sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        write(out/'provenance/source_binding.json',binding)
        commands=[[sys.executable,'-u',f'scripts/analysis/{name}.py','--parent',str(parent),'--output',str(out)] for name in ['so12_cancer_backward_metrics','audit_so12_cancer_backward_union','so12_cancer_backward_report']]
        write(out/'provenance/commands.json',{'cwd':str(p.project_root),'commands':commands})
        with registry.transaction(immediate=True) as c:c.execute("UPDATE evaluations SET status='running' WHERE evaluation_id=?",(EVALUATION,))
        started=time.monotonic();(out/'logs').mkdir(exist_ok=True)
        try:
            for index,command in enumerate(commands):
                name=Path(command[2]).stem;step=time.monotonic();print(json.dumps({'stage':name,'status':'running'}),flush=True)
                with (out/f'logs/{index}_{name}.log').open('a') as log:subprocess.run(command,cwd=p.project_root,stdout=log,stderr=subprocess.STDOUT,check=True)
                write(out/f'stages/{name}.json',{'passed':True,'runtime_seconds':time.monotonic()-step})
                require(shutil.disk_usage(out).free>=5*2**30,'Disk reserve breached')
                print(json.dumps({'stage':name,'status':'complete'}),flush=True)
            for relative,expected in binding.items():require(record(p.project_root/relative)==expected,f'Source changed: {relative}')
            verify_inventory(parent,PARENT_SHA)
            write(out/'execution.json',{'passed':True,'status':'complete','runtime_seconds':time.monotonic()-started,'peak_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'free_bytes':shutil.disk_usage(out).free,'failures':[]})
        except BaseException as error:
            write(out/f'failures/{time.time_ns()}.json',{'error':str(error),'traceback':traceback.format_exc()});raise


def publish():
    p,parent,out,final=locations();registry=Registry(p.state_root/'tracking/bagm.sqlite3')
    if not final.exists():
        for name in ['execution.json','focused_tests.json','independent_metrics_review.json','manual_visual_review.json','independent_visual_review.json','report_manifest.json']:
            require(read(out/name)['passed'],f'Incomplete check: {name}')
        metrics=read(out/'metrics/backward_union.json');require(metrics['passed'] and len(metrics['results'])==56,'Metric coverage incomplete')
        for relative,digest in read(out/'backward_metrics_summary.json')['output_sha256'].items():require(record(out/relative)['sha256']==digest,f'Metric output changed: {relative}')
        audit=read(out/'independent_metrics_review.json');require(audit['metrics_sha256']==record(out/'metrics/backward_union.json')['sha256'],'Audit source changed')
        report=read(out/'report_manifest.json')
        require(record(out/report['report_path'])==report['report_record'],'Report changed')
        for item in report['figures'].values():
            for file in item.values():require(record(out/file['path'])=={'sha256':file['sha256'],'size_bytes':file['size_bytes']},'Figure changed')
        for relative,expected in read(out/'provenance/source_binding.json').items():require(record(p.project_root/relative)==expected,f'Source changed: {relative}')
        verify_inventory(parent,PARENT_SHA);require(shutil.disk_usage(out).free>=5*2**30,'Disk reserve breached')
        require(sum(x.stat().st_size for x in out.rglob('*') if x.is_file())<100*2**20,'Storage budget exceeded')
        shutil.copyfile(p.project_root/f'experiments/campaigns/{CAMPAIGN}/README.md',out/'task_contract_at_publication.md')
        files={str(x.relative_to(out)):record(x) for x in sorted(out.rglob('*')) if x.is_file() and str(x.relative_to(out)) not in ['manifest.json','_SUCCESS']}
        write(out/'manifest.json',{'evaluation_id':EVALUATION,'campaign_id':CAMPAIGN,'source_run_id':RUN,'passed':True,'status':'complete','parent_manifest_sha256':PARENT_SHA,'summary':read(out/'summary.json'),'files':files})
        write(out/'_SUCCESS',{'manifest_sha256':record(out/'manifest.json')['sha256']})
        final.parent.mkdir(parents=True,exist_ok=True);out.rename(final)
    manifest=verify_inventory(final);require(not out.exists() and manifest['evaluation_id']==EVALUATION,'Publication identity differs')
    now=datetime.now(timezone.utc).isoformat();digest=record(final/'manifest.json')['sha256']
    with registry.transaction(immediate=True) as c:
        existing=c.execute('SELECT sha256 FROM artifacts WHERE evaluation_id=?',(EVALUATION,)).fetchall();require(all(x['sha256']==digest for x in existing),'Artifact identity differs')
        if not existing:c.execute('INSERT INTO artifacts(run_id,evaluation_id,kind,path,sha256,size_bytes,status,created_at) VALUES(?,?,?,?,?,?,?,?)',(RUN,EVALUATION,'so12_cancer_backward_union',str(final/'manifest.json'),digest,(final/'manifest.json').stat().st_size,'present',now))
        changed=c.execute("UPDATE evaluations SET status='completed',artifact_path=?,metrics_json=?,finished_at=? WHERE evaluation_id=? AND run_id=?",(str(final),json.dumps(manifest['summary']),now,EVALUATION,RUN));require(changed.rowcount==1,'Missing evaluation')
        c.execute("UPDATE campaigns SET status='completed',updated_at=? WHERE campaign_id=?",(now,CAMPAIGN))
    print(json.dumps({'path':str(final),'manifest_sha256':digest}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['initialize','execute','publish','verify'],required=True);args=parser.parse_args()
    if args.stage=='initialize':initialize()
    elif args.stage=='execute':execute()
    elif args.stage=='publish':publish()
    else:print(json.dumps({'files_verified':len(verify_inventory(locations()[3])['files'])}))
