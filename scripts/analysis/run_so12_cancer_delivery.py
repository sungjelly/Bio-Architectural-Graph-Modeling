"""Managed final metrics, comparison figures and portable report generation."""
from contextlib import ExitStack
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
from spatial_benchmark.paths import current_paths
from spatial_benchmark.fingerprints import sha256_file
from run_so12_cancer_comparison import EVALUATION
from run_so12_paper_test_h0_leiden import read,write,wait_child


def main():
    paths=current_paths();out=paths.scratch_root/'active_runs'/EVALUATION
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:stop.set())
    config=read(out/'clustering_summary.json')
    if not config['passed'] or not config['all_methods_complete'] or len(config['partitions'])!=56:
        raise RuntimeError('All 56 frozen partitions must be ready')
    scripts=['so12_cancer_metrics.py','so12_cancer_core_mixing.py','so12_cancer_figures.py','so12_cancer_report.py']
    binding={f'scripts/analysis/{name}':sha256_file(paths.project_root/'scripts/analysis'/name) for name in scripts}
    registration_source='scripts/analysis/so12_cancer_registration.py'
    binding[registration_source]=sha256_file(paths.project_root/registration_source)
    binding['scripts/analysis/run_so12_cancer_delivery.py']=sha256_file(Path(__file__))
    for name in binding:
        target=out/'provenance/stage_sources/delivery'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(paths.project_root/name,target)
    write(out/'provenance/delivery_source_binding.json',binding)
    def event(status,**extra):
        row={'status':status,'time':datetime.now(timezone.utc).isoformat(),**extra}
        write(out/'stages/delivery.json',row)
        with (out/'stages/delivery.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    with ExitStack() as stack:
        lock=stack.enter_context((paths.state_root/'locks/so12-cancer-delivery.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for script in scripts:
                if stop.is_set():raise RuntimeError('Stopped')
                if shutil.disk_usage(out).free<5.1*2**30:raise RuntimeError('Insufficient reserved disk')
                event('running',script=script)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                with (out/'logs/delivery.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u','scripts/analysis/'+script,'--output',str(out)],
                        cwd=paths.project_root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:wait_child(child,stop)
                    finally:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()
                event('script_complete',script=script)
            for name,digest in binding.items():
                if sha256_file(paths.project_root/name)!=digest:raise RuntimeError('Delivery source changed')
            event('complete')
        except BaseException as error:
            event('failed',error=str(error));raise


if __name__=='__main__':main()
