"""One reviewed exact-file cleanup; never scans/selects additional payloads."""
from pathlib import Path
import hashlib,json,os,shutil,sqlite3,subprocess
from datetime import datetime,timezone
base=Path(__file__).resolve().parent
root=base.parents[2]
assert root == Path('/workspace/BAGM')
audit=base/'audit.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
assert sha(audit)=='9d5c2b0c877b8f6163dc76c7c7afe7cd4b62bf72aa3ee78829e50dfd3b28eb59'
a=json.loads(audit.read_text())
assert not (base/'application_receipt.json').exists()
protected=[Path(v['path']) for v in a['protected']]
records=a['files']; names={r['path'] for r in records}
# Fail closed if any selected payload is open by any live process.
for proc in Path('/proc').glob('[0-9]*/fd'):
 try:
  for fd in proc.iterdir():
   try: target=os.readlink(fd)
   except OSError: continue
   assert target not in names, 'Candidate open by live process'
 except (PermissionError,FileNotFoundError): pass
con=sqlite3.connect(root/'state/tracking/bagm.sqlite3',timeout=30)
con.row_factory=sqlite3.Row
registered=[]
for r in records:
 p=Path(r['path']); st=p.lstat()
 assert not p.is_symlink() and p.resolve()==p and p.is_file()
 assert all(p != q and q not in p.parents for q in protected)
 assert st.st_ino==r['inode'] and st.st_mtime_ns==r['mtime_ns']
 assert st.st_size==r['size_bytes'] and sha(p)==r['sha256']
 actual=con.execute('SELECT * FROM artifacts WHERE path=?',(str(p),)).fetchall()
 assert {v['artifact_id'] for v in actual}=={v['artifact_id'] for v in r['registered_artifacts']}
 for item in r['registered_artifacts']:
  row=con.execute('SELECT * FROM artifacts WHERE artifact_id=?',(item['artifact_id'],)).fetchone()
  for key in ['run_id','path','sha256','size_bytes','status']: assert row[key]==item[key]
  run=con.execute('SELECT status FROM runs WHERE run_id=?',(item['run_id'],)).fetchone()
  assert run['status'] in ('completed','failed','cancelled')
  assert item['run_id']!=a['active_run_id']
  registered.append(item)
backup=base/'registry_before.sqlite3'
with sqlite3.connect(backup) as dest:
 con.backup(dest); assert dest.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
before=shutil.disk_usage(root).free
stamp=datetime.now(timezone.utc).isoformat()
with con:
 for item in registered:
  n=con.execute('UPDATE artifacts SET status=? WHERE artifact_id=? AND status=?',('retention_deletion_pending',item['artifact_id'],item['status'])).rowcount
  assert n==1
  con.execute('UPDATE checkpoint_catalog SET verification_status=?,updated_at=? WHERE artifact_id=?',('retention_deletion_pending',stamp,item['artifact_id']))
with (base/'deleted_files.jsonl').open('x') as log:
 for r in records:
  p=Path(r['path']); st=p.lstat()
  assert st.st_ino==r['inode'] and st.st_mtime_ns==r['mtime_ns'] and st.st_size==r['size_bytes']
  p.unlink()
  log.write(json.dumps({'path':r['path'],'sha256':r['sha256'],'size_bytes':r['size_bytes']})+'\n');log.flush();os.fsync(log.fileno())
assert all(not Path(r['path']).exists() for r in records)
with con:
 for item in registered:
  n=con.execute('UPDATE artifacts SET status=? WHERE artifact_id=? AND status=?',('deleted_by_retention',item['artifact_id'],'retention_deletion_pending')).rowcount
  assert n==1
  con.execute('UPDATE checkpoint_catalog SET verification_status=?,updated_at=? WHERE artifact_id=?',('deleted_by_retention',stamp,item['artifact_id']))
for item in registered:
 assert con.execute('SELECT status FROM artifacts WHERE artifact_id=?',(item['artifact_id'],)).fetchone()[0]=='deleted_by_retention'
assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
status=subprocess.check_output(['supervisorctl','status','bagm_so12_relative_qkv_worker'],text=True).strip()
assert 'RUNNING' in status
active=root/'scratch/active_runs'/a['active_run_id']
checkpoints=list(active.glob('checkpoints/*'))
assert checkpoints and all(p.exists() for p in checkpoints)
receipt={'audit_sha256':sha(audit),'deleted_file_count':len(records),'deleted_bytes':sum(r['size_bytes'] for r in records),
 'registered_artifacts_tombstoned':len(registered),'before_free_bytes':before,'after_free_bytes':shutil.disk_usage(root).free,
 'sqlite_backup':str(backup),'sqlite_integrity':'ok','training_service':status,'active_checkpoint_files_preserved':len(checkpoints),
 'all_scoped_deletions_verified':True,'completed_at':datetime.now(timezone.utc).isoformat()}
(base/'application_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
