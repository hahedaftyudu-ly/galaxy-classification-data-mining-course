#!/usr/bin/env python3
"""Local-only reproducible pipeline; logs commands, elapsed times and hardware."""
import argparse,datetime,importlib.metadata,json,os,platform,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
import config

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['verify','recompute','raw'],default='verify');args=ap.parse_args()
 base=config.BASE; out=base/'results/runs'/datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
 out.mkdir(parents=True,exist_ok=False)
 packages={}
 for p in ('numpy','pandas','pyarrow','scikit-learn','scipy','joblib','mlxtend','shap','matplotlib','nbformat','nbconvert','ipykernel'):
  packages[p]=importlib.metadata.version(p)
 import psutil
 record={'mode':args.mode,'started_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
         'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'processor':platform.processor(),
         'logical_cpus':os.cpu_count(),'memory_bytes':psutil.virtual_memory().total,'packages':packages,
         'network':'local only; no SSH or acquisition scripts','steps':[]}
 (out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 steps=[]
 if args.mode=='raw': steps += [('build_fused_dataset.py',['--fresh-catalog']),('finalize_acquisition.py',[])]
 if args.mode!='verify': steps += [('make_splits.py',[])]
 steps += [('validate_data_delivery.py',[]),('verify_splits.py',[])]
 if args.mode!='verify':
  steps += [('governance_missing.py',[]),('governance_outliers.py',[]),('governance_consistency.py',[]),
            ('summarize_governance.py',[]),('classification_experiment.py',['--fresh','--tune-size','30000']),
            ('clustering_analysis.py',['--k','4']),('association_rules.py',[]),('supplement_evaluation.py',[])]
 steps += [('evaluation_figures.py',[]),('build_review_report.py',[]),('validate_course_submission.py',[])]
 env=os.environ.copy();env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'1'})
 started=time.monotonic()
 for index,(script,extra) in enumerate(steps):
  cmd=[sys.executable,str(base/'src'/script)]+extra; t=time.monotonic()
  print('RUN',' '.join(cmd),flush=True)
  log=out/f'{index:02d}_{script}.log'
  with log.open('w') as f: proc=subprocess.run(cmd,cwd=base,env=env,stdout=f,stderr=subprocess.STDOUT)
  record['steps'].append({'script':script,'arguments':extra,'returncode':proc.returncode,'elapsed_s':round(time.monotonic()-t,3),'log':str(log.relative_to(base))})
  record['elapsed_s']=round(time.monotonic()-started,3);record['passed']=proc.returncode==0
  (out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
  if proc.returncode:raise SystemExit(f'Failed: {script}; see {log}')
 record['finished_at']=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
 (out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'passed':True,'run':str(out),'elapsed_s':record['elapsed_s']},ensure_ascii=False))
if __name__=='__main__':main()
