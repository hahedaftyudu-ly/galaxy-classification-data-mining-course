#!/usr/bin/env python3
"""Independent result-level checks for the course's quantitative requirements."""
import hashlib,importlib.metadata,json,re,time
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
import config
B=config.BASE

def read(p):return json.loads((B/p).read_text())
def main():
 t=time.time();checks=[]
 def check(name,condition):
  if not condition:raise AssertionError(name)
  checks.append(name)
 pf=pq.ParquetFile(config.FUSED);check('at least one million unique source rows and 59 fields',pf.metadata.num_rows>=1000000 and len(pf.schema_arrow)==59)
 c=read('results/classification/model_comparison.json');o=pq.read_table(B/'results/classification/oof_predictions.parquet').to_pandas()
 f=pq.read_table(config.FOLDS).to_pandas();m=f.merge(o,on='source_id',validate='1:1',suffixes=('_expected',''))
 check('all OOF sources match the split, with unchanged fold',len(m)==len(f)==pf.metadata.num_rows and m.source_id.is_unique and np.array_equal(m.fold_expected,m.fold))
 check('baseline plus three models',set(c['cv'])=={'LR','NB','RF','HGB'})
 for model in c['cv']:
  p=o[[f'prob_{model}_{x}' for x in config.CLASSES]].to_numpy();y=o.y.to_numpy()
  check(model+' probability finite, normalized and argmax-consistent',np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5) and np.array_equal(p.argmax(1),o['pred_'+model]))
  check(model+' exactly five outer folds',len(c['cv'][model]['folds'])==5)
  for block in c['cv'][model]['folds']:
   k=block['fold'];v=o.fold==k
   check(f'{model} fold {k} F1 recomputation',abs(f1_score(y[v],p[v].argmax(1),average='macro')-block['macro_f1'])<1e-7)
  for k,block in c['outer_tuning'][model].items():
   tr=np.flatnonzero(o.fold.to_numpy()!=int(k));chosen,_=train_test_split(tr,train_size=block['tune_rows'],stratify=y[tr],random_state=config.SEED+int(k))
   digest=hashlib.sha256('\n'.join(o.source_id.to_numpy()[chosen]).encode()).hexdigest()
   check(f'{model} fold {k} inner data identity and no outer validation overlap',digest==block['selected_id_sha256'] and np.all(o.fold.to_numpy()[chosen]!=int(k)) and block['inner_outer_overlap']==0)
 cl=read('results/clustering/clustering.json');st=cl['stability']
 check('bootstrap with replacement, ten rounds and ARI≥0.6',st['replace'] is True and st['bootstrap_rounds']>=10 and st['ari_mean']>=.6 and len(st['ari_values'])>=10)
 a=read('results/association/association_rules.json');check('sensitivity covers three threshold types and >=3 groups',len(a['sensitivity'])>=3 and len({x['support'] for x in a['sensitivity']})>=3 and len({x['confidence'] for x in a['sensitivity']})>=3 and len({x['lift_min_exclusive'] for x in a['sensitivity']})>=3)
 check('at least five actionable, nontrivial, non-label-conditioned rules',len(a['top_rules'])>=5 and all(len(x['consequent_items'])==1 and x['consequent_items'][0].startswith('class_') and all(not y.startswith('class_') for y in x['antecedent_items']) and x['lift']>1.1 for x in a['top_rules']))
 s=read('results/evaluation/supplement.json');check('three distinct held-out local explanations',len(s['local_samples'])>=3 and len({x['source_id'] for x in s['local_samples']})>=3)
 for x in s['local_samples']:
  row=o[o.source_id==x['source_id']].iloc[0]
  check('held-out explanation '+x['source_id'],row.fold==0 and x['additivity_error']<1e-5 and abs(sum(z['shap'] for z in x['contributions'])+x['expected_probability']-x['probability'])<1e-5 and abs(row['prob_RF_'+x['pred']]-x['probability'])<1e-5)
 check('six error cases and five slicing dimensions',len(s['error_samples'])>=5 and len({x['source_id'] for x in s['error_samples']})>=5 and len({x['dimension'] for x in s['error_slices']})>=5)
 for dim in {x['dimension'] for x in s['error_slices']}:
  check('slice partition total '+dim,sum(x['n'] for x in s['error_slices'] if x['dimension']==dim)==len(o))
 text=(B/'docs/期末报告_星系分类网络.md').read_text();pics=re.findall(r'!\[.*?\]\((.*?)\)',text)
 check('>=8 embedded report figures',len(pics)>=8 and all((B/'docs'/p).is_file() for p in pics))
 check('all six required main chapters',all('## '+x in text for x in ('一、选题与组内任务分配','二、需求分析','三、设计思路与实现方法','四、结果展示和关键代码分析','五、团队沟通记录','六、总结个人贡献与存在问题')))
 for line in (B/'requirements.txt').read_text().splitlines():
  if '==' in line:
   p,v=line.split('==');check('dependency pinned '+p,importlib.metadata.version(p)==v)
 nb=read('notebooks/星系分类网络_一键重跑.ipynb');check('Notebook contains executable local pipeline',any('run_local_pipeline.py' in ''.join(cell['source']) for cell in nb['cells']))
 result={'passed':True,'checks':checks,'check_count':len(checks),'rows':len(o),'figure_count':len(pics),'team_facts':'pending: group number, three member names, contributions summing to 100%, real collaboration records','limitations':['no fresh independent test set','no spatial block split','full raw-mode pipeline not rerun end to end in this review; corrected stages and verify mode executed'],'elapsed_s':round(time.time()-t,2)}
 (B/'results/review_20261116/submission_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
if __name__=='__main__':main()
