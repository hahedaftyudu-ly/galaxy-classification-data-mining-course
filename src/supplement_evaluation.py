#!/usr/bin/env python3
"""Held-out local SHAP explanations and error slices; never refits a model."""
import json,time,os
from pathlib import Path
import joblib,numpy as np,pandas as pd,pyarrow.parquet as pq
import config
os.environ.setdefault('MPLCONFIGDIR',str(config.BASE/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import shap

BASE=config.BASE; OUT=BASE/'results/evaluation'; FIG=BASE/'results/figures'
CLASSES=list(config.CLASSES)
plt.rcParams['font.sans-serif']=['PingFang SC','Heiti TC','Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False

def main():
 t=time.time(); OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
 result=json.loads((BASE/'results/classification/model_comparison.json').read_text())
 best=result['tests']['best_model']
 oof=pq.read_table(BASE/'results/classification/oof_predictions.parquet').to_pandas()
 df=pq.read_table(config.FUSED,columns=[config.GROUP_KEY,config.LABEL,'ra','dec']+config.CATWISE_NUMERIC).to_pandas()
 df=df.merge(oof,on='source_id',validate='1:1'); assert len(df)==len(oof)
 df['w1_w2_color']=df.w1mpro-df.w2mpro
 df['wrong']=df.y!=df['pred_'+best]
 groups=[]
 specs={'真实类别':df.class_01,'W1亮度':pd.cut(df.w1mpro,[-np.inf,15,16,np.inf],labels=['≤15 mag','15–16 mag','>16 mag']),
        'W1信噪比':pd.cut(df.w1snr,[-np.inf,25,50,np.inf],labels=['≤25','25–50','>50']),
        'W1−W2颜色':pd.cut(df.w1_w2_color,[-np.inf,.05,.5,np.inf],labels=['≤0.05 mag','0.05–0.5 mag','>0.5 mag']),
        '赤纬':pd.cut(df.dec,[-90,0,30,60,90],labels=['−90–0度','0–30度','30–60度','60–90度'])}
 for dim,cats in specs.items():
  for label,rows in df.groupby(cats,observed=True):
   groups.append({'dimension':dim,'group':str(label),'n':len(rows),'wrong':int(rows.wrong.sum()),
                  'error_rate':float(rows.wrong.mean()),'macro_f1':float(f1_score(rows.y,rows['pred_'+best],labels=[0,1,2],average='macro',zero_division=0))})
 pd.DataFrame(groups).to_csv(OUT/'error_slices.csv',index=False)
 samples=[]
 for code,c in enumerate(CLASSES):
  e=df[(df.y==code)&df.wrong].copy(); e['confidence']=e[[f'prob_{best}_{c}' for c in CLASSES]].max(axis=1)
  for _,r in e.sort_values(['confidence','source_id'],ascending=[False,True]).head(2).iterrows():
   if r['pred_'+best]==1: hypothesis='红端颜色与QSO候选区域重叠；需用光谱或形态复核，不能仅据测光认定标签错误。'
   elif r.w1_w2_color<=.5: hypothesis='蓝端或中间色的星系与恒星测光重叠；亮度、SNR和形态可能影响判别，尚未验证物理成因。'
   else: hypothesis='颜色与预测类别常见区域不一致，需核对多特征贡献、交叉匹配及观测质量；当前仅为待验证原因。'
   samples.append({'source_id':r.source_id,'true':c,'pred':CLASSES[int(r['pred_'+best])],
    'confidence':float(r.confidence),'w1_w2_color':float(r.w1_w2_color),'w1mpro':float(r.w1mpro),'w1snr':float(r.w1snr),'hypothesis':hypothesis})
 # Exact saved fold-0 RF, with train-fold clipping. Select one held-out source
 # per class by source_id, independently of whether the prediction is correct.
 state=joblib.load(BASE/'results/classification/rf_fold0.joblib')
 assert state['signature']==result['cache_signature']
 held=df[df.fold==state['heldout_fold']].sort_values('source_id')
 picks=pd.concat([held[held.y==i].head(1) for i in range(3)])
 assert len(picks)==3 and (picks.fold==0).all()
 global_picks=held.sample(n=100,random_state=config.SEED)
 explain=pd.concat([picks,global_picks]).drop_duplicates('source_id')
 X=np.clip(explain[state['features']].to_numpy(float),state['lo'],state['hi'])
 print('Explaining',len(X),'held-out sources',flush=True)
 explainer=shap.TreeExplainer(state['model'])
 vals=explainer.shap_values(X,check_additivity=True)
 arr=np.stack(vals,axis=2) if isinstance(vals,list) else np.asarray(vals)
 assert arr.shape==(len(X),len(state['features']),3)
 prob=state['model'].predict_proba(X); expected=np.asarray(explainer.expected_value)
 assert np.allclose(expected+arr.sum(axis=1),prob,atol=1e-5)
 cached=explain[[f'prob_RF_{c}' for c in CLASSES]].to_numpy()
 assert np.allclose(prob,cached,atol=1e-5), 'not the original OOF predictor'
 importance=dict(sorted(zip(state['features'],np.abs(arr).mean(axis=(0,2)).tolist()),key=lambda kv:kv[1],reverse=True))
 (OUT/'shap_importance.json').write_text(json.dumps(importance,ensure_ascii=False,indent=2)+'\n')
 local=[]
 for j,(_,r) in enumerate(picks.iterrows()):
  pos=int(np.flatnonzero(explain.source_id.to_numpy()==r.source_id)[0]); c=int(prob[pos].argmax()); values=arr[pos,:,c]
  order=np.argsort(np.abs(values))[::-1]
  local.append({'source_id':r.source_id,'true':CLASSES[int(r.y)],'pred':CLASSES[c],
   'fold':0,'expected_probability':float(expected[c]),'probability':float(prob[pos,c]),
   'shap_sum':float(values.sum()),'additivity_error':float(abs(expected[c]+values.sum()-prob[pos,c])),
   'contributions':[{'feature':state['features'][i],'raw_value':float(r[state['features'][i]]),
                     'model_value':float(X[pos,i]),'shap':float(values[i])} for i in order]})
 fig,axes=plt.subplots(1,3,figsize=(13,4.4))
 for ax,item in zip(axes,local):
  top=item['contributions'][:7][::-1]; v=[x['shap'] for x in top]
  ax.barh([x['feature'] for x in top],v,color=['#0072B2' if x>=0 else '#D55E00' for x in v]);ax.axvline(0,color='black',lw=.6)
  ax.set_title(f"已有标签 {item['true']} → 预测 {item['pred']}\n概率 {item['probability']:.3f}",fontsize=10)
  ax.set_xlabel('对预测类别的概率贡献（无量纲）')
 fig.suptitle('三个留出样本的局部 SHAP 解释（RF，第0折）')
 fig.text(.01,.01,'数据来源：本机融合表；蓝色提高、橙色降低该类概率；每幅仅显示绝对贡献前7项。',fontsize=8)
 fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(FIG/'fig13_local_shap.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(6.4,4.3)); keys=list(importance)[::-1]
 ax.barh(keys,[importance[k] for k in keys],color='#0072B2');ax.set_xlabel('平均 |SHAP|（三类概率贡献，无量纲）');ax.set_title('全局特征重要性（RF，第0折留出样本）')
 fig.text(.01,.01,f'数据来源：本机融合表；{len(explain)}个留出样本；原始折模型，无重新拟合。',fontsize=7)
 fig.tight_layout(rect=(0,.04,1,1));fig.savefig(FIG/'fig09_shap_importance.png',dpi=150,bbox_inches='tight');plt.close(fig)
 extra={'model':'RF','heldout_fold':0,'global_explanation_rows':len(explain),'selection':'one source per true class in fold 0 sorted by source_id, plus 100 seeded random held-out sources',
        'output_space':'class probability','expected_values':expected.tolist(),'local_samples':local,'error_samples':samples,'error_slices':groups,'best_error_model':best,'elapsed_s':round(time.time()-t,2)}
 (OUT/'supplement.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'local_samples':len(local),'error_samples':len(samples),'error_groups':len(groups),'elapsed_s':extra['elapsed_s']}),flush=True)
if __name__=='__main__':main()
