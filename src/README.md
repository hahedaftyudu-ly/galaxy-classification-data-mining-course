# 课程项目源码运行说明

Python 3.13；依赖以项目根目录requirements.txt为准。原始快照在data/raw，融合表在data/processed。原课程作业未修改。

```bash
python -m pip install -r requirements.txt
python src/run_local_pipeline.py --mode verify
```

- verify：检查融合表与划分，重绘图表、重建报告Markdown并独立验收。
- recompute：从融合表重做治理、嵌套调参、全量外层训练、聚类、规则和SHAP。模型使用--fresh，不复用旧拟合检查点。可耗时数十分钟。
- raw：先从本地原始星表和395个JSONL快照重建融合，再执行recompute。

以上入口不调用SSH。各次运行在results/runs/日期时间下记录命令、退出码、硬件、依赖和阶段耗时。Notebook包装同一入口，不再以“只显示已保存结果”冒充完整重算。

## 主要脚本

- build_fused_dataset.py --fresh-catalog：从本地原始快照重建候选与融合表。
- finalize_acquisition.py、validate_data_delivery.py：字典、数据独立验收。
- make_splits.py、verify_splits.py：固定源级分层5折，seed=20261112。
- governance_missing.py、governance_outliers.py、governance_consistency.py：三策略缺失模拟、两种异常规则、处理前后一致性。
- classification_experiment.py --fresh --tune-size 30000：每个外层训练折内抽3万源做3折网格搜索，最终在完整外层训练折重训。检查点绑定数据/划分/代码和依赖身份。4模型为LR、NB、RF、HGB。
- clustering_analysis.py --k 4：k搜索、画像、10次有放回bootstrap，在唯一OOB源上比较ARI。
- association_rules.py：算法效率和项集等价性、三种阈值变化、可行动方向和冗余约束。
- supplement_evaluation.py：原第0折RF模型的留出局部SHAP、误差切片和6个错例；不重新拟合模型。
- evaluation_figures.py：从最终结果重绘基础图；不再偷偷训练解释用替代模型。
- build_review_report.py：从JSON生成主报告Markdown、分类及聚类/规则分项报告和CSV。
- render_course_report.py：使用保留的课程模板生成DOCX，须用Codex bundled Python运行；生成后以documents技能render_docx.py渲染、逐页验证，PDF使用同次导出。
- validate_course_submission.py：独立检查OOF指标、内外层数据身份、bootstrap、SHAP加和、规则、图表与依赖。

src/resources/course_template.docx是从原课程DOC转换的排版底稿。原DOC保留不动。修订前的文档、脚本、Notebook和结果在results/review_20261116/before保留，供对照，不是当前成绩依据。

## 模型与解释范围

分类器只输入14个CatWISE测光量与W1-W2颜色；图像元数据用于融合验收和来源追溯。没有端到端图像网络训练。rf_fold0.joblib保存第0折精确RF及截尾边界，用于复核对应留出解释；其余折预测及配置均保留，未宣称交付所有折的模型权重。

没有新独立测试集，也未按天区分组；统计检验与解释均需结合这些限制。可选8个派生特征/时间窗以及不平衡专项不申报。

## 历史远程只读审计

audit_catalog_readonly.py、audit_image_sample_readonly.py、audit_150k_readonly.py、export_metadata_sample_readonly.py及collect_million_metadata_readonly.py为历史只读采集工具。本轮模板复核未访问远程。今后操作远程前必须先读取远程AGENTS.md和任务说明.md，并遵守本项目严格只读约定。
