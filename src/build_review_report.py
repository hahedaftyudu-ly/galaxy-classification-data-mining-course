#!/usr/bin/env python3
"""Build report Markdown from auditable result files, without fabricated team facts."""
import json,platform,sys
from pathlib import Path
import config
B=config.BASE

def read(path):return json.loads((B/path).read_text())
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows])
def ms(a,k):return f"{a[k]['mean']:.4f}±{a[k]['std']:.4f}"
def main():
 c=read('results/classification/model_comparison.json');g=read('results/governance/governance_summary.json')
 cl=read('results/clustering/clustering.json');a=read('results/association/association_rules.json')
 e=read('results/evaluation/evaluation_summary.json');s=read('results/evaluation/supplement.json')
 co=read('results/governance/consistency.json');best=c['tests']['best_model']; agg=c['cv'][best]['aggregate'];t=c['tests']
 lines=[]
 def add(text=''):lines.append(text.strip()+'\n')
 def fig(n,title,file):add(f'![图{n} {title}](../results/figures/{file})')
 add('# 数据挖掘技术与应用结课报告')
 add('## 融合多波段观测数据的星系分类网络')
 add('组号：待填　　小组成员：待填　　日期：2026年11月16日')
 add('## 摘要')
 add(f'本项目将 SDSS×CatWISE 结构化交叉匹配星表与 Pan-STARRS 五波段图像文件元数据关联，得到 1,314,678 条有效记录和 59 个字段，完成数据治理、分类比较、聚类、关联规则和解释分析。分类器输入为 14 个 CatWISE 测光量及 W1−W2 颜色；图像元数据用于来源追溯和文件质量筛选，未进入预测器。采用固定外层分层 5 折，每个训练折内独立开展 3 折超参数搜索。{best} 的 Macro-F1 为 {ms(agg,"macro_f1")}。聚类取 k={cl["k_chosen"]}，有放回 bootstrap 的 OOB ARI 为 {cl["stability"]["ari_mean"]:.4f}±{cl["stability"]["ari_std"]:.4f}。关联规则经约束和去冗余后保留 {a["rules_after_prune"]} 条。报告包含 13 幅图、6 个错误案例、3 个留出样本的局部 SHAP 解释，以及可运行的 Notebook。')
 add('## 一、选题与组内任务分配')
 add('### 1.1 选题背景与任务边界')
 add('巡天星表中，星系、类星体和恒星在颜色、亮度、信噪比等测光特征上存在重叠。本项目建立三分类表格基线，识别容易混淆的样本群，并形成候选筛选规则与复核建议。选题属于天文数据分析，课程模板允许自主选题，具有明确的研究用途。')
 add('项目名称中的“网络”指数据获取、治理和分析组成的流程。本轮使用本机表格模型，不是端到端图像网络，未使用图像像素或形态特征；不声称验证了多模态融合增益。原始图像留在服务器，已有文件快照在本机复用。')
 add('### 1.2 组内任务分配')
 add(table(['组号','成员','实际负责内容','任务贡献%'],[['待填','成员一待填','据实际完成情况填写','待填'],['待填','成员二待填','据实际完成情况填写','待填'],['待填','成员三待填','据实际完成情况填写','待填'],['总计','','','应为100%']]))
 add('## 二、需求分析')
 add('### 2.1 目标与验收标准')
 add('目标一是建立可追溯的异构融合表；目标二是以统一的源级划分比较分类方法，重点观察少数类 STAR；目标三是给出可解释的分群、规则及错例，支持后续观测复核。技术验收依据课程模板表1，团队事实按实际记录补入。')
 add(table(['模板要求','本项目对应证据'],[
 ['至少2类异构来源及百万记录','结构化星表＋独立FITS文件元数据；1,314,678条、59字段'],
 ['缺失≥3策略、异常≥2方法、一致性','删除/中位数/迭代填补；3σ/IQR；处理前后统计'],
 ['基线加≥3模型、分层5折、搜索与检验','LR基线＋NB/RF/HGB；训练折内3折搜索；McNemar等'],
 ['聚类选k、bootstrap ARI≥0.6','肘部/轮廓系数；10次有放回重采样及OOB评估'],
 ['关联规则、效率、阈值、≥5条解读','Apriori与FP-Growth；9组含提升度变化的组合；14条规则'],
 ['分群误差、≥5错例、≥3局部解释','类别/亮度/SNR/颜色/赤纬切片；6错例；3个局部SHAP'],
 ['≥8图、Notebook、源码及日志','13幅图；3种本地运行模式；固定依赖和执行日志']]))
 add('### 2.2 数据与计算可行性')
 add('原始星表实测 3,944,564 行、24 字段，文件头中的 4,253,435 不是实际行数。图像元数据来自 395 个分区的真实 FITS 文件清单、文件属性及部分主头，属于独立的半结构化来源，并非将星表转为 JSON 充数。融合表约 338 MB，可在本机 10 核 CPU、16 GB 内存上分析。分类使用 4 个工作线程，聚类为 20 万抽样，关联规则为 15 万抽样；不把抽样分析写成全量训练。')
 add('图像完整性与像素质量须区分：已有 150,000 源的 750,000 图文件清单完整；300 源/1,500 图像素抽检中有 21 图含非有限像素。百万级像素内容验收未执行，不属于本轮表格分类结果。')
 add('## 三、设计思路与实现方法')
 add('### 3.1 获取与融合流程')
 add('原始星表和元数据快照 → 固定哈希选源 → source_id 一对一关联 → 完整性验收与数据字典 → 固定分层五折 → 治理对照与分类 → 聚类、规则 → 误差与 SHAP → 图表、报告和 Notebook。')
 add(table(['阶段','记录数或字段数'],[['原星表实际行数','3,944,564'],['固定 BLAKE2b(source_id) 模3为0的候选','1,315,321'],['文件缺失或重复而拒绝','643'],['最终有效独立source_id','1,314,678'],['主表字段','59'],['完整r头／仅文件属性','9,968／1,304,710']]))
 add('通过固定哈希遍历所有分区，不按模型预测筛选数据；五个波段不重复计为五条样本。元数据与星表在 source_id 上 1:1 内连接，并检查切图中心与 SDSS 位置偏移≤3角秒。字典记录全部59字段的名称、含义、类型、单位、范围、缺失率、来源与时点；原始快照、采集脚本、分区清单及拒绝记录均保留。采集区间为2026年11月11日02:26:55—02:45:59 UTC，历史星表导出时点另行保留，不等同于观测时刻。')
 add('### 3.2 固定划分与防泄漏')
 add('固定种子20261112，按class_01分层、按source_id划分。五折大小依次为262,937、262,936、262,936、262,935、262,934；类别比例最大偏差2.66×10⁻⁶。同源多波段始终在同一条记录内。填补、截尾、标准化和类别权重只在训练折估计。')
 add('本次修订将超参数搜索移到每个外层训练折内部：从约105万外层训练源中分层抽取30,000源做内层3折网格搜索，再以所选配置在完整外层训练折重训并预测约26万留出源。每个外层折的内层样本与其验证集交集为0，5折均产生OOF概率。检查点绑定主表哈希、折文件哈希、代码哈希、依赖版本和调参样本规模；输入变化不会误用旧预测缓存。')
 add('source_id、类别、路径、采集时间、文件mtime及头校验值均不作为预测特征。坐标只用于诊断。划分未按天区分块，近邻源和观测选择可能相关；本次属于已查看过数据后的修订实验，没有新的独立测试集，不能据此宣称完全无偏的外部泛化能力。')
 add('### 3.3 数据治理')
 add('原星表24字段未检出显式缺失，不能据此判定真实MCAR/MAR/MNAR机制。缺失实验是受控模拟：MCAR随机遮蔽W1或W2整组7特征；MAR按可见W1信噪比训练分位数遮蔽W2；MNAR按将被隐藏的W2星等训练分位数遮蔽W2。各机制设置5%/10%/20%缺失率，三策略共享掩码，比较删除、中位数和IterativeImputer(LinearRegression,max_iter=3)，统一下游LR。')
 add('原始尺度不同，重构误差逐字段保留原单位，跨字段汇总使用训练标准差归一化。删除法只在完整子集评估，因此另报覆盖率，并保存共同完整验证子集上的对照分数。缺失模拟中的重构指标不构成独立回归任务；本项目没有时序预测任务。')
 add('异常筛选比较3σ与IQR，阈值来自训练折。真实亮源、暗源和巡天选择尾部可能呈重尾，不因异常标记直接删除源；以训练折1%/99%分位数截尾修正极端输入。类型、编码、单位、重复键、范围、时区和关联基数在一致性环节独立检查。')
 add('### 3.4 特征与分类方法')
 add('输入为W1/W2各7个测光量（背景、背景误差、SNR、通量、通量误差、星等、星等误差）加W1−W2颜色，共15个特征。只新增一个领域颜色特征；模板中的不平衡专项与“8个衍生特征＋时间窗”均为可选项，本项目不申报这两项。单历元资料不应伪造时间窗特征。')
 add('LR提供线性基线；NB检验条件独立近似；RF建模非线性并可解释；HGB以提升树作另一集成对照。搜索空间：LR的C∈{0.1,1,10}和类别权重；NB的var_smoothing∈{1e−9,1e−7,1e−5}和先验；RF的max_depth∈{None,30}、叶最小样本∈{1,5}和类别权重；HGB的learning_rate∈{0.05,0.1}、叶节点数∈{31,63}和类别权重。RF最终100棵树，HGB最终最多120轮并使用训练折内早停验证。LR/NB另外标准化。')
 add('评价使用Accuracy、Macro-F1、Macro PR-AUC（以average precision定义）与Log loss，汇报五折均值及样本标准差。OOF配对预测作McNemar，折级Macro-F1作Friedman与Nemenyi。McNemar判断正确/错误率差异，不能把其显著性解释为Macro-F1或某个模型在所有指标上都更好；折间训练样本重叠，统计检验为探索性证据。')
 add('### 3.5 聚类与关联规则')
 add('K-means在20万分层抽样源、15个截尾并标准化特征上拟合，不把类别输入聚类器。k=2至10同时计算肘部与轮廓系数，取舍结合簇画像。10次有放回抽取20万行，各次重新拟合K-means，在未抽中的唯一OOB源上预测并与参考聚类比较ARI；预处理固定，因此报告的是给定表示下的聚类稳定性。')
 add('关联规则使用15万分层抽样，每源4个项：颜色、星等、SNR与类别。颜色分箱为≤0.05、(0.05,0.5]、>0.5 mag；W1为≤15、(15,16]、>16 mag；SNR为≤25、(25,50]、>50。Apriori与FP-Growth在同一矩阵上对比时间，并逐项校验项集及支持度一致。主阈值为support≥0.05、confidence≥0.7、lift>1.1。只保留后项为单个类别、前项不含类别的可行动规则，去除空项、重复和同后项下不提高置信度的更具体前项。')
 add('### 3.6 解释与工程实现')
 add('按真实类别、W1亮度、SNR、颜色、赤纬切片统计误差；从各类错误中按置信度、source_id确定性选2例，共6例，原因作为待验证假设。SHAP使用保存的第0折RF原模型及原截尾边界，选择第0折中每个已有类别按source_id排序的首个样本作局部解释，再加100个固定随机留出样本汇总全局重要性。样本不按预测成败挑选。校验SHAP基值＋贡献和等于类别概率，也校验模型概率与原OOF概率相同。')
 add('代码按data、src、notebooks、results、docs组织。requirements.txt固定实际分析环境版本；Python 3.13.7。断言覆盖百万行/字段、唯一键、5折完整、内外层无交集、概率归一化、项集等价、ARI门槛和SHAP加和。执行日志保留阶段耗时、硬件和命令；Notebook可验证现有结果、重算分析或从本地原始快照重建，全程不调用远程采集。')
 add('## 四、结果展示和关键代码分析')
 add('### 4.1 数据分布与治理结果')
 fig(1,'三类样本规模','fig01_class_distribution.png');fig(2,'三类W1−W2颜色分布','fig02_color_by_class.png')
 add('GALAXY 845,159源（64.29%），QSO 240,616（18.30%），STAR 228,903（17.41%）。少数类表现需结合Macro-F1、PR-AUC和各类召回评估。')
 mr=g['missing']['rows'];picked=[r for r in mr if r['mechanism']=='MCAR' or float(r['rate'])==.2]
 # Compact one row per mechanism/rate, preserving coverage.
 rows=[]
 for mech,rate in dict.fromkeys((r['mechanism'],r['rate']) for r in picked):
  block={r['strategy']:r for r in picked if r['mechanism']==mech and r['rate']==rate}
  rows.append([f'{mech} {rate:.0%}',f"{block['deletion']['macro_f1']:.4f}",f"{block['deletion']['coverage']:.0%}" if 'coverage' in block['deletion'] else ('80%' if rate==.2 else f'{1-rate:.0%}'),f"{block['median']['macro_f1']:.4f}",f"{block['iterative']['macro_f1']:.4f}"])
 add(table(['机制与比例','删除F1','删除覆盖率','中位数F1','迭代F1'],rows))
 fig(3,'整波段缺失策略比较','fig03_missing_strategies.png')
 add('删除法在MCAR下约0.727，但覆盖率随缺失率降至80%；MAR/MNAR下完整子集存在选择偏差，不能直接把删除法分数与全量填补分数当作同一任务比较。迭代填补在9组场景中的8组优于中位数，但MNAR 20%时中位数0.5565略高于迭代0.5523，因此不能写成“始终最差”或“所有机制最优”。若以MCAR且要求近全覆盖为参考，采用迭代填补；真实缺失机制改变时应重新验证。本轮主分类测光量无缺失，无需实际填补。')
 fig(4,'两种异常规则的标记比例','fig04_outlier_rules.png')
 add('IQR把约9.6%—12.8%的通量记录标为异常，3σ较少。LR探针对照从Macro-F1 0.7271提高至0.7490，五折差值均值+0.0220、标准差0.0005；据此保留全部源并做训练折截尾。该治理选择来自同份数据的探索，结果不等同于新外部测试上的增益。')
 add(table(['一致性项目','处理前','处理后'],[['有效行数','1,314,678','1,314,678'],['重复source_id','0','0'],['星表数值缺失单元格','0','0'],['非法标签/坐标/关联偏移','0','0'],['折号关联缺失','0','0'],['时间表示','UTC ISO8601','保持UTC，无需换算']]))
 add('r_mjd_obs只覆盖9,968源（0.758%）。其余头字段是未采集，不是观测自然缺失；FITS的BUNIT未提供处保留未知，单位字典一致不代表未知单位已被实测确认。')
 add('### 4.2 分类比较与统计检验')
 add(table(['模型','Accuracy','Macro-F1','Macro PR-AUC','STAR召回'],[[m,ms(v['aggregate'],'accuracy'),ms(v['aggregate'],'macro_f1'),ms(v['aggregate'],'macro_pr_auc'),f"{v['aggregate']['recall_mean']['STAR']:.3f}"] for m,v in c['cv'].items()]))
 fig(5,'分层五折模型比较','fig05_model_comparison.png')
 add(f'按OOF Macro-F1选择{best}，其错误率为{e["error_rate"]:.2%}。这不是独立测试结果；各模型调参使用的30,000源开发子集随外层折变化，最终模型在完整训练折拟合，所有配置见results/classification/model_comparison.json中的outer_tuning。')
 add(table(['配对比较','前者错/后者对','前者对/后者错','McNemar统计量','p值'],[[f"{x['a']} / {x['b']}",x['a_wrong_b_right'],x['a_right_b_wrong'],f"{x['statistic']:.3f}",'<1e−300' if x['p_value']<1e-300 else f"{x['p_value']:.4g}"] for x in t['mcnemar']]))
 add(f'Friedman统计量{t["friedman"]["statistic"]:.3f}，p={t["friedman"]["p_value"]:.5g}；Nemenyi临界差{t["nemenyi"]["critical_difference"]:.3f}。显著秩差对：'+('、'.join(x['a']+'–'+x['b'] for x in t['nemenyi']['pairs'] if x['significant']) or '无')+'。大样本下很小的p值不等于效果很大，应结合指标差、类别召回和相关性限制判断。')
 fig(6,'OOF混淆矩阵','fig06_confusion_matrix.png');fig(7,'OOF ROC曲线','fig07_roc.png');fig(8,'OOF PR曲线','fig08_pr.png')
 add('### 4.3 分群误差与典型错例')
 slice_rows=[x for x in s['error_slices'] if x['dimension'] in ('W1亮度','W1信噪比')]
 add(table(['分群维度','分组','源数','错误率','Macro-F1'],[[x['dimension'],x['group'],f"{x['n']:,}",f"{x['error_rate']:.2%}",f"{x['macro_f1']:.4f}"] for x in slice_rows]))
 add('类别、颜色及赤纬的全部切片保存在results/evaluation/error_slices.csv。切片支持定位薄弱观测条件，分组样本组成不同，不能仅凭错误率推断信噪比或天区的因果影响。以下六个源保留标识，便于回到星表核查：')
 for i,r in enumerate(s['error_samples'],1):
  add(f"案例{i}：source_id={r['source_id']}，已有标签{r['true']}→预测{r['pred']}，最大类概率{r['confidence']:.4f}；W1−W2={r['w1_w2_color']:.3f} mag，W1={r['w1mpro']:.2f} mag，SNR={r['w1snr']:.1f}。{r['hypothesis']}")
 add('高模型概率不是物理可靠度；这些案例尚未做独立光谱/图像标签复审，不能直接归因于标签错误或已确认的物理机制。')
 add('### 4.4 全局和局部解释')
 fig(9,'留出样本全局SHAP重要性','fig09_shap_importance.png')
 imp=read('results/evaluation/shap_importance.json')
 add('全局平均绝对贡献前三项为'+ '、'.join(list(imp)[:3])+f'；统计基于{s["global_explanation_rows"]}个第0折留出样本，用于描述该RF模型的局部总体趋势，不是131万源的全量SHAP普查。相关特征之间会分摊贡献，重要性不等于因果效应。')
 fig(10,'三个留出源的局部SHAP解释','fig13_local_shap.png')
 for i,r in enumerate(s['local_samples'],1):
  top=r['contributions'][:3]; desc='；'.join(f"{x['feature']}={x['raw_value']:.3g}，贡献{x['shap']:+.4f}" for x in top)
  add(f"局部样本{i}：source_id={r['source_id']}，已有标签{r['true']}，预测{r['pred']}。预测类基值{r['expected_probability']:.4f}＋全部特征贡献{r['shap_sum']:+.4f}＝预测概率{r['probability']:.4f}。绝对贡献前三：{desc}。正贡献提高该预测类概率，负贡献降低；其余项见supplement.json。")
 add('三个源都在留出折，第0折模型未用其标签训练；解释概率与原OOF概率一致，加和误差小于10⁻⁵。SHAP只解释模型为何给出该预测，不证明预测在物理上正确。')
 add('### 4.5 聚类结果')
 fig(11,'聚类PCA投影','fig10_cluster_pca.png');fig(12,'肘部与轮廓系数','fig11_k_selection.png')
 add(f'肘部法给k={cl["k_elbow"]}，轮廓系数最高在k={cl["k_silhouette"]}（0.3080）。选k={cl["k_chosen"]}（轮廓{cl["silhouette_chosen"]:.4f}）是保留亮度/颜色/SNR细分画像的解释性折中，非两指标共同最优。10次有放回重采样OOB ARI={cl["stability"]["ari_mean"]:.4f}±{cl["stability"]["ari_std"]:.4f}，超过0.6；这验证给定预处理下稳定性，不替代科学有效性。')
 names=['低信噪比星系富集群','主体混合群','明亮高信噪比群','暗弱红端混合群']
 add(table(['簇','规模','星系/QSO/恒星%','颜色中位','W1均值','命名'],[[x['cluster'],f"{x['size']:,}",'/'.join(f"{100*x['class_share'].get(c,0):.1f}" for c in config.CLASSES),f"{x['color_median']:.2f}",f"{x['means']['w1mpro']:.2f}",names[x['cluster']]] for x in cl['profiles']]))
 add('颜色与星等单位均为mag。簇0优先用于星系候选复核；簇1作为主体混合样本使用分类器；簇2适合优先人工复核和观测跟进；簇3优先补充光谱、形态或变光资料。原始簇命名中的“主序”缺乏恒星主序证据，修订为“主体混合群”。')
 add('### 4.6 关联规则与用途')
 add(f"Apriori耗时{a['efficiency']['apriori_s']:.2f}s，FP-Growth {a['efficiency']['fpgrowth_s']:.2f}s，均得到{a['efficiency']['apriori_itemsets']}个频繁项集，项集和支持度逐项一致。当前只有12项、每事务4项，树构建开销可能主导，不能外推算法优劣。")
 add(table(['支持度下限','置信度下限','提升度严格下限','规则数'],[[x['support'],x['confidence'],x['lift_min_exclusive'],x['rules']] for x in a['sensitivity']]))
 add('上述9组组合同时覆盖支持度、置信度和提升度变化，用于整体阈值敏感性比较；其中置信度与提升度成对变化，不能据此单独归因于某个阈值。')
 add(f"主阈值下原始{a['rules_before_prune']}条，限制可行动方向后{a['actionable_candidates']}条，去冗余后{a['rules_after_prune']}条。以下5条的前项均不含类别：")
 for i,r in enumerate(a['top_rules'][:5],1):
  tail=[
   '中等SNR下再要求暗弱且偏红，可组成较严格的QSO优先复核池；与规则2比较条件增加后的置信度和支持度折中。',
   '中等SNR的红源不限制亮度，适合较宽的QSO预选入口；其支持度大于规则1，条件更少，需接受更低的置信度。',
   '暗弱加红端只需两个测光条件，便于批量初筛；适合将候选交给后续光谱或形态复核，不保证外部巡天精确率。',
   '只用颜色即可建立粗筛队列，比多条件规则易实施；复杂规则并非一定必要，应比较筛选成本、覆盖和误报。',
   '中等亮度低SNR的样本在本抽样中星系富集，可将该队列优先安排更深成像或星系候选复核；低SNR不等同于数据无效。'
  ][i-1]
  add(f"规则{i}：{r['antecedent']} → {r['consequent']}；支持度{r['support']:.3f}、置信度{r['confidence']:.3f}、提升度{r['lift']:.2f}。{tail}")
 add('规则只在15万抽样事务上挖掘，未在独立样本验证，属于描述性候选筛选依据，不能替代分类模型或作为因果规律。')
 fig(13,'关联规则支持度与置信度','fig12_association_rules.png')
 add('### 4.7 关键代码与分析')
 add('源级分层划分使用NumPy按类别取索引和随机置换，再均匀分配折号，保证一源一折：')
 add('```python\nrng = np.random.default_rng(config.SEED)\nfor cls in sorted(np.unique(labels)):\n    idx = np.flatnonzero(labels == cls)\n    order = idx[rng.permutation(len(idx))]\n    fold[order] = np.arange(len(order)) % 5\n```')
 add('异构融合使用pandas的一对一校验，额外断言防止多对多关联将行数膨胀：')
 add('```python\njoined = catalog.loc[frame.index].join(\n    frame, how="inner", validate="one_to_one")\nassert len(joined) == valid\n```')
 add('每个外层训练折单独抽取内层样本，搜索函数内部重新拟合截尾和标准化。内外层无交集断言保护最终验证源：')
 add('```python\ntr_idx = np.flatnonzero(fold != k)\nva_idx = np.flatnonzero(fold == k)\nchosen, _ = train_test_split(\n    tr_idx, train_size=30000, stratify=y[tr_idx],\n    random_state=config.SEED + int(k))\nassert not np.intersect1d(chosen, va_idx).size\nbest, all_results = tune_model(name, X[chosen], y[chosen])\n```')
 add('有放回重采样会包含重复源，稳定性评价仅在未抽中的唯一OOB源上计一次，避免重复源使ARI虚高：')
 add('```python\nidx = rng.choice(n, size=n, replace=True)\nboot_model = KMeans(n_clusters=k, random_state=seed).fit(X[idx])\noob = np.ones(n, dtype=bool)\noob[np.unique(idx)] = False\nari = adjusted_rand_score(labels[oob], boot_model.predict(X[oob]))\n```')
 add('### 4.8 运行方式与日志')
 add('默认Notebook模式verify会检查融合表、划分和全部交付结果，重新绘图并生成报告Markdown；recompute从本地融合表重做治理、嵌套调参、全部训练和解释；raw先从本地原始星表和395个JSONL快照重建融合表，再执行recompute。完整重跑使用--fresh，忽略旧拟合检查点。')
 add('```bash\npython -m pip install -r requirements.txt\npython src/run_local_pipeline.py --mode verify\n# 完整重算，可耗时数十分钟\npython src/run_local_pipeline.py --mode recompute\n# 从本地原始快照重建并重算\npython src/run_local_pipeline.py --mode raw\n```')
 add('隔离目录已跳过候选Parquet缓存，从原星表和395份JSONL快照重建融合表，耗时175.276秒；重建后的1,314,678行、59字段文件与原表SHA-256完全一致，证明该本地原始重建环节可复现。完整raw模式没有把所有后续分析再重跑一遍。')
 add('Notebook为notebooks/星系分类网络_一键重跑.ipynb，运行Jupyter Lab后执行全部单元即可。各次流水线日志在results/runs/日期时间/，记录硬件、依赖、命令、退出码和阶段耗时。报告打印文件的生成使用src/render_course_report.py及documents技能的渲染工具；Notebook的主要复现产物为图表、结果表和Markdown。')
 add(f'本轮分类实际耗时{c["elapsed_s"]:.1f}s，聚类{cl["elapsed_s"]:.1f}s，规则{a["elapsed_s"]:.1f}s，补充解释{s["elapsed_s"]:.1f}s；原缺失实验耗时1,523.6s。旧分类汇总中的5.4s仅为缓存汇总时间，不能写作完整训练耗时。修订前版本在results/review_20261116/before中保留。')
 add('## 五、团队沟通记录')
 add('现有chat目录保存项目用户与助手的技术讨论，可用于追溯技术决策。它不等同于三位组员的真实沟通。本节应由小组依据真实日期、参与者和讨论结论补齐。')
 add(table(['实际日期','参与成员','讨论与任务衔接','决定和产出'],[['待填','待填','按真实记录填写','待填'],['待填','待填','按真实记录填写','待填']]))
 add('技术决策可追溯记录：11月11日形成数据采集与只读边界；11月12日完成百万级融合验收；11月16日完成分类、聚类、规则和模板复核。以上是项目工作记录，不作为已发生的组员会议证明。')
 add('## 六、总结个人贡献与存在问题')
 add('### 6.1 个人贡献')
 add(table(['成员','本人完成的具体工作','与他人交接内容','确认比例'],[['成员一待填','待填','待填','待填'],['成员二待填','待填','待填','待填'],['成员三待填','待填','待填','待填']]))
 add('### 6.2 存在问题与改进')
 add('真实缺失机制未知，模拟实验不能替代实际缺失评估；可收集有缺失的新数据并以相同覆盖率比较策略。源级随机划分未阻断空间相关，应增加天区分组验证与新独立测试。主模型只使用CatWISE测光，没有证据表明图像元数据提高了分类性能；后续应在授权范围内加入可用形态或像素表征，并做受控消融。')
 add('STAR错误较多，需结合成本、召回和概率校准制定复核阈值；不能因总体Accuracy较高就忽略少数类。SHAP全局统计样本较小，可扩大未见源覆盖并检查解释稳定性。关联规则阈值和分箱具有主观性，应另设独立验证集。bootstrap稳定性建立在固定预处理上，可再做重采样中重新估计预处理的敏感性实验。')
 add('模板可选的专项不平衡对照及8个衍生特征含时间窗项均未申报；不将其算作必需缺口。组号、三位组员姓名、贡献比例和真实团队沟通仍待实际填写。')
 add('## 附录 交付材料索引')
 add('数据字典与验收见docs/数据字典_正式版.md、docs/数据获取与融合_验收报告.md；划分见docs/防泄漏与划分说明.md；分类完整结果见results/classification/model_comparison.json、oof_predictions.parquet；局部解释与误差切片见results/evaluation/supplement.json、error_slices.csv；聚类和规则见results/clustering及results/association；13幅图见results/figures；模板对照结论见docs/课程模板符合性检查与补足报告.md。')
 add('数据来源依据与字段解释链接保存在数据字典的来源章节，原始查询头保存在docs/历史IRSA查询头.txt。数据文件和代码留在本机课程目录，未向公开服务上传。')
 (B/'docs/期末报告_星系分类网络.md').write_text('\n'.join(lines),encoding='utf-8')
 # Keep companion reports consistent with the final source of truth.
 classification=['# 分类模型比较与统计检验\n',f"外层分层5折，内层每训练折30,000源3折搜索；总耗时{c['elapsed_s']:.1f}s。\n",table(['模型','Accuracy','Macro-F1','Macro PR-AUC'],[[m,ms(v['aggregate'],'accuracy'),ms(v['aggregate'],'macro_f1'),ms(v['aggregate'],'macro_pr_auc')] for m,v in c['cv'].items()]),'\n## 各外层折所选参数\n']
 classification.append(table(['模型','外层折','最优参数','内层F1','与外层验证重叠数'],[[m,k,json.dumps(v['best']['params'],ensure_ascii=False),f"{v['best']['mean_macro_f1']:.4f}",v['inner_outer_overlap']] for m,folds in c['outer_tuning'].items() for k,v in folds.items()]))
 classification.append('\n## 统计检验\n\n'+json.dumps(t,ensure_ascii=False,indent=2))
 classification.append('\n检验是探索性比较。McNemar衡量错误率差异，不直接检验Macro-F1；折间训练集重叠，且未作新独立测试。图像元数据只用于验收与追溯，预测器仅使用15个测光与颜色特征。\n')
 (B/'docs/分类模型比较报告.md').write_text('\n'.join(classification))
 start='### 4.5 聚类结果';end='### 4.7 关键代码与分析';txt='\n'.join(lines);sub=txt[txt.index(start):txt.index(end)]
 (B/'docs/聚类与关联规则报告.md').write_text('# 聚类与关联规则报告\n\n'+sub)
 import csv
 rows=[]
 for m,v in c['cv'].items():
  row={'model':m}
  for key in ('accuracy','macro_f1','macro_pr_auc','log_loss'):
   row[key]=v['aggregate'][key]['mean'];row[key+'_std']=v['aggregate'][key]['std']
  rows.append(row)
 with (B/'results/classification/model_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Report Markdown and companion summaries rebuilt from verified results.')
if __name__=='__main__':main()
