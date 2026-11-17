# 数据获取与融合：运行说明

## 查看本地分析表

主表：data/processed/fused_sources.parquet。一行一个独立天体源，字段定义见docs/数据字典_正式版.md。目录中的图像路径指向<REMOTE_HOST>原始资料，不代表本机下载了百万源像素。

```python
import pandas as pd
from pathlib import Path
root = Path("<PROJECT_ROOT>")
df = pd.read_parquet(root / "data/processed/fused_sources.parquet",
                     columns=["source_id","class_01","w1mpro","w2mpro","band_count","r_bytes"])
print(df.shape)
print(df["class_01"].value_counts())
```

## 复查或重跑

使用本机已记录的Python与依赖（requirements.txt和results/local_environment.json）。本轮没有安装软件。

```bash
python3 '<PROJECT_ROOT>/src/run_acquisition_pipeline.py'
```

已有原始快照会先校验并复用；已完成图像分区会跳过。这样可以在本机复查已归档数据，并重建融合表、字典和验证记录。缺少快照时会只读访问<REMOTE_HOST>，要求当前机器具备既有SSH配置和权限。

## 文件布局

- data/raw/SDSSxWISE_cat.tbl.gz：完整原始星表快照，SHA-256与服务器一致。
- data/raw/image_metadata/images*.jsonl.gz：真实图像文件元数据JSONL，含分区开始/结束记录及拒绝条目。
- data/raw/image_metadata/manifests/：分区数量、采集时点、文件SHA-256。
- data/processed/catalog_selected.parquet：1,315,321个固定哈希候选的原24字段。
- data/processed/fused_sources.parquet：通过关联和规则验收的最终主表。
- results/acquisition/：传输、导入、融合、拒绝、执行协议和校验结果。
- src/versions/：各采集版本，与协议中源码哈希对应。

## 解释边界

本次完成“星表＋真实图像元数据”采集融合。file_only只检查文件清单/属性和位置关联；header_and_file还读取完整r主头。逐像素检查覆盖范围单独记录。不能把元数据采集合格写成全部图像像素无缺陷，也不能把未采集头字段当作自然缺失去填造。

历史2022年查询链未完整重建；本次来源是明确标识的既有文件快照。当前采集时间、历史导出时间、图像MJD及文件修改时间均分开记录。原SDSS输入版本未独立核验。

治理、分类、聚类与关联规则均已完成；分类以修订后的外层五折、训练折内三折搜索为准。元数据未作为预测特征。

## 本地复现入口

`python src/run_local_pipeline.py --mode verify` 验证现有结果；`--mode recompute`强制重做全部分析；`--mode raw`先从本地原始星表和395个JSONL快照重建，再重算。三个模式均不访问服务器。在线采集属于独立步骤，不能将本地raw模式误写为重新下载全部图像。
