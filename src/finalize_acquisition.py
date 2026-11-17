#!/usr/bin/env python3
"""Produce the complete field dictionary and evidence-based acquisition report."""
from pathlib import Path
import datetime, json, hashlib, platform
from zoneinfo import ZoneInfo
import pyarrow.parquet as pq
BASE=Path(__file__).resolve().parents[1]
DOC=BASE/"docs"
def main():
 summary=json.loads((BASE/"results/acquisition/fusion_summary.json").read_text())
 imported=json.loads((BASE/"results/acquisition/catalog_import.json").read_text())
 previous=json.loads((DOC/"星表数据字典_初稿.json").read_text())
 originals={f["name"]:f for f in previous["fields"]}
 meta={
"partition":("图像分区名称","无","images1—images395"),
"image_directory":("实际源图像目录","无","服务器绝对路径；目录名为source_id"),
"g_path":("实际g波段文件路径","无","服务器绝对FITS路径"),
"r_path":("实际r波段文件路径","无","服务器绝对FITS路径"),
"i_path":("实际i波段文件路径","无","服务器绝对FITS路径"),
"z_path":("实际z波段文件路径","无","服务器绝对FITS路径"),
"y_path":("实际y波段文件路径","无","服务器绝对FITS路径"),
"band_count":("被识别且唯一的grizy波段数","个","本数据固定为5"),
"r_bytes":("r波段实际文件大小","byte","正整数且能被2880整除"),
"r_mtime_ns":("r文件最后修改时间；非观测或本轮采集时间","ns since Unix epoch UTC","文件系统原值"),
"r_header_sha256":("已读取r主头块的SHA-256","无","64位十六进制；未采集头时为null"),
"r_header_bytes":("r主头块长度","byte","正的2880整数倍；未采集头时为null"),
"r_width":("主头声明的图像宽","pixel","已读头记录为240；未采集头时为null"),
"r_height":("主头声明的图像高","pixel","已读头记录为240；未采集头时为null"),
"r_bitpix":("FITS BITPIX编码","bit/pixel","8/16/32/64/-32/-64；未采集头时为null"),
"r_filter":("r波段标记；file_only取自文件名，header_and_file取自头","无","r"),
"r_metadata_scope":("元数据采集深度","无","file_only/header_and_file"),
"r_mjd_obs":("头内标称观测MJD；未核验TIMESYS时不强转UTC","day","有限MJD；未采集头时为null"),
"r_exptime":("头内标称输入曝光时间之和，不保证等于逐像素有效曝光","s","正有限数；未采集头时为null"),
"r_bunit":("原头BUNIT标签；缺省不代表Jy","单位标签","原始字符串或null；本轮未见提供"),
"r_ctype1":("原WCS第一轴投影类型","无","原始字符串或null"),
"r_ctype2":("原WCS第二轴投影类型","无","原始字符串或null"),
"r_crval1":("原WCS第一轴参考坐标，不直接等于切图中心","deg","原始有限值或null"),
"r_crval2":("原WCS第二轴参考坐标，不直接等于切图中心","deg","原始有限值或null"),
"r_crpix1":("原WCS第一轴参考像素，可在切图外","pixel","原始有限值或null"),
"r_crpix2":("原WCS第二轴参考像素，可在切图外","pixel","原始有限值或null"),
"r_cdelt1":("原WCS第一轴尺度参数；几何使用仍需完整WCS","deg/pixel","原始有限值或null"),
"r_cdelt2":("原WCS第二轴尺度参数；几何使用仍需完整WCS","deg/pixel","原始有限值或null"),
"cutout_ra":("从真实切图文件名提取的中心赤经","deg","0≤值<360"),
"cutout_dec":("从真实切图文件名提取的中心赤纬","deg","-90≤值≤90"),
"cutout_arcsec":("文件名记录的切图边长；旧文件名未记录时留空","arcsec","正有限值或null"),
"pixel_quality_checked":("本轮采集是否逐图读像素检查","无","false；其他抽检证据单列"),
"metadata_collected_utc":("该分区本轮采集开始时点；不是逐源精确读取时刻","ISO8601 UTC","带+00:00的UTC字符串"),
"cutout_sdss_offset_arcsec":("文件名切图中心到SDSS输入位置的球面角距离","arcsec","0—3"),
"cutout_wise_offset_arcsec":("文件名切图中心到CatWISE位置的球面角距离","arcsec","非负有限值")
}
 fields=[]
 for name,s in summary["field_statistics"].items():
  if name in originals:
   old=originals[name];meaning=old["meaning"];unit=old["unit"];allowed=old["allowed_range_note"];source=old["source"]
   if name=="w1sigflux":meaning="W1原始轮廓拟合流量不确定度；CatWISE说明文字band 2疑似排版问题，按字段、WISE说明及信噪比关系解释"
   time_info="CatWISE2020资料或SDSS原输入表；逐源原始观测时点未提供。IRSA历史导出2022-04-19 04:31:09（时区未标明）；本轮导出时点见catalog_transfer.json。"
   missing_reason="当前融合样本按已记录缺失规则统计；不得把未检出显式缺失等同于已验证全部业务哨兵值。"
  else:
   meaning,unit,allowed=meta[name];source="Pan-STARRS实际FITS文件元数据"
   if name in ("cutout_sdss_offset_arcsec","cutout_wise_offset_arcsec"):source="融合衍生：真实文件名切图中心与原星表坐标"
   time_info="本轮分区采集时间见metadata_collected_utc；标称观测时点仅在已读取头时由r_mjd_obs提供。r_mtime_ns为文件修改时间，不替代观测或采集时间。"
   missing_reason="未读取头造成的结构性未采集、原头未提供或旧文件名未编码；按r_metadata_scope区分；不得用统计填补伪造日期、单位或头校验值。"
  fields.append({"name":name,"meaning":meaning,"type":s["type"],"unit":unit,"allowed_range":allowed,"observed_min":s.get("min"),"observed_max":s.get("max"),"missing_count":s["missing_count"],"missing_pct":s["missing_pct"],"source":source,"source_time":time_info,"missing_interpretation":missing_reason})
 refs=[
 {"title":"CatWISE字段定义","url":"https://irsa.ipac.caltech.edu/data/WISE/CatWISE/gator_docs/catwise_colDescriptions.html"},
 {"title":"WISE流量与流量误差定义","url":"https://irsa.ipac.caltech.edu/data/WISE/docs/release/All-Sky/expsup/sec2_2a.html"},
 {"title":"PS1 Stack Images与EXPTIME说明","url":"https://outerspace.stsci.edu/spaces/PANSTARRS/pages/298812339/PS1%2BStack%2BImages"}]
 dictionary={"status":"实测融合表对应的数据字典","row_count":summary["effective_rows"],"field_count":len(fields),"dataset_sha256":summary["sha256"],"references":refs,"fields":fields}
 (DOC/"数据字典_正式版.json").write_text(json.dumps(dictionary,ensure_ascii=False,indent=2)+"\n")
 lines=["# 融合分析表数据字典","","有效记录："+format(summary["effective_rows"],",")+"；字段："+str(len(fields))+"。","",
 "完整的逐字段来源时点与缺失解释见同名JSON。实测范围与合法范围分别记录；文件修改时间、历史导出时间、观测MJD和本轮采集时点不混用。","",
 "| 字段 | 含义 | 类型 | 单位 | 合法范围/取值 | 实测范围 | 缺失率 | 来源 |","|---|---|---|---|---|---|---|---|"]
 for f in fields:
  ran="—" if f["observed_min"] is None else str(f["observed_min"])+" ～ "+str(f["observed_max"])
  clean=lambda x:str(x).replace("|","/")
  lines.append("| "+" | ".join(map(clean,[f["name"],f["meaning"],f["type"],f["unit"],f["allowed_range"],ran,str(f["missing_pct"])+"%",f["source"]]))+" |")
 lines.extend(["","## 来源与时点","",
 "星表采用原始SDSS输入标签与CatWISE观测字段。原SDSS具体版本未独立核验，保留来源不确定性。CatWISE2020为2010—2018资料构建的目录，不能给每源强行指定同一个观测日期。",
 "IRSA文件头历史导出时点为2022-04-19 04:31:09，未注明时区；本轮只读获取时间记录于results/acquisition/catalog_transfer.json和各分区采集清单。",
 "file_only记录未采集的头字段不是观测自然缺失；BUNIT不提供时单位保持未知。EXPTIME是头中标称输入曝光之和，不等同于逐像素有效曝光。",
 "W1误差字段的CatWISE说明文字存在疑似band 2排版问题；本字典依据字段名和WISE对应定义解释为W1，并保留该差异说明。",""])
 lines.extend("- ["+r["title"]+"]("+r["url"]+")" for r in refs)
 (DOC/"数据字典_正式版.md").write_text("\n".join(lines)+"\n")
 classes="\n".join("| "+k+" | "+format(v,",")+" | "+f"{v/summary['effective_rows']*100:.4f}% |" for k,v in summary["classes"].items())
 scopes=summary["metadata_scopes"];reasons="\n".join("- "+k+"："+format(v,",")+"条。" for k,v in summary["rejection_reasons"].items())
 report=f"""# 数据获取与融合验收报告

本报告只验收模板的“数据获取”三项，不表示数据治理、建模和最终课程报告已全部完成。

## ① 两类异构数据

结构化源为SDSS×CatWISE交叉匹配星表，原始实测{summary['source_rows']:,}条、24字段。半结构化源为从真实Pan-STARRS FITS文件提取的JSONL图像元数据，包含文件路径、五波段结构、r文件属性和文件名切图信息；部分记录另有实际FITS头。不是把同一星表转换JSON后当作新来源。全部采集档案及拒绝记录已保存在本机，原始图像继续留在服务器。

## ② 有效融合记录数

| 阶段 | 数量 |
|---|---:|
| 原始星表 | {summary['source_rows']:,} |
| 固定哈希规则选中候选 | {summary['hash_selected_rows']:,} |
| 文件元数据接受 | {summary['metadata_accepted']:,} |
| 文件元数据拒绝 | {summary['metadata_rejected']:,} |
| 关联/坐标等融合拒绝 | {summary['fusion_rejected']:,} |
| 最终有效天体记录 | **{summary['effective_rows']:,}** |
| 主分析表字段 | **{summary['fields']}** |
| 最终独立source_id | {summary['unique_source_ids']:,} |

满足“融合后不少于100万条”这一数量路径。每条对应一个唯一source_id，不把五波段或增强图片重复计数；本轮未证明所有物理对象互异或图像空间不重叠。使用source_id做1:1连接；切图文件名中心与SDSS输入位置相距不超过3角秒。

| 类别 | 数量 | 占比 |
|---|---:|---:|
{classes}

排除原因：
{reasons}

## ③ 数据字典、脚本与采集时间

- docs/数据字典_正式版.md与JSON：全部字段的名称、含义、类型、单位、合法/实测范围、缺失率、来源与时点。
- src/copy_catalog_readonly.py：原星表读取和SHA-256复核。
- src/collect_million_metadata_readonly.py与run_metadata_collection.py：服务器只读采集、本机按分区归档和断点续采。
- src/build_fused_dataset.py：哈希选样、融合、统计和数据断言。
- results/acquisition/catalog_transfer.json、catalog_import.json、各运行protocol/log、分区manifests：实际采集与处理时间、输入身份和运行环境。
- docs/历史IRSA查询头.txt：保留旧文件中的原始查询参数；并未声称已重建2022年的完整采集链。

本次以“复用既有项目公开巡天资料的只读再采集”交付。历史原始下载时间不明；原SDSS输入版本未独立核验，均如实保留，不以本轮采集时间替代旧观测时间。

## 质量检查范围

file_only：{scopes.get('file_only',0):,}源；header_and_file：{scopes.get('header_and_file',0):,}源。file_only通过实际文件清单、五波段唯一性、r文件字节长度和关联位置检查；没有逐源验证完整FITS头与像素。已读头记录来自最初分区，不应当作随机总体质量估计。

像素内容的独立抽检见第二阶段检查报告：300源、1500图中21图含非有限像素。这不等于本轮百万级样本全部通过像素验收。pixel_quality_checked为false不应被改成true。未采集的头字段和未提供的BUNIT不能用统计填补造值。

## 交付文件与复核

主表：data/processed/fused_sources.parquet（{summary['file_bytes']:,}字节）。

SHA-256：{summary['sha256']}

原星表SHA-256：{summary['raw_catalog_sha256']}

验证了原始行数、源ID唯一性、395分区完整、原始元数据归档哈希、每分区计数、1:1关联、标签/坐标/测光、切图中心偏移、百万条阈值以及Parquet行列数。服务器全程只读，未进行训练。

## 本阶段可用于报告的文字

本项目复用既有SDSS×CatWISE交叉匹配星表，并从Pan-STARRS五波段FITS切图提取文件元数据，构建异构数据分析集。以固定源ID哈希规则确定候选，再进行五波段文件结构、代表波段文件属性、主键和位置关联检查，最终得到{summary['effective_rows']:,}条独立天体记录、{summary['fields']}个字段。采集脚本、数据字典、时间记录、拒绝记录、输入输出校验值及执行日志均一并存档。该数据集是星表与图像元数据的融合结果，完整图像像素质量仍按单独流程审计。
"""
 (DOC/"数据获取与融合_验收报告.md").write_text(report)
 (DOC/"历史IRSA查询头.txt").write_text("\n".join(imported["original_header"])+"\n")
 (BASE/"data/processed/model_feature_scope.json").write_text(json.dumps({
 "status":"输入范围建议，尚未训练或做数据依赖特征选择",
 "baseline_numeric_features":["w1sky","w1sigsk","w2sky","w2sigsk","w1snr","w2snr","w1flux","w1sigflux","w2flux","w2sigflux","w1mpro","w1sigmpro","w2mpro","w2sigmpro"],
 "label":"class_01","group_key":"source_id",
 "exclude_from_default_predictors":["source_id","source_name","cntr_01","all file paths","partition","r_mtime_ns","r_header_sha256","metadata_collected_utc","r_metadata_scope","partially collected FITS header fields"],
 "note":"工程字段用于追溯与质量分群；不能默认作为天体分类特征。未来填补实验不填造观测时点和元数据采集深度。"
 },ensure_ascii=False,indent=2))
 print(json.dumps({"dictionary_fields":len(fields),"effective_rows":summary["effective_rows"],"report":str(DOC/"数据获取与融合_验收报告.md")},ensure_ascii=False))
if __name__=="__main__":main()
