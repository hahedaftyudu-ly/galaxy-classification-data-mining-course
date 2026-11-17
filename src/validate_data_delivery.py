#!/usr/bin/env python3
"""Independent streaming checks of the delivered Parquet and dictionary."""
from pathlib import Path
import collections,datetime,hashlib,json,math
from zoneinfo import ZoneInfo
import numpy as np
import pyarrow.parquet as pq
BASE=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
 return h.hexdigest()
def main():
 s=json.loads((BASE/"results/acquisition/fusion_summary.json").read_text())
 d=json.loads((BASE/"docs/数据字典_正式版.json").read_text())
 path=BASE/"data/processed/fused_sources.parquet";pf=pq.ParquetFile(path)
 assert pf.metadata.num_rows>=1000000
 assert pf.metadata.num_rows==s["effective_rows"]
 assert len(pf.schema_arrow)==s["fields"]==d["field_count"]>=20
 assert set(f["name"] for f in d["fields"])==set(pf.schema_arrow.names)
 assert sha(path)==s["sha256"]
 ids=set();counts=collections.Counter();header_counts=collections.Counter();rows=0
 columns=["source_id","class_01","dist_x","cutout_sdss_offset_arcsec","band_count","r_bytes","r_metadata_scope","pixel_quality_checked","g_path","r_path","i_path","z_path","y_path"]
 for batch in pf.iter_batches(batch_size=65536,columns=columns):
  x=batch.to_pydict();n=len(x["source_id"]);rows+=n
  for i,sid in enumerate(x["source_id"]):
   assert sid and sid not in ids;ids.add(sid)
   assert int.from_bytes(hashlib.blake2b(sid.encode(),digest_size=8).digest(),"big")%3==0
   assert x["class_01"][i] in ("GALAXY","QSO","STAR")
   assert 0<=x["dist_x"][i]<=3 and 0<=x["cutout_sdss_offset_arcsec"][i]<=3
   assert x["band_count"][i]==5
   assert x["r_bytes"][i]>0 and x["r_bytes"][i]%2880==0
   assert x["pixel_quality_checked"][i] is False
   for b in "grizy":assert Path(x[b+"_path"][i]).parent.name==sid
  counts.update(x["class_01"]);header_counts.update(x["r_metadata_scope"])
 assert rows==len(ids)==s["effective_rows"]
 assert dict(counts)==s["classes"] and dict(header_counts)==s["metadata_scopes"]
 result={"validated_at":datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"passed":True,"rows":rows,"unique_ids":len(ids),"fields":s["fields"],"classes":counts,"metadata_scopes":header_counts,"checks":["million-row threshold","dictionary covers every delivered field","output SHA256","global source-ID uniqueness","hash selection rule","label domain","match/cutout offset ranges","five bands","r FITS byte length","all five path parent IDs","pixel audit scope preserved","summary class/scope totals"]}
 (BASE/"results/acquisition/delivery_validation.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":main()
