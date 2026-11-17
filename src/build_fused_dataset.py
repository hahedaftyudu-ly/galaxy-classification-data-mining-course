#!/usr/bin/env python3
"""Build and validate the local million-source heterogeneous analysis table."""
import collections, datetime, gzip, hashlib, json, math, pathlib, time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
BASE=pathlib.Path(__file__).resolve().parents[1]
RAW=BASE/"data/raw";OUT=BASE/"data/processed";RES=BASE/"results/acquisition"
CATALOG=RAW/"SDSSxWISE_cat.tbl.gz"
STRINGS={"class_01","source_name","source_id"}
META_TYPES={
"partition":pa.string(),"image_directory":pa.string(),
"g_path":pa.string(),"r_path":pa.string(),"i_path":pa.string(),"z_path":pa.string(),"y_path":pa.string(),
"band_count":pa.int16(),"r_bytes":pa.int64(),"r_mtime_ns":pa.int64(),"r_header_sha256":pa.string(),
"r_header_bytes":pa.int32(),"r_width":pa.int16(),"r_height":pa.int16(),"r_bitpix":pa.int16(),
"r_filter":pa.string(),"r_metadata_scope":pa.string(),"r_mjd_obs":pa.float64(),"r_exptime":pa.float64(),
"r_bunit":pa.string(),"r_ctype1":pa.string(),"r_ctype2":pa.string(),
"r_crval1":pa.float64(),"r_crval2":pa.float64(),"r_crpix1":pa.float64(),"r_crpix2":pa.float64(),
"r_cdelt1":pa.float64(),"r_cdelt2":pa.float64(),"cutout_ra":pa.float64(),
"cutout_dec":pa.float64(),"cutout_arcsec":pa.float64(),"pixel_quality_checked":pa.bool_(),
"metadata_collected_utc":pa.string(),"cutout_sdss_offset_arcsec":pa.float64(),"cutout_wise_offset_arcsec":pa.float64()}
def now():return datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
 return h.hexdigest()
def sep(ra1,dec1,ra2,dec2):
 r1=np.deg2rad(np.asarray(ra1,dtype=float));r2=np.deg2rad(np.asarray(ra2,dtype=float))
 d1=np.deg2rad(np.asarray(dec1,dtype=float));d2=np.deg2rad(np.asarray(dec2,dtype=float))
 a=np.sin((d2-d1)/2)**2+np.cos(d1)*np.cos(d2)*np.sin((r2-r1)/2)**2
 return np.rad2deg(2*np.arcsin(np.sqrt(np.clip(a,0,1))))*3600
def dump(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def log(message,obj):print(message+" "+json.dumps(obj,ensure_ascii=False),flush=True)
def load_catalog(fresh=False):
 destination=OUT/"catalog_selected.parquet"
 if destination.exists() and not fresh:
  return pd.read_parquet(destination).set_index("source_id",drop=False),json.loads((RES/"catalog_import.json").read_text())
 metadata=[];names=None
 f=gzip.open(CATALOG,"rt")
 while True:
  pos=f.tell();line=f.readline()
  if not line:raise RuntimeError("no catalog data")
  if line.startswith("\\"):metadata.append(line.rstrip())
  elif line.startswith("|"):
   if names is None:names=[x.strip() for x in line.rstrip().split("|")[1:-1]]
  elif line.strip():f.seek(pos);break
 schema=pa.schema([(n,pa.string() if n in STRINGS else pa.int64() if n=="cntr_01" else pa.float64()) for n in names])
 dtype={n:"string" if n in STRINGS else "int64" if n=="cntr_01" else "float64" for n in names}
 selected=[];rows=0;class_counts=collections.Counter();selected_counts=collections.Counter()
 partial=destination.with_suffix(".parquet.partial");writer=pq.ParquetWriter(partial,schema,compression="zstd",compression_level=3)
 started=now();timer=time.monotonic()
 try:
  for chunk in pd.read_csv(f,sep=r"\s+",names=names,dtype=dtype,chunksize=100000,na_values=["null","NaN","nan","--"],keep_default_na=True):
   rows+=len(chunk);class_counts.update(chunk["class_01"].value_counts().to_dict())
   assert chunk["source_id"].notna().all()
   mask=chunk["source_id"].map(lambda x:int.from_bytes(hashlib.blake2b(x.encode(),digest_size=8).digest(),"big")%3==0)
   take=chunk.loc[mask].copy()
   assert take[list(n for n in names if n not in STRINGS)].notna().all().all()
   selected_counts.update(take["class_01"].value_counts().to_dict())
   selected.append(take)
   writer.write_table(pa.Table.from_pandas(take,schema=schema,preserve_index=False))
   if rows%500000==0:log("CATALOG_IMPORT",{"rows":rows,"selected":sum(len(x) for x in selected)})
 finally:writer.close();f.close()
 assert rows==3944564,rows
 table=pd.concat(selected,ignore_index=True)
 assert table["source_id"].is_unique
 partial.replace(destination)
 result={"started_at":started,"completed_at":now(),"elapsed_s":round(time.monotonic()-timer,3),"raw_rows":rows,"selected_rows":len(table),"class_rows":class_counts,"selected_classes":selected_counts,"columns":names,"original_header":metadata,"raw_file_bytes":CATALOG.stat().st_size,"raw_sha256":sha256(CATALOG),"selection":"BLAKE2b(source_id,digest_size=8) modulo3 equals0","path":str(destination)}
 dump(RES/"catalog_import.json",result)
 return table.set_index("source_id",drop=False),result
def main():
 OUT.mkdir(parents=True,exist_ok=True);RES.mkdir(parents=True,exist_ok=True)
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument("--fresh-catalog",action="store_true");args=ap.parse_args()
 started=now();timer=time.monotonic();catalog,imported=load_catalog(fresh=args.fresh_catalog)
 cat_schema=pq.read_schema(OUT/"catalog_selected.parquet")
 schema=pa.schema(list(cat_schema)+[pa.field(k,v) for k,v in META_TYPES.items()])
 raw=RAW/"image_metadata";manifests=raw/"manifests"
 parts=sorted(manifests.glob("images*.json"),key=lambda p:int(p.stem[6:]))
 assert len(parts)==395,("incomplete collection",len(parts))
 dest=OUT/"fused_sources.parquet";partial=dest.with_suffix(".parquet.partial")
 writer=pq.ParquetWriter(partial,schema,compression="zstd",compression_level=3)
 chunks=[];buffered=0;total=0;selected=0;metadata_accepted=0;metadata_rejected=0
 seen=set();classes=collections.Counter();reasons=collections.Counter();rejection_examples=[]
 scopes=collections.Counter();numeric={};missing=collections.Counter()
 rejected_path=RES/"fusion_rejected.jsonl";rejected_file=rejected_path.open("w")
 try:
  for partfile in parts:
   manifest=json.loads(partfile.read_text());path=raw/(partfile.stem+".jsonl.gz")
   assert sha256(path)==manifest["archive_sha256"],("archive mismatch",path)
   rows=[];valid=bad=0
   with gzip.open(path,"rt") as f:
    for line in f:
     r=json.loads(line)
     if r["kind"]=="rejected":
      bad+=1;reasons["metadata:"+r["reason"]]+=1
      rejected_file.write(json.dumps(r,ensure_ascii=False)+"\n")
     elif r["kind"]=="image":
      valid+=1
      assert r["source_id"] not in seen,("duplicate image source",r["source_id"])
      seen.add(r["source_id"])
      r.pop("kind");r["r_metadata_scope"]=r.get("r_metadata_scope") or ("header_and_file" if r.get("r_header_sha256") else "file_only")
      r["metadata_collected_utc"]=manifest["collected_at_start"];rows.append(r)
   assert valid==manifest["accepted"] and bad==manifest["rejected"]
   selected+=valid+bad;metadata_accepted+=valid;metadata_rejected+=bad
   frame=pd.DataFrame(rows).set_index("source_id",drop=True)
   absent=frame.index.difference(catalog.index)
   assert len(absent)==0,("metadata without selected catalog",absent.tolist()[:3])
   joined=catalog.loc[frame.index].join(frame,how="inner",validate="one_to_one")
   assert len(joined)==valid
   joined["cutout_sdss_offset_arcsec"]=sep(joined["cutout_ra"],joined["cutout_dec"],joined["ra_01"],joined["dec_01"])
   joined["cutout_wise_offset_arcsec"]=sep(joined["cutout_ra"],joined["cutout_dec"],joined["ra"],joined["dec"])
   coord=np.isfinite(joined[["ra_01","dec_01","ra","dec"]].to_numpy(dtype=float)).all(axis=1)
   phot=np.isfinite(joined[["w1flux","w2flux","w1mpro","w2mpro"]].to_numpy(dtype=float)).all(axis=1)
   ok=coord & phot & joined["class_01"].isin(["GALAXY","QSO","STAR"]).to_numpy() & joined["dist_x"].between(0,3).to_numpy() & joined["cutout_sdss_offset_arcsec"].between(0,3).to_numpy() & (joined["band_count"]==5).to_numpy() & (joined["r_bytes"]>0).to_numpy() & (joined["r_bytes"]%2880==0).to_numpy()
   for _,r in joined.loc[~ok].iterrows():
    info={"kind":"fusion_rejected","source_id":r["source_id"],"partition":partfile.stem,"reason":"invalid label/coordinates/photometry/match or cutout center >3arcsec","cutout_sdss_offset_arcsec":float(r["cutout_sdss_offset_arcsec"]) if np.isfinite(r["cutout_sdss_offset_arcsec"]) else None}
    reasons["fusion_invalid"]+=1;rejected_file.write(json.dumps(info,ensure_ascii=False)+"\n")
    if len(rejection_examples)<20:rejection_examples.append(info)
   joined=joined.loc[ok].copy()
   for k in META_TYPES:
    if k not in joined:joined[k]=None
   joined=joined[schema.names]
   table=pa.Table.from_pandas(joined,schema=schema,preserve_index=False,safe=True)
   chunks.append(table);buffered+=len(table);total+=len(table)
   classes.update(joined["class_01"].value_counts().to_dict());scopes.update(joined["r_metadata_scope"].value_counts().to_dict())
   for field in schema:
    col=table.column(field.name);missing[field.name]+=col.null_count
    if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
     series=joined[field.name].dropna()
     if len(series):
      if pa.types.is_integer(field.type):lo,hi=int(series.min()),int(series.max())
      else:
       arr=series.to_numpy(dtype=float);arr=arr[np.isfinite(arr)]
       if not len(arr):continue
       lo,hi=float(arr.min()),float(arr.max())
      s=numeric.setdefault(field.name,{"min":lo,"max":hi})
      s["min"]=min(s["min"],lo);s["max"]=max(s["max"],hi)
   if buffered>=50000:writer.write_table(pa.concat_tables(chunks));chunks=[];buffered=0
   if len(seen)%100000<valid:log("FUSION_PROGRESS",{"metadata_ids":len(seen),"effective_rows":total,"part":partfile.stem})
  if chunks:writer.write_table(pa.concat_tables(chunks))
 finally:writer.close();rejected_file.close()
 assert total>=1000000,("effective records below threshold",total)
 assert selected==imported["selected_rows"],("selected census mismatch",selected,imported["selected_rows"])
 assert metadata_accepted+metadata_rejected==selected
 assert total+sum(v for k,v in reasons.items() if not k.startswith("metadata:"))==metadata_accepted
 check=pq.ParquetFile(partial);assert check.metadata.num_rows==total and len(check.schema_arrow)==len(schema)
 ids=check.read(columns=["source_id"]).column("source_id")
 assert len(set(ids.to_pylist()))==total
 assert missing["source_id"]==0 and missing["class_01"]==0
 partial.replace(dest)
 stats={n:{"type":str(schema.field(n).type),"missing_count":missing[n],"missing_pct":round(missing[n]/total*100,6),**numeric.get(n,{})} for n in schema.names}
 summary={"status":"completed","scope":"data acquisition and heterogeneous metadata fusion; not full pixel quality or model training","started_at":started,"completed_at":now(),"elapsed_s":round(time.monotonic()-timer,3),"source_rows":imported["raw_rows"],"hash_selected_rows":imported["selected_rows"],"partitions":len(parts),"metadata_accepted":metadata_accepted,"metadata_rejected":metadata_rejected,"fusion_rejected":metadata_accepted-total,"effective_rows":total,"fields":len(schema),"unique_source_ids":total,"join_cardinality":"1:1 source_id","classes":classes,"metadata_scopes":scopes,"rejection_reasons":reasons,"fusion_rejection_examples":rejection_examples,"field_statistics":stats,"file":str(dest),"file_bytes":dest.stat().st_size,"sha256":sha256(dest),"raw_catalog_sha256":imported["raw_sha256"],"assertions":["raw row count","selected source IDs unique","395 partitions completed","archive SHA256","per-part row counts","metadata IDs unique","all metadata IDs have selected catalog match","1:1 join","labels/coordinates/photometry/match quality","cutout center within3arcsec","effective records>=1million","Parquet row count/field count","final IDs unique/nonmissing"]}
 dump(RES/"fusion_summary.json",summary);log("FUSION_COMPLETE",summary)
if __name__=="__main__":main()
