#!/usr/bin/env python3
"""Fill the retained course template with result-backed report Markdown.
Run with the Codex bundled Python (python-docx), then render with render_docx.py.
"""
import copy,hashlib,json,re
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from docx import Document
from docx.shared import Pt,Cm,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
B=Path(__file__).resolve().parents[1]
REFERENCE=B/'src/resources/course_template.docx'
OUT=B/'docs/期末报告_星系分类网络.docx'

def font(run,name='宋体',size=12,bold=False):
 run.font.name='Times New Roman';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor(0,0,0)
 run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),name)

def rich(p,text,size=12):
 for part in re.split(r'(\*\*.*?\*\*|`[^`]+`)',text):
  if not part:continue
  bold=part.startswith('**');code=part.startswith('`');content=part[2:-2] if bold else part[1:-1] if code else part
  # Long identifiers and paths can wrap at punctuation without changing content.
  r=p.add_run(content);font(r,size=size,bold=bold)
  if code:r.font.name='Courier New';r.font.size=Pt(min(size,9))

def border(cell):
 pr=cell._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
 for side in ('top','left','bottom','right'):
  el=OxmlElement('w:'+side);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');b.append(el)
 pr.append(b);m=OxmlElement('w:tcMar')
 for side in ('top','left','bottom','right'):
  el=OxmlElement('w:'+side);el.set(qn('w:w'),'65');el.set(qn('w:type'),'dxa');m.append(el)
 pr.append(m)

def main():
 d=Document(REFERENCE); first_sec=copy.deepcopy(d.sections[0]._sectPr)
 body=d.element.body; cover=list(body)[:20]
 for el in list(body)[:-1]:body.remove(el)
 for el in cover:body.insert(len(body)-1,el)
 # Cover filling uses the reference's centered paragraph roles.
 texts={6:'题    目：星系分类网络',7:'小组成员：____________________',8:'____________________',9:'____________________',10:'成    绩：____________________',19:'2026 年 11 月 16 日'}
 for i,p in enumerate(d.paragraphs):
  if i in texts:
   p.clear();r=p.add_run(texts[i]);font(r,'黑体',14)
  # Cover blanks are writable spacing slots, compacted for a one-page cover.
  if not p.text.strip():p.paragraph_format.line_spacing=Pt(9);p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
  if i in (1,2):p.paragraph_format.line_spacing=Pt(28)
  if i in (4,5):
   for r in p.runs:font(r,'黑体',26,True)
   p.paragraph_format.line_spacing=1.5
  p.paragraph_format.keep_with_next=False
 from docx.enum.style import WD_STYLE_TYPE
 for name in ("Title","Heading 1","Heading 2","Heading 3"):
  if name not in d.styles:d.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
 d.paragraphs[5].style='Title'
 # Preserve the source two-section model: cover then body, same page geometry.
 p=d.add_paragraph();p._p.get_or_add_pPr().append(first_sec)
 for name,size,bold in [('Normal',12,False),('Title',26,True),('Heading 1',14,True),('Heading 2',12,True),('Heading 3',12,True)]:
  st=d.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.bold=bold;st.font.color.rgb=RGBColor(0,0,0)
  st.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体' if name=='Normal' else '黑体')
  fmt=st.paragraph_format;fmt.space_before=Pt(10 if 'Heading' in name else 0);fmt.space_after=Pt(6);fmt.line_spacing=1.3
  if 'Heading' in name:fmt.keep_with_next=True
 raw=(B/'docs/期末报告_星系分类网络.md').read_text().splitlines();i=0
 # Report starts after the cover metadata. Keep the actual report topic visible.
 while i<len(raw) and raw[i]!='## 摘要':i+=1
 while i<len(raw):
  text=raw[i].strip()
  if not text:i+=1;continue
  if text.startswith('```'):
   code=[];i+=1
   while i<len(raw) and not raw[i].startswith('```'):code.append(raw[i]);i+=1
   # Preserve code indentation, fit at source body width without clipping.
   for line in code:
    p=d.add_paragraph();p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.1;p.paragraph_format.keep_with_next=True
    r=p.add_run(line);font(r,size=8.5);r.font.name='Courier New'
   if code:p.paragraph_format.keep_with_next=False
   i+=1;continue
  if text.startswith('|'):
   block=[]
   while i<len(raw) and raw[i].strip().startswith('|'):block.append(raw[i].strip());i+=1
   rows=[[x.strip() for x in line.strip('|').split('|')] for line in block]
   rows=[r for r in rows if not all(re.fullmatch(r':?-+:?',x or '-') for x in r)]
   cols=len(rows[0]);t=d.add_table(rows=0,cols=cols);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
   total=Cm(14.65);weights=[max(3,min(28,max(len(r[j]) if j<len(r) else 0 for r in rows))) for j in range(cols)]
   if cols>=5:weights=[max(5,min(18,w)) for w in weights]
   if cols==2:weights=[14,30]
   for j,w in enumerate(weights):t.columns[j].width=int(total*w/sum(weights))
   for n,row in enumerate(rows):
    cells=t.add_row().cells
    for j,text2 in enumerate(row):
     cell=cells[j];cell.width=int(total*weights[j]/sum(weights));cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;border(cell)
     p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3);p.paragraph_format.line_spacing=Pt(13)
     snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'false');p._p.get_or_add_pPr().append(snap)
     p.paragraph_format.keep_with_next=n<len(rows)-1
     p.alignment=WD_ALIGN_PARAGRAPH.LEFT if len(text2)>16 else WD_ALIGN_PARAGRAPH.CENTER
     rich(p,text2,size=9 if cols>=5 else 10)
     if n==0:
      for rr in p.runs:rr.bold=True
      fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'F0F0F0');cell._tc.get_or_add_tcPr().append(fill)
    pr=t.rows[-1]._tr.get_or_add_trPr();no=OxmlElement('w:cantSplit');pr.append(no)
    if n==0:rep=OxmlElement('w:tblHeader');pr.append(rep)
   d.add_paragraph().paragraph_format.space_after=Pt(0)
   continue
  match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',text)
  if match:
   caption,path=match.groups();f=(B/'docs'/path).resolve()
   p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.paragraph_format.space_before=Pt(2);p.paragraph_format.space_after=Pt(0)
   from PIL import Image
   with Image.open(f) as im:w,h=im.size
   max_height=16.5 if f.name.startswith("fig03") else 18.5 if f.name.startswith("fig13") else 14.0 if f.name.startswith("fig11") else 9.2
   width=min(14.65,max_height*w/h);p.add_run().add_picture(str(f),width=Cm(width))
   p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(8);p.paragraph_format.line_spacing=1.1
   rich(p,caption,size=10)
  elif text.startswith('#'):
   level=len(text)-len(text.lstrip('#'));title=text[level:].strip()
   p=d.add_paragraph(title,style='Heading 1' if level<=2 else 'Heading 2')
   for r in p.runs:font(r,'黑体',14 if level<=2 else 12,True)
  else:
   p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY;p.paragraph_format.first_line_indent=Cm(.74);p.paragraph_format.widow_control=True
   rich(p,text)
   if text.startswith(("局部样本", "案例", "规则")):p.paragraph_format.keep_together=True
  i+=1
 # Source grid made auto line spacing jump to ~40pt. Disable snapping in
 # writable body slots and use readable deterministic text line heights.
 for p in d.paragraphs[21:]:
  if p.style is None:p.style='Normal'
  snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'false');p._p.get_or_add_pPr().append(snap)
  if p._p.findall('.//'+qn('w:drawing')):p.paragraph_format.line_spacing=1
  elif p.style.name.startswith('Heading'):p.paragraph_format.line_spacing=Pt(19)
  elif p.runs and all(r.font.name=='Courier New' for r in p.runs):p.paragraph_format.line_spacing=Pt(12)
  elif p.runs and all(r.font.size and r.font.size.pt<=10 for r in p.runs):p.paragraph_format.line_spacing=Pt(14)
  else:p.paragraph_format.line_spacing=Pt(20)
 setting=OxmlElement('w:updateFields');setting.set(qn('w:val'),'true');d.settings.element.append(setting)
 temp=OUT.with_name('.report_work.docx');d.save(temp)
 # python-docx may serialize untouched parts. Restore all preserve-only package
 # parts byte-for-byte to avoid header/footer/theme/numbering regressions.
 editable={'word/document.xml','word/styles.xml','word/settings.xml','word/_rels/document.xml.rels','[Content_Types].xml','docProps/core.xml','docProps/app.xml'}
 preserved=[]
 with ZipFile(REFERENCE) as ref,ZipFile(temp) as new,ZipFile(OUT,'w',ZIP_DEFLATED) as dest:
  for info in new.infolist():
   n=info.filename
   data=ref.read(n) if n in ref.namelist() and n not in editable else new.read(n)
   dest.writestr(info,data)
   if n in ref.namelist() and n not in editable:preserved.append(n)
  for n in ref.namelist():
   if n not in new.namelist() and n not in editable:dest.writestr(n,ref.read(n));preserved.append(n)
 temp.unlink()
 with ZipFile(REFERENCE) as ref,ZipFile(OUT) as final:
  assert all(ref.read(n)==final.read(n) for n in preserved)
 assert len(Document(OUT).sections)==2
 log={'reference_sha256':hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'preserved_parts':preserved,'sections':2,'images':len(Document(OUT).inline_shapes)}
 (B/'results/review_20261116/docx_template_fidelity.json').write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n')
 print(OUT)
if __name__=='__main__':main()
