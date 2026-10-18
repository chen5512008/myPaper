"""Repair WPS truncated UTF-16 mappings only; verify identical rendered page pixels."""
from pathlib import Path
from io import BytesIO
import re,json,hashlib,argparse
import pymupdf as fitz
from fontTools.ttLib import TTFont
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
src=Path(a.input);dst=Path(a.output);assert src!=dst and not dst.exists()
d=fitz.open(src);before=[hashlib.sha256(pg.get_pixmap(matrix=fitz.Matrix(1,1)).samples).hexdigest() for pg in d];bad_before=sum(pg.get_text().count('\ufffd') for pg in d)
seen=set();changes=[]
for pg in d:
    for fo in pg.get_fonts(full=True):
        xref=fo[0]
        if xref in seen:continue
        seen.add(xref)
        typ,ref=d.xref_get_key(xref,'ToUnicode')
        if typ!='xref':continue
        cmapxref=int(ref.split()[0]);stream=d.xref_stream(cmapxref)
        pairs=re.findall(rb'<([0-9A-Fa-f]{4})>\s*<([dD][89aAbB][0-9a-fA-F]{2})>',stream)
        if not pairs:continue
        desc=d.xref_get_key(xref,'DescendantFonts')[1];child=int(re.search(r'(\d+)\s+0\s+R',desc).group(1))
        assert d.xref_get_key(child,'CIDToGIDMap')[1]=='/Identity'
        fontbytes=d.extract_font(xref)[3]
        try:tt=TTFont(BytesIO(fontbytes))
        except Exception:tt=TTFont(BytesIO(fontbytes),fontNumber=0)
        order=tt.getGlyphOrder();reverse={}
        for cp,gn in tt.getBestCmap().items():reverse.setdefault(gn,[]).append(cp)
        for code,high in pairs:
            gid=int(code,16);choices=[cp for cp in reverse.get(order[gid],[]) if cp>0xffff and int.from_bytes(chr(cp).encode('utf-16-be')[:2],'big')==int(high,16)]
            assert len(choices)==1,(fo[3],gid,choices)
            dest=chr(choices[0]).encode('utf-16-be').hex().upper().encode()
            rx=rb'<'+code+rb'>\s*<'+high+rb'>'
            stream,n=re.subn(rx,b'<'+code+b'> <'+dest+b'>',stream);assert n==1
            changes.append({'font':fo[3],'gid':gid,'unicode':hex(choices[0])})
        d.update_stream(cmapxref,stream)
d.save(dst,garbage=3,deflate=True);d.close();fixed=fitz.open(dst)
after=[hashlib.sha256(pg.get_pixmap(matrix=fitz.Matrix(1,1)).samples).hexdigest() for pg in fixed];bad_after=sum(pg.get_text().count('\ufffd') for pg in fixed)
assert before==after,'Mapping repair must not alter visible page pixels'
assert bad_after==0,(bad_before,bad_after)
report={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'pages':len(fixed),'pages_pixel_identical':True,'replacement_characters_before':bad_before,'replacement_characters_after':bad_after,'mapping_entries_repaired':changes,'scope':'Only truncated surrogate mappings; no glyph substitution, no rasterization, native DOCX equations unchanged.'}
dst.with_suffix('.mapping_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='mapping_entries_repaired'},indent=2))
