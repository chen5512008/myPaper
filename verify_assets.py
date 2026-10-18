from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
items=json.loads((root/'ASSETS.json').read_text(encoding='utf-8'))
missing=[]
for item in items:
    path=(root/item['path']).resolve()
    if not path.is_relative_to(root.resolve()):raise SystemExit('Unsafe manifest path')
    if not path.is_file():missing.append(item['path']);continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    if h.hexdigest()!=item['sha256']:raise SystemExit('Hash mismatch: '+item['path'])
if missing:raise SystemExit('Extract reproduction-assets.zip first; missing '+str(len(missing))+' files. First: '+missing[0])
print('Verified '+str(len(items))+' snapshot files')
