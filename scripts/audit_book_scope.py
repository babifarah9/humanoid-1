"""Check every numbered TOC entry against the source-aligned requirements.

python scripts/audit_book_scope.py --text toc.txt
python scripts/audit_book_scope.py --docx master_toc.docx
This is coverage of topics, not evidence that features are implemented.
"""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile


def normalize(text):
    return ' '.join(text.split())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--text');source.add_argument('--docx')
    p.add_argument('--catalog',default=str(Path(__file__).resolve().parents[1]/'requirements/book_features.json'))
    a=p.parse_args()
    if a.text:text=Path(a.text).read_text()
    else:
        with zipfile.ZipFile(a.docx) as z:
            tree=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        text='\n'.join(''.join(p.itertext()) if not p.findall('.//w:t',ns) else ''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in tree.findall('.//w:p',ns))
    chapters={int(n):normalize(t) for n,t in re.findall(r'^[ \t]*Chapter (\d+)\s*[—–-]\s*(.+)$',text,re.M)}
    sections={n:normalize(t) for n,t in re.findall(r'^[ \t]*(\d+(?:\.\d+)+|[A-F]\.\d+)\s+(.+)$',text,re.M)}
    catalog=json.loads(Path(a.catalog).read_text())
    expected_chapters={c['chapter']:normalize(c['title']) for c in catalog['capabilities']}
    expected_sections={s['id']:normalize(s['title']) for c in catalog['capabilities'] for s in c['sections']}
    expected_sections.update({s['id']:normalize(s['title']) for s in catalog['appendices']})
    mismatches=dict(chapters=[n for n in set(chapters)|set(expected_chapters) if chapters.get(n)!=expected_chapters.get(n)],
        sections=[n for n in set(sections)|set(expected_sections) if sections.get(n)!=expected_sections.get(n)])
    passed=not any(mismatches.values())
    print(json.dumps(dict(coverage_matches=passed,chapter_count=len(chapters),toc_entry_count=len(sections),mismatches=mismatches),indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
