import re, glob

def wc(text):
    text = re.sub(r'%.*', '', text)
    text = re.sub(r'\\begin\{(table|figure|equation)\}.*?\\end\{\1\}', '', text, flags=re.S)
    text = re.sub(r'\\(label|cite|ref|eqref|input)\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?', '', text)
    text = re.sub(r'[{}\\$]', '', text)
    return len(text.split())

total = 0
for f in sorted(glob.glob('sections/*.tex')):
    t = open(f, encoding='utf-8').read()
    n = wc(t)
    total += n
    print(f"{f}: {n}")
print(f"TOTAL (sections, excl. tables/figures): {total}")

m = open('main.tex', encoding='utf-8').read()
abs_match = re.search(r'\\abstract\{(.*?)\n\}', m, re.S)
if abs_match:
    print("Abstract words:", wc(abs_match.group(1)))
