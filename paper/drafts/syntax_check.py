import re, glob, sys

def check_braces(text, fname):
    depth = 0
    issues = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text):
            i += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth < 0:
                issues.append(f"{fname}: unmatched }} near char {i}")
                depth = 0
        i += 1
    if depth != 0:
        issues.append(f"{fname}: unbalanced braces, ending depth={depth}")
    return issues

def check_environments(text, fname):
    begins = re.findall(r'\\begin\{([a-zA-Z*]+)\}', text)
    ends = re.findall(r'\\end\{([a-zA-Z*]+)\}', text)
    issues = []
    from collections import Counter
    cb, ce = Counter(begins), Counter(ends)
    for env in set(cb) | set(ce):
        if cb[env] != ce[env]:
            issues.append(f"{fname}: env '{env}' begin={cb[env]} end={ce[env]}")
    return issues

all_issues = []
for f in sorted(glob.glob('**/*.tex', recursive=True)):
    text = open(f, encoding='utf-8').read()
    all_issues += check_braces(text, f)
    all_issues += check_environments(text, f)

if all_issues:
    for x in all_issues:
        print("ISSUE:", x)
else:
    print("No brace/environment issues found.")
