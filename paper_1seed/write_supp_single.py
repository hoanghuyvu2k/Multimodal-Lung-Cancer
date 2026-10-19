import re

# discussion: bỏ vế "repeated-seed analyses"
p = 'sections/discussion.tex'
s = open(p, encoding='utf-8').read()
a = "reported from one pinned environment), and the repeated-seed analyses use the same patients in every partition, so\nthey quantify partition variability, not variability across patients."
assert s.count(a) == 1
s = s.replace(a, "reported from one pinned environment).")
open(p, 'w', encoding='utf-8').write(s)

P = 'supplementary/supplementary.tex'
s = open(P, encoding='utf-8').read()

# bỏ Note S4, S5 (từ \subsection*{...S4...} đến \section{Supplementary Figures})
a = s.index("\\subsection*{Supplementary Note S4")
b = s.index("\\section{Supplementary Figures}")
s = s[:a] + s[b:]

# bỏ bảng S3 một-phân-hoạch (trùng với bảng 21 tổ hợp trong bài chính)
a = s.index("\\begin{table}[H]\n\\centering\n\\footnotesize\n\\caption{Single-partition")
b = s.index("\\end{table}", a) + len("\\end{table}")
s = s[:a].rstrip() + "\n" + s[b:]

# Note S2
a = s.index("BioClinBERT-PCA16 was evaluated in the IHC-A arm only.")
b = s.index("Because the two encoders were given different prompt variants")
s = s[:a] + ("BioClinBERT-PCA16 was evaluated in the IHC-A arm only. In the single 10-fold partition it reached AUC~$=0.770$, below\n"
             "MiniLM-based NLP-PCA16 ($0.784$; $\\Delta\\text{AUC} = -0.014$) and slightly below the no-clinical baseline ($0.772$;\n"
             "$\\Delta\\text{AUC} = -0.003$). Unlike MiniLM-based NLP-PCA16, BioClinBERT therefore did not improve on numeric\n"
             "labs ($0.768$) in this arm (main-text AUC table for the IHC-A arm).\n") + s[b:]

# Note S3: nửa độ rộng bootstrap 20-seed -> DeLong
a = "patient-level\nbootstrap half-width of $\\pm 0.065$ of the 20-seed ensemble AUC reported in the\nmain text."
if a not in s:
    m = re.search(r"patient-level\s+bootstrap half-width of \$\\pm 0\.06\d\$ of the 20-seed ensemble AUC reported in the\s+main text\.", s)
    assert m, 'S3 sentence'
    a = m.group(0)
s = s.replace(a, "DeLong half-width of $\\pm 0.07$ of the single-partition AUC at the primary benchmark reported in the\nmain text.")
open(P, 'w', encoding='utf-8').write(s)

# chú thích S3 và S4
s = open(P, encoding='utf-8').read()
a = s.index("\\caption{Bootstrap distribution of $\\Delta\\text{AUC}$")
b = s.index("\\label{fig:supp_bootstrap}")
s = s[:a] + ("\\caption{Bootstrap distribution of $\\Delta\\text{AUC}$ (NLP raw $-$ numeric\n"
             "    Labs, IHC-G arm) across 5{,}000 paired patient-level bootstrap resamples of the\n"
             "    discovery cohort, using the out-of-fold risk scores of the single 10-fold partition.\n"
             "    The vertical dashed line marks the observed $\\Delta\\text{AUC} = 0.019$; the 95\\% bootstrap\n"
             "    interval is $[-0.012, +0.051]$.}\n  ") + s[b:]
open(P, 'w', encoding='utf-8').write(s)
print([x for x in re.findall(r".*(?:seed|repeated|20-seed|5~seeds|Table~S3|tab:loso|tab:miss).*", s)])
