"""Tiện ích TabPFN vs LR: cross-val + external. Cap feature cao chiều bằng SelectKBest in-fold."""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

LR_KW = dict(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs")
_TAB = {}


def get_tabpfn():
    if "clf" not in _TAB:
        import torch
        from tabpfn import TabPFNClassifier
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _TAB["clf"] = lambda: TabPFNClassifier(device=dev, ignore_pretraining_limits=True)
        _TAB["dev"] = dev
    return _TAB["clf"]


def _prep(Xtr, Xte, ytr, k):
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
    if k and k < Xtr.shape[1]:
        sel = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
        Xtr, Xte = Xtr[:, sel.get_support()], Xte[:, sel.get_support()]
    sc = RobustScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def cv_auc(X, y, model="lr", k=50, seeds=(42, 7, 123, 2024, 31337), folds=10):
    X = X.values if hasattr(X, "values") else X
    y = np.asarray(y)
    mk = get_tabpfn() if model == "tabpfn" else None
    aucs = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=s)
        sc, lab = [], []
        for tr, te in skf.split(X, y):
            Xtr, Xte = _prep(X[tr], X[te], y[tr], k)
            if model == "tabpfn":
                clf = mk().fit(Xtr, y[tr]); p = clf.predict_proba(Xte)[:, 1]
            else:
                clf = LogisticRegression(**LR_KW).fit(Xtr, y[tr]); p = clf.decision_function(Xte)
            sc.extend(p); lab.extend(y[te])
        aucs.append(roc_auc_score(lab, sc))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def ext_auc(Xtr, ytr, Xte, yte, model="lr", k=50):
    from validation.run_validation import boot_ci
    Xtr2, Xte2 = _prep(Xtr.values if hasattr(Xtr, "values") else Xtr,
                       Xte.values if hasattr(Xte, "values") else Xte, np.asarray(ytr), k)
    if model == "tabpfn":
        clf = get_tabpfn()().fit(Xtr2, ytr); p = clf.predict_proba(Xte2)[:, 1]
    else:
        clf = LogisticRegression(**LR_KW).fit(Xtr2, ytr); p = clf.decision_function(Xte2)
    return float(roc_auc_score(yte, p)), boot_ci(yte, p), int(len(yte))
