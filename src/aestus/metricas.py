"""Métricas con NumPy puro, para que el dashboard no dependa de scikit-learn."""
import numpy as np


def matriz_confusion(y, score, umbral):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= umbral).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def precision_recall_f1(cm):
    p = cm["TP"] / (cm["TP"] + cm["FP"]) if cm["TP"] + cm["FP"] else 0.0
    r = cm["TP"] / (cm["TP"] + cm["FN"]) if cm["TP"] + cm["FN"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _conteos_por_umbral(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    orden = np.argsort(-score, kind="mergesort")
    s, y = score[orden], y[orden]
    cortes = np.r_[np.where(np.diff(s))[0], y.size - 1]  # último índice de cada umbral distinto
    tps = np.cumsum(y)[cortes]
    fps = (cortes + 1) - tps
    return tps, fps, s[cortes]


def curva_roc(y, score):
    tps, fps, _ = _conteos_por_umbral(y, score)
    tpr = np.r_[0, tps / tps[-1]]
    fpr = np.r_[0, fps / fps[-1]]
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def curva_pr(y, score):
    """Curva precision-recall y average precision (igual que scikit-learn)."""
    tps, fps, _ = _conteos_por_umbral(y, score)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    return recall, precision, ap


def reducir(x, y, n=300):
    """Submuestrea una curva para graficarla rápido sin perder la forma."""
    if len(x) <= n:
        return x, y
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[idx], y[idx]
