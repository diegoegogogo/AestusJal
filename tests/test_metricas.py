from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aestus.metricas import curva_pr, curva_roc, matriz_confusion, precision_recall_f1
from aestus.modelos import OFICIALES, POR_NOMBRE

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def datos():
    pred = pd.read_csv(DATA / "predicciones_2025_2026.csv")
    umbrales = pd.read_csv(DATA / "umbrales.csv").set_index("modelo")["umbral"]
    oficiales = pd.read_csv(DATA / "metricas_oficiales_test.csv").set_index("Modelo")
    return pred, umbrales, oficiales


def test_datos_completos(datos):
    pred, umbrales, _ = datos
    assert len(pred) == 78_875
    assert pred["municipio"].nunique() == 125
    assert not pred.isna().any().any()
    assert set(umbrales.index) == set(POR_NOMBRE)


def test_etiqueta_es_mas_de_3_detecciones(datos):
    pred, _, _ = datos
    esperada = (pred["conteo_proximos_7d"] > 3).astype(int)
    assert (esperada == pred["actividad_elevada_siguiente_7d"]).all()


@pytest.mark.parametrize("modelo", OFICIALES, ids=lambda m: m.nombre)
def test_reproduce_metricas_oficiales(datos, modelo):
    pred, umbrales, oficiales = datos
    y, s = pred["actividad_elevada_siguiente_7d"], pred[modelo.columna]
    of = oficiales.loc[modelo.nombre]
    _, _, ap = curva_pr(y, s)
    _, _, auc = curva_roc(y, s)
    p, r, f1 = precision_recall_f1(matriz_confusion(y, s, umbrales[modelo.nombre]))
    assert ap == pytest.approx(of["PR_AUC"], abs=1e-3)
    assert auc == pytest.approx(of["ROC_AUC"], abs=1e-3)
    assert p == pytest.approx(of["Precision"], abs=1e-3)
    assert r == pytest.approx(of["Recall"], abs=1e-3)
    assert f1 == pytest.approx(of["F1"], abs=1e-3)


def test_curvas_en_caso_pequeno():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    _, _, auc = curva_roc(y, s)
    _, _, ap = curva_pr(y, s)
    assert auc == pytest.approx(0.75)
    assert ap == pytest.approx(0.8333333)  # valor de referencia de scikit-learn
    assert matriz_confusion(y, s, 0.5) == {"TN": 2, "FP": 0, "FN": 1, "TP": 1}
