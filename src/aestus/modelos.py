"""Catálogo de modelos: columna de score, umbral elegido en validación y rol."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Modelo:
    nombre: str
    columna: str
    rol: str  # "clasificador", "referencia" o "exploratorio"


MODELOS = [
    Modelo("Gradient Boosting", "score_gradient_boosting", "clasificador"),
    Modelo("Random Forest", "score_random_forest", "clasificador"),
    Modelo("Regresión logística", "score_regresion_logistica", "clasificador"),
    Modelo("Árbol de decisión", "score_arbol_decision", "clasificador"),
    Modelo("Climatología histórica", "score_climatologia", "referencia"),
    Modelo("Media móvil 7d", "score_media_movil_7d", "referencia"),
    Modelo("GB + municipio", "score_gb_municipio", "exploratorio"),
]
OFICIALES = [m for m in MODELOS if m.rol != "exploratorio"]
POR_NOMBRE = {m.nombre: m for m in MODELOS}
