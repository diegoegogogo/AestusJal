"""Genera los archivos de data/ a partir del paquete maestro de AestusJal.

Uso:
    python scripts/preparar_datos.py RUTA/AestusJal_PAQUETE_MAESTRO
"""
import io
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

SALIDA = Path(__file__).resolve().parents[1] / "data"
RENOMBRAR = {
    "score_Regresion_logistica": "score_regresion_logistica",
    "score_Árbol_de_decision": "score_arbol_decision",
    "score_Random_Forest": "score_random_forest",
    "score_Gradient_Boosting": "score_gradient_boosting",
    "score_Media_movil_7d": "score_media_movil_7d",
    "score_Climatologia_historica": "score_climatologia",
}


def main(paquete: Path) -> None:
    res = paquete / "06_RESULTADOS_COMPROBACION"
    with zipfile.ZipFile(res / "Resultados_4_modelos_originales.zip") as z:
        def leer(nombre):
            ruta = next(n for n in z.namelist() if n.endswith(nombre))
            return z.read(ruta)
        pred = pd.read_csv(io.BytesIO(leer("predicciones_test_seleccionadas.csv.gz")), compression="gzip")
        resultados = pd.read_csv(io.BytesIO(leer("resultados_test_todos.csv")), encoding="utf-8-sig")

    pred = pred.rename(columns=RENOMBRAR)
    exploratorio = pd.read_csv(res / "predicciones_test_2025_2026_EXPLORATORIAS.csv.gz")
    pred = pred.merge(exploratorio[["cvegeo", "fecha", "score_gb_municipio"]], on=["cvegeo", "fecha"], how="left", validate="1:1")

    serie = pd.read_csv(paquete / "02_DATOS_TRES_BASES" / "03_Serie_diaria_125_municipios_386375.csv")
    serie = serie[["cvegeo", "fecha", "municipio", "detecciones_dia", "conteo_proximos_7d"]]
    pred = pred.merge(serie, on=["cvegeo", "fecha"], how="left", validate="1:1")

    if pred[["municipio", "score_gb_municipio"]].isna().any().any():
        raise SystemExit("El cruce dejó filas sin municipio o sin score exploratorio.")
    pred = pred.round(6).sort_values(["municipio", "fecha"])
    SALIDA.mkdir(exist_ok=True)
    pred.to_csv(SALIDA / "predicciones_2025_2026.csv", index=False)

    # Umbrales elegidos en validación (conjunto B) y el del modelo exploratorio.
    umbrales = resultados[resultados["Conjunto"] == "B"][["Modelo", "umbral_elegido_en_validacion"]]
    diag = pd.read_csv(res / "diagnostico_2025_2026_EXPLORATORIO.csv", encoding="utf-8-sig")
    u_exp = diag.loc[diag["Modelo"].str.startswith("B + municipio") & diag["Segmento"].str.contains("completo"), "umbral_validacion"].iloc[0]
    umbrales = pd.concat([umbrales, pd.DataFrame({"Modelo": ["GB + municipio"], "umbral_elegido_en_validacion": [u_exp]})])
    umbrales.columns = ["modelo", "umbral"]
    umbrales.to_csv(SALIDA / "umbrales.csv", index=False)

    resultados[resultados["Conjunto"] == "B"].to_csv(SALIDA / "metricas_oficiales_test.csv", index=False)
    shutil.copy(res / "resumen_validacion_2022_2024.csv", SALIDA / "resumen_validacion_2022_2024.csv")
    shutil.copy(res / "diagnostico_2025_2026_EXPLORATORIO.csv", SALIDA / "diagnostico_2025_2026.csv")
    print(f"Listo: {len(pred):,} filas, {pred['municipio'].nunique()} municipios.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
