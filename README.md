# AestusJal · Dashboard de actividad térmica por municipio de Jalisco

[![Pruebas](https://github.com/diegoegogogo/AestusJal/actions/workflows/tests.yml/badge.svg)](https://github.com/diegoegogogo/AestusJal/actions/workflows/tests.yml)

Tablero interactivo en Streamlit para explorar los resultados de **AestusJal**, un proyecto universitario que
clasifica si un municipio de Jalisco tendrá **actividad térmica elevada** (más de 3 detecciones satelitales
NASA FIRMS VIIRS) durante los 7 días siguientes.

**Demo en vivo:** _(enlace de Streamlit Community Cloud)_

## Qué se puede hacer

| Pestaña | Contenido |
|---|---|
| 🔎 Consulta | Buscar un municipio y una fecha: realidad observada, score de cada modelo frente a su umbral, y si acertó. Serie de tiempo con umbral y días positivos. |
| 📊 Comparación | Métricas de los 4 clasificadores y 2 referencias, PR-AUC frente al azar, curvas ROC y Precision-Recall, matriz de confusión. |
| 🧪 Exploratorio | Variante con municipio como variable, separada y marcada como exploratoria, comparada por periodo. |

## Hallazgos que muestra el tablero

- **Gradient Boosting** obtiene el mejor PR-AUC (0.687) en el test 2025–2026, pero la **climatología municipal-semanal** queda muy cerca (0.679) y el IC95 % de la diferencia incluye el cero.
- Los modelos aprenden sobre todo la **estacionalidad**: alertan durante la temporada seca y fallan al detectar el **inicio** de cada temporada.

## Ejecutar en local

```bash
python -m pip install -r requirements.txt
python -m pytest            # 9 pruebas: reproducen las métricas oficiales del proyecto
python -m streamlit run app.py
```

## Datos y reproducibilidad

`data/` contiene las predicciones del periodo de test (78,875 municipio-día, 125 municipios) y las tablas de métricas.
Se regeneran desde el paquete del proyecto con `python scripts/preparar_datos.py RUTA/AestusJal_PAQUETE_MAESTRO`.
Las métricas se calculan con NumPy (`src/aestus/metricas.py`) y las pruebas verifican que coincidan con las oficiales.

**Fuentes:** NASA FIRMS (VIIRS) para las detecciones térmicas e INEGI para los límites municipales.

## Advertencias metodológicas

- Una detección térmica satelital **no equivale** a un incendio confirmado.
- El periodo 2025–2026 se observó durante el análisis: los resultados son **exploratorios**, no una validación final independiente.
- Los scores no son probabilidades calibradas; cada modelo alerta con su propio umbral elegido en validación 2022–2024.

## Equipo

Proyecto transversal de la Licenciatura en Inteligencia Artificial y Ciencia de Datos, Universidad de Guadalajara.

- Edgar Alfonso Sánchez Gueta
- Christopher Joseph Ontiveros Cedillo
- Diego Antonio Garibay Méndez — autor de este dashboard

Los datos integrados, los modelos y las evaluaciones son trabajo del equipo; este repositorio contiene la capa de visualización.
