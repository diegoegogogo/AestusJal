# CLAUDE.md · AestusJal dashboard

Contexto para Claude Code y para el equipo (Edgar, Christopher, Diego). Responde en español.

## Qué es

Dashboard en Streamlit de AestusJal: clasificación binaria de **actividad térmica elevada** por municipio de Jalisco.
Etiqueta = 1 si hay **más de 3 detecciones NASA FIRMS VIIRS en los 7 días siguientes**.
Demo pública: https://aestusjal.streamlit.app/ — **todo lo que llega a `main` se publica solo.**

## Comandos (Windows: anteponer `python -m`)

```bash
python -m pip install -r requirements.txt
python -m pytest                      # 9 pruebas; deben pasar antes de cualquier PR
python -m streamlit run app.py
python scripts/preparar_datos.py RUTA/AestusJal_PAQUETE_MAESTRO   # regenera data/
```

El paquete maestro NO está en el repositorio; se comparte aparte dentro del equipo.

## Estructura

| Ruta | Contenido |
|---|---|
| `app.py` | Interfaz: pestañas Consulta, Comparación y Exploratorio |
| `src/aestus/modelos.py` | Catálogo de modelos: nombre, columna de score y rol (clasificador, referencia, exploratorio) |
| `src/aestus/metricas.py` | Matriz de confusión, ROC, PR y average precision con NumPy puro (sin scikit-learn) |
| `scripts/preparar_datos.py` | Genera `data/` desde el paquete maestro |
| `data/` | Predicciones del test 2025–2026 (78,875 municipio-día, 125 municipios), umbrales y métricas oficiales |
| `tests/test_metricas.py` | Verifica que se reproducen las métricas oficiales del proyecto |

## Reglas metodológicas (no romper)

- **El test 2025–2026 es exploratorio**: ya se observó durante el análisis. No presentarlo como validación final.
- **GB + municipio es exploratorio**: se diseñó después de ver el test. Va solo en su pestaña, con advertencia; nunca como modelo ganador.
- **Los scores no son probabilidades** (modelos descalibrados por pesos). Cada modelo alerta con **su propio umbral elegido en validación 2022–2024** (`data/umbrales.csv`); no ajustar umbrales usando el test.
- **Métrica principal: PR-AUC** (prevalencia ≈ 0.149). La exactitud engaña con este desbalance.
- La ventaja de Gradient Boosting sobre la climatología (+0.0076 PR-AUC) tiene un IC95 % que incluye el cero: no afirmar que es superior.
- Una detección térmica **no equivale** a un incendio confirmado.
- Si un cambio altera métricas, las pruebas deben seguir reproduciendo las cifras oficiales; no "arreglar" pruebas para que pasen.

## Convenciones

- Interfaz y textos en español; meses en español en las gráficas (`EJE_FECHA`).
- Colores fijos por modelo en `COLORES` (paleta Okabe-Ito); un modelo nuevo necesita su color ahí y su entrada en `modelos.py`.
- Gráficas con Altair; `st.altair_chart(..., width="stretch")` requiere Streamlit ≥ 1.50. `np.trapezoid` requiere NumPy ≥ 2.0.
- Antes de afirmar que una gráfica se ve bien, revisarla renderizada, no solo que el código corra.

## Flujo de trabajo

1. Rama propia por cambio (`git checkout -b nombre-del-cambio`). **Nunca subir directo a `main`.**
2. `python -m pytest` en verde.
3. Pull Request: GitHub Actions corre las pruebas; otra persona del equipo revisa.
4. Al fusionar a `main`, Streamlit Cloud redespliega la demo en unos minutos.
