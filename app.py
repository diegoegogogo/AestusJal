"""AestusJal: consulta por municipio y fecha, y comparación de modelos.
Ejecutar con: python -m streamlit run app.py
"""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from aestus.metricas import curva_pr, curva_roc, matriz_confusion, precision_recall_f1, reducir  # noqa: E402
from aestus.modelos import OFICIALES, POR_NOMBRE  # noqa: E402

DATA = Path(__file__).parent / "data"
Y = "actividad_elevada_siguiente_7d"
DIF_GB_CLIM_IC95 = (-0.0107, 0.0262)  # bootstrap por bloques de 14 días, paquete AestusJal

st.set_page_config(page_title="AestusJal", layout="wide")


@st.cache_data
def cargar():
    pred = pd.read_csv(DATA / "predicciones_2025_2026.csv", parse_dates=["fecha"])
    umbrales = pd.read_csv(DATA / "umbrales.csv").set_index("modelo")["umbral"].to_dict()
    oficiales = pd.read_csv(DATA / "metricas_oficiales_test.csv")
    validacion = pd.read_csv(DATA / "resumen_validacion_2022_2024.csv", encoding="utf-8-sig")
    diagnostico = pd.read_csv(DATA / "diagnostico_2025_2026.csv", encoding="utf-8-sig")
    return pred, umbrales, oficiales, validacion, diagnostico


@st.cache_data
def curvas(nombres: tuple[str, ...]):
    pred, *_ = cargar()
    roc, pr = [], []
    for n in nombres:
        s = pred[POR_NOMBRE[n].columna]
        fpr, tpr, auc = curva_roc(pred[Y], s)
        rec, prec, ap = curva_pr(pred[Y], s)
        x, y = reducir(fpr, tpr)
        roc.append(pd.DataFrame({"FPR": x, "TPR": y, "Modelo": n, "Etiqueta": f"{n} · ROC-AUC {auc:.3f}"}))
        x, y = reducir(rec, prec)
        pr.append(pd.DataFrame({"Recall": x, "Precision": y, "Modelo": n, "Etiqueta": f"{n} · PR-AUC {ap:.3f}"}))
    return pd.concat(roc), pd.concat(pr)


MESES = "['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']"
EJE_FECHA = alt.Axis(
    title="Fecha", format="%m",
    labelExpr=f"{MESES}[month(datum.value)] + (month(datum.value) == 0 ? ' ' + year(datum.value) : '')",
)


# Paleta Okabe-Ito (apta para daltonismo); un color fijo por modelo en todas las gráficas.
COLORES = {
    "Gradient Boosting": "#0072B2", "Random Forest": "#009E73", "Regresión logística": "#E69F00",
    "Árbol de decisión": "#CC79A7", "Climatología histórica": "#7F7F7F", "Media móvil 7d": "#B8B8B8",
    "GB + municipio": "#D55E00",
}
NOMBRES_METRICAS = {
    "PR_AUC": "PR-AUC", "F1": "F1", "Precision": "Precisión", "Recall": "Recall (exhaustividad)",
    "Accuracy": "Exactitud", "ROC_AUC": "ROC-AUC", "Balanced_accuracy": "Exactitud balanceada",
}


def escala_modelos(nombres):
    return alt.Scale(domain=list(nombres), range=[COLORES[n] for n in nombres])


def grafica_curvas(df, x, y, nombres, titulo_x, titulo_y, diagonal=False):
    leyenda = alt.Legend(orient="bottom", columns=1, labelLimit=0, title=None)
    linea = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X(f"{x}:Q", title=titulo_x, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(f"{y}:Q", title=titulo_y, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Etiqueta:N", legend=leyenda, scale=alt.Scale(
            domain=[df.loc[df["Modelo"] == n, "Etiqueta"].iloc[0] for n in nombres],
            range=[COLORES[n] for n in nombres])),
        strokeDash=alt.condition(alt.datum.Modelo == "Climatología histórica", alt.value([6, 3]), alt.value([1, 0])),
        tooltip=["Etiqueta:N"],
    )
    if diagonal:
        d = alt.Chart(pd.DataFrame({x: [0, 1], y: [0, 1]})).mark_line(strokeDash=[3, 3], color="gray").encode(x=f"{x}:Q", y=f"{y}:Q")
        linea = linea + d
    return linea.properties(height=380)


def resultado(real: int, alerta: int) -> str:
    return {(1, 1): "✔ Acierto (alertó)", (0, 0): "✔ Acierto (sin alerta)",
            (0, 1): "✘ Falsa alarma", (1, 0): "✘ No detectó"}[(real, alerta)]


pred, UMBRAL, oficiales, validacion, diagnostico = cargar()
prevalencia = pred[Y].mean()

TODOS = f"★ Todos los municipios ({pred['municipio'].nunique()})"


def consulta_municipio(municipio, c2, c3):
    sub = pred[pred["municipio"] == municipio].set_index("fecha").sort_index()
    positivos = sub.index[sub[Y] == 1]
    inicial = (positivos.min() if len(positivos) else sub.index.min()).date()
    fecha = c2.date_input("Fecha", value=inicial, key=f"fecha_{municipio}",
                          min_value=sub.index.min().date(), max_value=sub.index.max().date(),
                          help="Abre en el primer día con actividad elevada real del municipio.")
    modelo_graf = c3.selectbox("Modelo para la gráfica", [m.nombre for m in OFICIALES])

    ts = pd.Timestamp(fecha)
    if ts not in sub.index:
        st.warning("No hay registro para esa fecha en este municipio.")
    else:
        fila = sub.loc[ts]
        real = int(fila[Y])
        filas = []
        for m in OFICIALES:
            alerta = int(fila[m.columna] >= UMBRAL[m.nombre])
            filas.append({"Modelo": m.nombre, "Tipo": m.rol,
                          "Score": round(float(fila[m.columna]), 3),
                          "Umbral de validación": round(UMBRAL[m.nombre], 3),
                          "¿Alertó?": "Sí" if alerta else "No",
                          "Resultado": resultado(real, alerta)})
        tabla = pd.DataFrame(filas)
        aciertos = tabla["Resultado"].str.startswith("✔").sum()

        st.subheader(f"{municipio} · {fecha:%d/%m/%Y}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Realidad: actividad elevada en 7 días", "Sí" if real else "No")
        k2.metric("Detecciones ese día", int(fila["detecciones_dia"]))
        k3.metric("Detecciones en los 7 días siguientes", int(fila["conteo_proximos_7d"]))
        k4.metric("Modelos que acertaron", f"{aciertos} de {len(tabla)}")
        st.dataframe(tabla, hide_index=True, width="stretch", column_config={
            "Score": st.column_config.NumberColumn(format="%.3f"),
            "Umbral de validación": st.column_config.NumberColumn(format="%.3f"),
        })
        st.caption(
            "El score **no es una probabilidad**: los modelos se entrenaron con pesos y quedaron descalibrados "
            f"(score medio de Gradient Boosting ≈ {pred['score_gradient_boosting'].mean():.2f} frente a "
            f"una prevalencia real de {prevalencia:.3f}). Cada modelo alerta cuando su score supera su propio "
            "umbral, elegido en validación 2022–2024. La media móvil usa conteos, no un score de 0 a 1."
        )

    st.subheader(f"{modelo_graf} en {municipio}: score, umbral y realidad")
    col = POR_NOMBRE[modelo_graf].columna
    g = sub.reset_index()[["fecha", col, Y]].rename(columns={col: "score"})
    g["Realidad"] = g[Y].map({1: "Actividad elevada", 0: "Sin actividad elevada"})
    base = alt.Chart(g).encode(x=alt.X("fecha:T", axis=EJE_FECHA))
    linea = base.mark_line(color="#4C78A8").encode(y=alt.Y("score:Q", title="Score del modelo"),
                                                   tooltip=["fecha:T", alt.Tooltip("score:Q", format=".3f"), "Realidad"])
    puntos = base.transform_filter(alt.datum[Y] == 1).mark_tick(color="#E45756", thickness=2, size=12).encode(
        y=alt.value(0), tooltip=["fecha:T", "Realidad"])
    regla = alt.Chart(pd.DataFrame({"u": [UMBRAL[modelo_graf]]})).mark_rule(strokeDash=[6, 4], color="#F58518").encode(y="u:Q")
    st.altair_chart((linea + regla + puntos).properties(height=320), width="stretch")
    st.caption("Línea naranja punteada: umbral de alerta. Marcas rojas arriba: días cuya ventana de 7 días tuvo actividad elevada real.")

    cm = matriz_confusion(sub[Y], sub[col], UMBRAL[modelo_graf])
    p, r, _ = precision_recall_f1(cm)
    st.markdown(
        f"**En {municipio}, {modelo_graf}:** {cm['TP']} días con alerta correcta, {cm['FP']} días de falsa alarma, "
        f"{cm['FN']} días positivos sin alerta. Precisión {p:.0%}, recall {r:.0%} "
        f"({int(sub[Y].sum())} días positivos de {len(sub)})."
    )

    st.subheader(f"Detecciones térmicas diarias en {municipio}")
    det = sub.reset_index()[["fecha", "detecciones_dia"]]
    st.altair_chart(alt.Chart(det).mark_bar(color="#4C78A8").encode(
        x=alt.X("fecha:T", axis=EJE_FECHA),
        y=alt.Y("detecciones_dia:Q", title="Detecciones por día"),
        tooltip=["fecha:T", alt.Tooltip("detecciones_dia:Q", title="Detecciones")],
    ).properties(height=220), width="stretch")


def consulta_todos(c2, c3):
    """Vista agregada: los 125 municipios a la vez, para una fecha y un modelo."""
    positivos_por_dia = pred.groupby("fecha")[Y].sum()
    fecha = c2.date_input("Fecha", value=positivos_por_dia.idxmax().date(), key="fecha_todos",
                          min_value=pred["fecha"].min().date(), max_value=pred["fecha"].max().date(),
                          help="Abre en el día con más municipios con actividad elevada real.")
    modelo_graf = c3.selectbox("Modelo para la gráfica", [m.nombre for m in OFICIALES])
    col = POR_NOMBRE[modelo_graf].columna
    umbral = UMBRAL[modelo_graf]

    dia = pred[pred["fecha"] == pd.Timestamp(fecha)]
    if dia.empty:
        st.warning("No hay registros para esa fecha.")
    else:
        n = len(dia)
        alertas_modelo = int((dia[col] >= umbral).sum())
        st.subheader(f"Todos los municipios ({n}) · {fecha:%d/%m/%Y}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Municipios con actividad elevada real", f"{int(dia[Y].sum())} de {n}")
        k2.metric("Detecciones ese día (suma)", f"{int(dia['detecciones_dia'].sum()):,}")
        k3.metric("Detecciones en los 7 días siguientes (suma)", f"{int(dia['conteo_proximos_7d'].sum()):,}")
        k4.metric(f"Municipios en alerta ({modelo_graf})", f"{alertas_modelo} de {n}")

        filas = []
        for m in OFICIALES:
            cm = matriz_confusion(dia[Y], dia[m.columna], UMBRAL[m.nombre])
            alertas = cm["TP"] + cm["FP"]
            reales = cm["TP"] + cm["FN"]
            filas.append({
                "Modelo": m.nombre, "Tipo": m.rol, "Municipios en alerta": alertas,
                "Alerta correcta": cm["TP"], "Falsa alarma": cm["FP"], "No detectó": cm["FN"],
                "Sin alerta correcta": cm["TN"],
                "Precisión": cm["TP"] / alertas if alertas else None,
                "Recall": cm["TP"] / reales if reales else None,
                "Aciertos": f"{cm['TP'] + cm['TN']} de {n}",
            })
        st.markdown("**Resultado de cada modelo ese día, sumando todos los municipios**")
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch", column_config={
            "Precisión": st.column_config.NumberColumn(format="percent"),
            "Recall": st.column_config.NumberColumn(format="percent"),
        })
        st.caption("Precisión y recall quedan vacíos cuando no hubo alertas o no hubo casos reales ese día. "
                   "Cada modelo alerta con su propio umbral, elegido en validación 2022–2024.")

        with st.expander(f"Detalle por municipio ({modelo_graf})"):
            det = dia[["municipio", Y, "detecciones_dia", "conteo_proximos_7d", col]].copy()
            det["¿Alertó?"] = (det[col] >= umbral).map({True: "Sí", False: "No"})
            det["Resultado"] = [resultado(int(r), int(s >= umbral)) for r, s in zip(det[Y], det[col])]
            det[Y] = det[Y].map({1: "Sí", 0: "No"})
            det = det.rename(columns={
                "municipio": "Municipio", Y: "Realidad: actividad elevada", "detecciones_dia": "Detecciones ese día",
                "conteo_proximos_7d": "Detecciones 7 días siguientes", col: "Score"})
            det = det.sort_values(["Realidad: actividad elevada", "Score"], ascending=[False, False])
            st.dataframe(det, hide_index=True, width="stretch", column_config={
                "Score": st.column_config.NumberColumn(format="%.3f"),
                "Detecciones ese día": st.column_config.NumberColumn(format="%d"),
                "Detecciones 7 días siguientes": st.column_config.NumberColumn(format="%d"),
            })

    st.subheader(f"{modelo_graf}: municipios en alerta y con actividad real, día por día")
    serie = pd.DataFrame({
        "fecha": pred.groupby("fecha")[Y].sum().index,
        "Municipios con actividad elevada real": pred.groupby("fecha")[Y].sum().to_numpy(),
        "Municipios en alerta del modelo": pred.groupby("fecha")[col].apply(lambda s: int((s >= umbral).sum())).to_numpy(),
    }).melt("fecha", var_name="Serie", value_name="Municipios")
    dominio = ["Municipios con actividad elevada real", "Municipios en alerta del modelo"]
    base = alt.Chart(serie).mark_line(strokeWidth=2).encode(
        x=alt.X("fecha:T", axis=EJE_FECHA),
        y=alt.Y("Municipios:Q", title="Municipios (de 125)"),
        color=alt.Color("Serie:N", scale=alt.Scale(domain=dominio, range=["#E45756", COLORES[modelo_graf]]),
                        legend=alt.Legend(orient="bottom", columns=1, labelLimit=0, title=None)),
        tooltip=["fecha:T", "Serie:N", "Municipios:Q"])
    marca = alt.Chart(pd.DataFrame({"fecha": [pd.Timestamp(fecha)]})).mark_rule(
        strokeDash=[4, 4], color="gray").encode(x="fecha:T")
    st.altair_chart((base + marca).properties(height=320), width="stretch")
    st.caption("Línea gris punteada: la fecha elegida. Si las dos curvas suben y bajan juntas, el modelo sigue bien "
               "la temporada; la distancia entre ambas son falsas alarmas o casos omitidos.")

    cm = matriz_confusion(pred[Y], pred[col], umbral)
    p, r, _ = precision_recall_f1(cm)
    st.markdown(
        f"**En los {pred['municipio'].nunique()} municipios y todo el periodo, {modelo_graf}:** "
        f"{cm['TP']:,} municipio-día con alerta correcta, {cm['FP']:,} de falsa alarma y "
        f"{cm['FN']:,} positivos sin alerta. Precisión {p:.0%}, recall {r:.0%} "
        f"({int(pred[Y].sum()):,} municipio-día positivos de {len(pred):,})."
    )

    st.subheader("Detecciones térmicas diarias en todo Jalisco")
    tot = pred.groupby("fecha", as_index=False)["detecciones_dia"].sum()
    st.altair_chart(alt.Chart(tot).mark_bar(color="#4C78A8").encode(
        x=alt.X("fecha:T", axis=EJE_FECHA),
        y=alt.Y("detecciones_dia:Q", title="Detecciones por día"),
        tooltip=["fecha:T", alt.Tooltip("detecciones_dia:Q", title="Detecciones")],
    ).properties(height=220), width="stretch")



st.title("AestusJal · Actividad térmica elevada por municipio de Jalisco")
st.caption(
    "Clasificación binaria: ¿habrá más de 3 detecciones térmicas NASA FIRMS VIIRS en el municipio "
    "durante los 7 días siguientes? Periodo de test: 2025-01-01 a 2026-09-23."
)
st.info(
    "Una detección térmica **no equivale** a un incendio confirmado. El periodo 2025–2026 ya se observó "
    "durante el análisis, así que todo lo que muestra este tablero es **exploratorio**."
)

tab_consulta, tab_modelos, tab_exp = st.tabs(
    ["🔎 Consulta por municipio y fecha", "📊 Comparación de modelos", "🧪 Variante exploratoria"]
)

# ---------------------------------------------------------------- Consulta
with tab_consulta:
    c1, c2, c3 = st.columns([2, 1, 2])
    municipio = c1.selectbox("Municipio (escribe para buscar)", [TODOS, *sorted(pred["municipio"].unique())], index=1,
                             help="Elige un municipio o la opción de todos a la vez.")
    if municipio == TODOS:
        consulta_todos(c2, c3)
    else:
        consulta_municipio(municipio, c2, c3)


# ---------------------------------------------------------------- Modelos
with tab_modelos:
    st.subheader("Tabla comparativa (test 2025–2026, conjunto B)")
    cols = ["Modelo", *NOMBRES_METRICAS]
    tabla_of = oficiales[cols].sort_values("PR_AUC", ascending=False)
    tabla_of.insert(1, "Tipo", tabla_of["Modelo"].map(lambda n: POR_NOMBRE[n].rol))
    tabla_vista = tabla_of.rename(columns=NOMBRES_METRICAS)
    st.dataframe(tabla_vista.style.format(precision=3).highlight_max(
        subset=list(NOMBRES_METRICAS.values()), props="font-weight: bold; color: #56B4E9"),
        hide_index=True, width="stretch")
    st.caption(
        "En azul, el mejor valor de cada métrica. Métrica principal: **PR-AUC**, porque solo ~15 % de los "
        "municipio-día son positivos. Con ese desbalance, la exactitud engaña: un modelo que nunca alerte "
        f"tendría {1 - prevalencia:.3f}."
    )

    st.subheader("PR-AUC por modelo")
    orden = tabla_of["Modelo"].tolist()
    base_b = alt.Chart(tabla_of).encode(
        y=alt.Y("Modelo:N", sort=orden, title=None, axis=alt.Axis(labelLimit=0)),
        x=alt.X("PR_AUC:Q", title="PR-AUC (test)", scale=alt.Scale(domain=[0, 0.8])),
    )
    barras = base_b.mark_bar().encode(
        color=alt.Color("Modelo:N", scale=escala_modelos(orden), legend=None),
        tooltip=["Modelo", alt.Tooltip("PR_AUC:Q", format=".3f")])
    valores = base_b.mark_text(align="left", dx=6, fontSize=13, color="#9AA0A6").encode(text=alt.Text("PR_AUC:Q", format=".3f"))
    azar = alt.Chart(pd.DataFrame({"x": [prevalencia]})).mark_rule(strokeDash=[4, 4], color="#E45756").encode(x="x:Q")
    st.altair_chart((barras + valores + azar).properties(height=260), width="stretch")
    gb = oficiales.set_index("Modelo").loc["Gradient Boosting", "PR_AUC"]
    clim = oficiales.set_index("Modelo").loc["Climatología histórica", "PR_AUC"]
    st.caption(
        f"Línea roja: PR-AUC de un clasificador al azar (= prevalencia, {prevalencia:.3f}). "
        f"Gradient Boosting supera a la climatología por {gb - clim:+.4f}, pero el IC95 % exploratorio de esa "
        f"diferencia es [{DIF_GB_CLIM_IC95[0]:+.4f}, {DIF_GB_CLIM_IC95[1]:+.4f}] e incluye el cero: "
        "la ventaja no es concluyente."
    )

    elegidos = st.multiselect("Modelos en las curvas", [m.nombre for m in OFICIALES],
                              default=["Gradient Boosting", "Random Forest", "Regresión logística",
                                       "Árbol de decisión", "Climatología histórica"])
    if elegidos:
        roc, pr = curvas(tuple(elegidos))
        a, b = st.columns(2)
        a.markdown("**Curva ROC**")
        a.altair_chart(grafica_curvas(roc, "FPR", "TPR", elegidos, "Tasa de falsos positivos",
                                      "Tasa de verdaderos positivos", diagonal=True), width="stretch")
        b.markdown("**Curva Precision-Recall**")
        b.altair_chart(grafica_curvas(pr, "Recall", "Precision", elegidos, "Recall", "Precisión"), width="stretch")
        st.caption("Que las curvas casi se encimen **es el resultado**: los cuatro clasificadores ordenan los casos "
                   "de forma muy parecida a la climatología (línea gris punteada). Pasa el cursor para ver cada valor.")

    st.subheader("Matriz de confusión")
    mc = st.selectbox("Modelo", [m.nombre for m in OFICIALES], key="mc")
    cm = matriz_confusion(pred[Y], pred[POR_NOMBRE[mc].columna], UMBRAL[mc])
    dfcm = pd.DataFrame([
        {"Real": "Positivo", "Predicho": "Positivo", "n": cm["TP"], "Celda": "Verdaderos positivos"},
        {"Real": "Positivo", "Predicho": "Negativo", "n": cm["FN"], "Celda": "Falsos negativos"},
        {"Real": "Negativo", "Predicho": "Positivo", "n": cm["FP"], "Celda": "Falsos positivos"},
        {"Real": "Negativo", "Predicho": "Negativo", "n": cm["TN"], "Celda": "Verdaderos negativos"},
    ])
    corte = dfcm["n"].max() * 0.5
    calor = alt.Chart(dfcm).encode(
        x=alt.X("Predicho:N", sort=["Positivo", "Negativo"], title="Predicho", axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Real:N", sort=["Positivo", "Negativo"], title="Real"))
    texto = calor.mark_text(fontSize=20, fontWeight="bold").encode(
        text=alt.Text("n:Q", format=","),
        color=alt.condition(alt.datum.n > corte, alt.value("white"), alt.value("black")))
    rect = calor.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None), tooltip=["Celda", alt.Tooltip("n:Q", format=",")])
    m1, m2 = st.columns([1, 1])
    m1.altair_chart((rect + texto).properties(width=380, height=260), width="content")
    p, r, f1 = precision_recall_f1(cm)
    m2.markdown(
        f"**{mc}** · umbral {UMBRAL[mc]:.3f} (elegido en validación)\n\n"
        f"- De cada 100 alertas, **~{p * 100:.0f}** fueron correctas (precisión {p:.3f}).\n"
        f"- Detectó **~{r * 100:.0f} %** de los municipio-día positivos (recall {r:.3f}).\n"
        f"- F1 {f1:.3f} · {cm['FP']:,} falsas alarmas · {cm['FN']:,} positivos sin alerta."
    )

    with st.expander("Validación temporal 2022–2024 (resumen del paquete)"):
        st.dataframe(validacion.style.format(precision=3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Exploratorio
with tab_exp:
    st.warning(
        "**GB + municipio** se diseñó *después* de observar el test 2025–2026. Sus cifras sirven para "
        "diagnóstico y para plantear hipótesis, **no** para declarar un nuevo modelo ganador."
    )
    renombre = {"B + municipio (pesos)": "GB + municipio", "B original (pesos)": "Gradient Boosting",
                "Climatologia historica": "Climatología histórica"}
    dg = diagnostico.assign(Modelo=diagnostico["Modelo"].map(renombre))
    segmentos = ["2025-2026 completo", "2025", "2026 hasta septiembre", "Marzo-junio", "Resto de meses"]
    nombres_exp = ["Gradient Boosting", "GB + municipio", "Climatología histórica"]

    st.subheader("PR-AUC por periodo")
    piv = dg.pivot(index="Segmento", columns="Modelo", values="PR_AUC").loc[segmentos, nombres_exp]
    piv["Ganancia GB + municipio vs GB"] = piv["GB + municipio"] - piv["Gradient Boosting"]
    st.dataframe(piv.style.format("{:.3f}").format("{:+.3f}", subset=["Ganancia GB + municipio vs GB"]), width="stretch")
    st.altair_chart(alt.Chart(dg).mark_bar().encode(
        x=alt.X("Modelo:N", sort=nombres_exp, title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("PR_AUC:Q", title="PR-AUC", scale=alt.Scale(domain=[0, 0.8])),
        color=alt.Color("Modelo:N", scale=escala_modelos(nombres_exp), legend=alt.Legend(orient="bottom", title=None)),
        column=alt.Column("Segmento:N", sort=segmentos, title=None, header=alt.Header(labelOrient="bottom")),
        tooltip=["Modelo", "Segmento", alt.Tooltip("PR_AUC:Q", format=".3f")],
    ).properties(width=120, height=220), width="content")
    st.caption("Ojo: en «Resto de meses» la prevalencia es de solo ~6 %, así que esos PR-AUC son más bajos "
               "por construcción. Compara los modelos dentro de cada periodo, no entre periodos.")

    with st.expander("Todas las métricas del diagnóstico"):
        st.dataframe(dg.rename(columns=NOMBRES_METRICAS).style.format(
            precision=3, subset=dg.rename(columns=NOMBRES_METRICAS).select_dtypes("number").columns.drop("n")),
            hide_index=True, width="stretch")

    _, pr_e = curvas(tuple(nombres_exp))
    st.markdown("**Precision-Recall: original vs variante exploratoria**")
    st.altair_chart(grafica_curvas(pr_e, "Recall", "Precision", nombres_exp, "Recall", "Precisión"), width="stretch")
