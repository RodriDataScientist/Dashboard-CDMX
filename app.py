# app.py
import base64
import io
from typing import Dict

import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

import dash
from dash import dcc, html

# ---------------------------
# Configs
# ---------------------------
CSV_PATH = "reviews.csv"
EXTERNAL_STYLESHEETS = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
]

N = 12 # top N / bottom N lugares a mostrar en el foco
MIN_TOPIC_MENTIONS = 40  # umbral mín. para mostrar tópico en treemap
PALETTE = px.colors.sequential.Viridis  # paleta única y agradable

# ---------------------------
# Mapeo de tópicos -> descripciones (usé la información que proporcionaste)
# Añade/ajusta entradas según tus datos
# ---------------------------
topic_descriptions: Dict[str, str] = {
    # GENERALES (ejemplos)
    "0": "El arte, la historia y la cultura mexicana: museos y murales (Diego Rivera).",
    "1": "Comprar boletos en línea para atracciones (Torre Latinoamericana, Anahuacalli...).",
    "2": "Mercados vibrantes con artesanías a precios accesibles.",
    "3": "Palacio de Bellas Artes: acústica, conciertos y Ballet Folklórico.",
    "4": "CDMX como destino obligatorio: Antropología, Castillo de Chapultepec, MUNAL.",
    "5": "Acuario Inbursa: experiencia interactiva, pingüinos y tiburones.",
    "6": "Hoteles históricos: servicio, comida y vistas (base para explorar).",
    "7": "Basílica de Guadalupe: peregrinación, arquitectura e historia.",
    "8": "Cineteca Nacional: cine de arte asequible, proyecciones de calidad.",
    "9": "Murales de Diego Rivera en el Palacio Nacional: gratuitos y emblemáticos.",
    "10": "Papalote Museo del Niño: interactivo para niños y adultos.",
    "11": "Castillo de Chapultepec: edificio histórico con vistas impresionantes.",
    "12": "Ciudad Universitaria (UNAM): patrimonio, murales y cultura.",
    "13": "Zoológico de Chapultepec: entrada gratuita, variedad de animales.",
    "14": "Bosque de Chapultepec: parque grande ideal para picnic y paseos.",
    "15": "Mercados de artesanías: comparar precios y caminar para encontrar tesoros.",
    "16": "Torre Latinoamericana: vistas panorámicas imperdibles.",
    "17": "Museo Memoria y Tolerancia: exposición fuerte para concientizar.",
    "18": "Bazar del Sábado en San Ángel: artesanías y ambiente bohemio.",
    "19": "Palacio de Correos: edificio histórico con interior marmóreo.",
    "20": "Comida en museos: buffets y desayunos recomendados.",
    "21": "Paseo Dominical por Reforma: cierre vial para peatones y ciclistas.",
    "22": "Museos tipo Papalote: actividades interactivas familiares.",
    "23": "Six Flags México: parque de diversiones con pase rápido disponible.",
    "24": "Críticas al servicio en Six Flags: personal y plataformas de pago.",
    "25": "Museo de Cera: figuras para fotos y entretenimiento.",
    "26": "Lucha libre en Arena México: show familiar y ambiente animado.",
    "27": "Problemas de organización y comida en Six Flags.",
    "28": "Tarjeta del Metrobús: forma económica de moverse por la ciudad.",
    "29": "Coyoacán: barrio encantador, churros y ambiente seguro.",
    "30": "Atracciones de Six Flags con largas filas en temporada alta.",
    "31": "Six Flags: variedad de juegos y montañas rusas familiares.",
    "32": "Críticas a la comida de los buffets (mala y cara).",
    "33": "Festival del Terror: aglomeraciones y largas filas.",
    "34": "Zócalo: plaza principal con eventos y edificios históricos.",
    "35": "Templo Mayor: vestigio azteca fascinante en el centro.",
    "36": "Museo Memoria y Tolerancia: reflexión sobre atrocidades humanas.",
    "37": "Mercados de artesanías: paciencia y habilidad para negociar.",
    "38": "World Press Photo: exhibición temporal fotoperiodística.",
    "39": "Polanco: zona exclusiva con restaurantes, pero tráfico pesado.",
    "40": "Mural histórico de Diego Rivera salvado tras 1985: traducción cultural.",
    "41": "Problemas de limpieza y control animal en parques y colonias.",
}

# ---------------------------
# Utilidades
# ---------------------------
def short_label(topic_id: str, max_len: int = 70) -> str:
    desc = topic_descriptions.get(topic_id, "")
    if desc:
        return f"Tópico {topic_id}"


def detect_review_column(df: pd.DataFrame):
    # busca columnas que parezcan contener texto de reseñas
    candidates = [c for c in df.columns if c.lower() in ("review", "review_text", "review_lematizada", "review_lemmatized", "texto", "comentario")]
    if candidates:
        return candidates[0]
    # fallback: la columna con mayor proporción de strings largos
    text_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    best = None
    best_score = 0
    for c in text_cols:
        sample = df[c].dropna().astype(str).head(200)
        avg_len = sample.map(len).mean() if not sample.empty else 0
        score = avg_len * (len(sample) / max(1, len(df)))
        if score > best_score:
            best_score = score
            best = c
    return best


# ---------------------------
# Cargar y preparar datos
# ---------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8", low_memory=False)
df_copy = df.copy()

# normalizar nombres de columnas (tu mapa original)
rename_map = {
    "Lugar": "lugar",
    "SentLabel": "sent_label",
    "SentScore": "sent_score",
    "Topic": "topic",
    "Review": "review_text",
    "review": "review_text",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# validaciones básicas
if "lugar" not in df.columns or "sent_label" not in df.columns:
    raise ValueError("El CSV debe contener al menos las columnas 'Lugar' y 'SentLabel' (o sus equivalentes).")

# limpiar 'lugar'
df["lugar"] = (
    df["lugar"]
    .astype(str)
    .str.replace(r"^reseñas_", "", regex=True)
    .str.replace("_", " ")
    .str.strip()
)

# normalizar tópico
if "topic" in df.columns:
    df["topic"] = df["topic"].astype(str).str.strip()
    df = df[~df["topic"].isin(["-1", "-1.0", "", "None", "nan"])]
else:
    df["topic"] = pd.NA

# detectar columna de reseñas para la wordcloud
review_col = detect_review_column(df)
if review_col:
    df["review_text"] = df.get("review_text", df.get(review_col))
else:
    df["review_text"] = df.get("review_text", "")

# ---------------------------
# Agregaciones por 'lugar'
# ---------------------------
agg = (
    df.groupby("lugar")
    .agg(
        mentions=("sent_label", "count"),
        pos=("sent_label", lambda x: (x.astype(str) == "POS").sum()),
        neg=("sent_label", lambda x: (x.astype(str) == "NEG").sum()),
        neu=("sent_label", lambda x: (x.astype(str) == "NEU").sum()),
    )
    .reset_index()
)
agg["pos_ratio"] = agg["pos"] / agg["mentions"]
agg = agg.sort_values(["pos_ratio", "mentions"], ascending=[False, False])

# top N y bottom N (foco)
top = agg.head(N)
resto = agg[~agg["lugar"].isin(top["lugar"])]
bottom = resto.sort_values("pos_ratio").head(N)
focus = pd.concat([top, bottom], ignore_index=True)

# ---------------------------
# Treemap de tópicos (solo dentro del foco)
# ---------------------------
if df["topic"].notna().any():
    tmp = df[df["lugar"].isin(focus["lugar"])]
    treemap_df = tmp.groupby("topic").size().reset_index(name="mentions")
    treemap_df = treemap_df[treemap_df["mentions"] >= MIN_TOPIC_MENTIONS].reset_index(drop=True)
    # enriquecer con descripciones y etiquetas
    def map_topic_row(row):
        tid = str(row["topic"]).strip()
        desc = topic_descriptions.get(tid, topic_descriptions.get(f"neg_{tid}", ""))
        label = short_label(tid)
        return pd.Series({"topic_id": tid, "label": label, "description": desc, "mentions": int(row["mentions"])})
    if not treemap_df.empty:
        treemap_df = treemap_df.apply(map_topic_row, axis=1)
else:
    treemap_df = pd.DataFrame(columns=["topic_id", "label", "description", "mentions"])

# ---------------------------
# Wordcloud (base64)
# ---------------------------
def make_wordcloud_base64(text_series: pd.Series, colormap="viridis"):
    text = " ".join(text_series.dropna().astype(str))
    if not text.strip() and df["topic"].notna().any():
        text = " ".join(df["topic"].dropna().astype(str))
    if not text.strip():
        return None
    wc = WordCloud(width=1200, height=600, background_color="white", colormap=colormap).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

wordcloud_src = make_wordcloud_base64(df["Review_Lematizada"].fillna("").astype(str), colormap="viridis")

# ---------------------------
# Figuras: barras de sentimiento (stacked) y treemap
# ---------------------------
# Preparar datos para gráfico de sentimiento
# Preparar datos para gráfico de sentimiento divergente
s = focus.copy()
s["pos_pct"] = s["pos"] / s["mentions"] * 100
s["neg_pct"] = s["neg"] / s["mentions"] * 100
s["neu_pct"] = s["neu"] / s["mentions"] * 100

# Marca bottom como negativos para graficar hacia la izquierda
s.loc[s["lugar"].isin(bottom["lugar"]), ["pos_pct", "neg_pct", "neu_pct"]] *= -1

sent_long = s.melt(
    id_vars=["lugar"],
    value_vars=["pos_pct", "neu_pct", "neg_pct"],
    var_name="sentiment",
    value_name="pct"
).replace({"pos_pct": "Positivas", "neg_pct": "Negativas", "neu_pct": "Neutrales"})

# Asegurarnos de que PALETTE es una lista de colores (ej. px.colors.sequential.Viridis)
PALETTE = px.colors.sequential.Viridis  # deja como está si ya la tenías

# Seleccionar 3 tonos distribuidos en la paleta; si la paleta es muy corta, usar fallback
n_colors = len(PALETTE) if PALETTE is not None else 0
if n_colors >= 3:
    # tomamos índices aproximadamente en 80%, 50%, 20% de la paleta (Pos, Neu, Neg)
    idxs = [int(round((n_colors - 1) * p)) for p in (0.8, 0.5, 0.2)]
    palette_for_sent = [PALETTE[i] for i in idxs]
else:
    # fallback: tonos intuitivos (verde, gris, rojo)
    palette_for_sent = ["#2ca02c", "#7f7f7f", "#d62728"]

# Mapa por nombre de sentimiento (asegúrate de que coincida con los nombres en tus datos)
color_map_sent = {"Positivas": palette_for_sent[0], "Neutrales": palette_for_sent[1], "Negativas": palette_for_sent[2]}

# ==========================
# Sentimiento Top N (mejores)
# ==========================
sent_top = top.melt(
    id_vars=["lugar"],
    value_vars=["pos", "neu", "neg"],
    var_name="sentiment",
    value_name="count"
).replace({"pos": "Positivas", "neu": "Neutrales", "neg": "Negativas"})

fig_sentiment_top = px.bar(
    sent_top,
    x="count",
    y="lugar",
    color="sentiment",
    orientation="h",
    title=f"{N} Lugares Mejor Evaluados",
    color_discrete_map=color_map_sent,
    barmode="stack",
    text="count"
)
fig_sentiment_top.update_traces(texttemplate="%{text}", textposition="inside")
fig_sentiment_top.update_layout(yaxis_title="", xaxis_title="Menciones")


# ==========================
# Sentimiento Bottom N (peores)
# ==========================
sent_bottom = bottom.melt(
    id_vars=["lugar"],
    value_vars=["pos", "neu", "neg"],
    var_name="sentiment",
    value_name="count"
).replace({"pos": "Positivas", "neu": "Neutrales", "neg": "Negativas"})

fig_sentiment_bottom = px.bar(
    sent_bottom,
    x="count",
    y="lugar",
    color="sentiment",
    orientation="h",
    title=f"{N} Lugares Peor Evaluados",
    color_discrete_map=color_map_sent,
    barmode="stack",
    text="count"
)
fig_sentiment_bottom.update_traces(texttemplate="%{text}", textposition="inside")
fig_sentiment_bottom.update_layout(yaxis_title="", xaxis_title="Menciones")


fig_sentiment_top.update_layout(
    title_font_size=20,       # tamaño del título
    xaxis_title_font_size=18, # eje X
    yaxis_title_font_size=18, # eje Y
    font=dict(size=14)        # tamaño general del texto
)

fig_sentiment_bottom.update_layout(
    title_font_size=20,
    xaxis_title_font_size=18,
    yaxis_title_font_size=18,
    font=dict(size=14)
)





# Treemap
if not treemap_df.empty:
    fig_treemap = px.treemap(
        treemap_df,
        path=[px.Constant(""), "label"],
        values="mentions",
        hover_data=["description"],
        title="Tópicos mencionados en los lugares de interés (foco)",
        template="plotly_white"
    )

    # mejorar hover y estilo
    fig_treemap.data[0].customdata = treemap_df[["description"]].values
    fig_treemap.update_traces(
        textinfo="label+value",
        textfont=dict(size=14),
        customdata=treemap_df[["description"]],
        hovertemplate="<b>%{label}</b><br>Menciones: %{value}<br>%{customdata[0]}<extra></extra>",
        root_color="lightgrey"
    )
    fig_treemap.update_layout(
    title_font_size=20,
    font=dict(size=14)
)
    fig_treemap.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=500)
else:
    # placeholder si no hay tópicos que cumplan el umbral
    fig_treemap = px.imshow([[0]], text_auto=False, template="plotly_white")
    fig_treemap.update_layout(
        title="No hay suficientes menciones por tópico dentro del foco para mostrar un treemap.",
        xaxis_showgrid=False, yaxis_showgrid=False, xaxis_showticklabels=False, yaxis_showticklabels=False,
        height=200, margin=dict(t=60, b=20, l=20, r=20)
    )

SUMMARY_MD = """
📌 **Resumen general de reseñas en la CDMX**

La Ciudad de México es un destino cultural y turístico imperdible, con gran reconocimiento en sus museos, arquitectura y tradiciones.

**Entre los sitios mejor evaluados destacan:**

- **Cultura y Patrimonio:** Museo de Antropología, Castillo de Chapultepec, Palacio de Bellas Artes (incluido su ballet), Murales de Diego Rivera (Palacio Nacional), MUNAL, Memoria y Tolerancia, Zócalo, Templo Mayor y Basílica de Guadalupe.
- **Experiencias Familiares y Modernas:** Papalote Museo del Niño, Zoológico y Bosque de Chapultepec, y el Acuario Inbursa (elogiado por especies y diseño, pero criticado por precios y aglomeraciones).
- **Ambientes y Compras:** La Torre Latino, el Bazar del Sábado en San Ángel, Coyoacán, Polanco, y el moderno centro comercial Antara Fashion Hall (agradable, pero con quejas recurrentes sobre el estacionamiento).
- **Cine:** Cineteca Nacional.

---

### Lo mejor valorado

- **Riqueza cultural y patrimonial:** museos, murales, arquitectura histórica y sitios religiosos.
- **Opciones para familias y niños:** zoológicos, Papalote, acuario, parques y espectáculos.
- **Experiencias urbanas seguras y agradables:** caminar por Reforma en domingo, barrios como Coyoacán o San Ángel.
- **Servicio en hoteles históricos y restaurantes de museos:** buena calidad y precios justos.

### Principales problemas reportados

- **Servicio y Valor:** Quejas por **Six Flags México** (mal servicio, largas filas), buffets y hoteles con mala atención o **precios excesivos** (incluyendo el Acuario Inbursa, souvenirs y áreas VIP).
- **Logística y Organización:** Fallos en boleteras en línea, **aglomeraciones**, **estacionamientos caóticos** (como en Antara Fashion Hall) y deficiente organización en eventos.
- **Espacios descuidados o inseguros:** Zoológico de Chapultepec mal mantenido, animales en situaciones precarias, basura en parques y falta de limpieza en colonias populares".

En conjunto, las reseñas resaltan a la CDMX como un destino cultural, familiar y diverso, donde las experiencias positivas superan ampliamente a las negativas, aunque persisten retos en servicios, organización y cuidado ambiental.
"""

# ---------------------------
# Dash App layout (mejoras en texto explicativo)
# ---------------------------
app = dash.Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS)
app.title = "Dashboard CDMX - Reseñas"

total_reviews = len(df_copy)
avg_pos_pct = agg["pos_ratio"].mean() * 100

app.layout = html.Div([
    html.Div([
        html.H1("Análisis de reseñas — Atracciones de la CDMX", className="h3"),
        html.H2("01/08/2010 - 01/09/2025", className="h4"),
    ], className="text-center my-3"),

    html.Div([
        html.Div([
            html.H6("Total de reseñas analizadas", className="text-muted"),
            html.H3(f"{total_reviews:,}"),
        ], className="card p-3 m-2 text-center", style={"width": "260px"}),

        html.Div([
            html.H6("Promedio % positivas (por lugar)", className="text-muted"),
            html.H3(f"{avg_pos_pct:.1f}%")
        ], className="card p-3 m-2 text-center", style={"width": "260px"}),

        html.Div([
            html.H6("Lugares en foco (Top / Bottom)", className="text-muted"),
            html.H3(f"{len(focus):,}")
        ], className="card p-3 m-2 text-center", style={"width": "260px"}),
    ], className="d-flex justify-content-center flex-wrap"),

    html.Hr(),

    html.Div([
        html.H4("Distribución de Sentimientos", className="text-center text-muted"),
        html.Div([
            html.Div([dcc.Graph(figure=fig_sentiment_top)], className="col-md-6 p-2"),
            html.Div([dcc.Graph(figure=fig_sentiment_bottom)], className="col-md-6 p-2"),
        ], className="row")
    ]),

    html.Hr(),

    html.Div([
        html.Div([dcc.Graph(figure=fig_treemap)], className="col-md-6 p-2"),
        html.Div([
            html.Div([
                dcc.Markdown(SUMMARY_MD, dangerously_allow_html=True),
            ], className="card p-3", style={"maxHeight": "520px", "overflow": "auto", "textAlign": "left"})
        ], className="col-md-6 p-2"),
    ], className="row"),
    html.Footer(html.Div("Dashboard generado con datos de reseñas en TripAdvisor — CDMX", className="text-center mt-2 mb-4 text-muted"))
], className="container-fluid")

if __name__ == "__main__":
    app.run(debug=True, port=8050)

