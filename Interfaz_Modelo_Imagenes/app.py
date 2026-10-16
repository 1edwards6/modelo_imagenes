from __future__ import annotations

import hashlib
import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from modelo_imagenes import cargar_imagen_grises, procesar_imagen


st.set_page_config(
    page_title="Modelo matemático de procesamiento de imágenes",
    page_icon="🖼️",
    layout="wide",
)

st.title("Modelo matemático de procesamiento de imágenes")
st.caption(
    "Interfaz experimental basada en los notebooks 01_exploracion y 02_modelo. "
    "El análisis matemático se realiza sobre imágenes en escala de grises de 8 bits."
)

archivo = st.file_uploader(
    "Carga una imagen desde tu computador",
    type=["png", "jpg", "jpeg", "jfif", "bmp", "tif", "tiff", "webp"],
    help="La imagen se convertirá a escala de grises de 8 bits, igual que en los notebooks.",
)

if archivo is None:
    st.info("Para comenzar, carga una imagen. Después podrás ajustar los parámetros y pulsar «Procesar imagen».")
    st.markdown(
        "**Incluye:** interpolación lineal o cúbica, exponente k, función de peso "
        "W(r) con dos intervalos, deformación global con λ, histogramas, curva T(r) y métricas."
    )
    st.stop()

contenido = archivo.getvalue()
identificador_imagen = hashlib.sha256(contenido).hexdigest()
try:
    imagen = cargar_imagen_grises(io.BytesIO(contenido))
except Exception as exc:
    st.error(f"No se pudo abrir la imagen: {exc}")
    st.stop()

with st.expander("Información de la imagen", expanded=True):
    col_img, col_info = st.columns([1, 2], gap="large")
    with col_img:
        st.image(imagen, caption="Imagen original (escala de grises)", use_container_width=True)
    with col_info:
        st.write(f"**Archivo:** {archivo.name}")
        st.write(f"**Dimensiones:** {imagen.shape[1]} × {imagen.shape[0]} píxeles")
        st.write(f"**Tipo de matriz:** {imagen.dtype}")
        st.write(f"**Intensidad mínima y máxima:** {int(imagen.min())} y {int(imagen.max())}")

st.divider()
st.subheader("Parámetros del modelo")

with st.form("parametros_modelo"):
    c1, c2, c3 = st.columns(3)
    with c1:
        metodo = st.selectbox(
            "Interpolación del histograma",
            ["Lineal", "Cúbica"],
            index=0,
            help="Se seleccionan puntos uniformes del histograma y se interpola sobre los 256 niveles.",
        )
        numero_puntos = st.slider(
            "Número de puntos de interpolación",
            min_value=2,
            max_value=256,
            value=50,
            step=1,
            help="El notebook utiliza 50 puntos en su bloque de reconstrucción principal.",
        )
    with c2:
        modo = st.selectbox(
            "Transformación que se aplicará",
            [
                "Interpolación base (CDF inversa)",
                "Interpolación + k",
                "Interpolación + W(r)",
                "Interpolación + k + W(r)",
                "Deformación global del eje con λ",
            ],
            index=0,
        )
        k = st.slider(
            "Exponente k",
            min_value=0.10,
            max_value=3.00,
            value=1.55,
            step=0.05,
            help="Valores iniciales del notebook: 1.55 para lineal y 1.25 para cúbica. Ajusta este control manualmente al cambiar de método.",
        )
    with c3:
        lambda_param = st.slider(
            "Parámetro global λ",
            min_value=0.25,
            max_value=5.00,
            value=1.00,
            step=0.05,
            help="λ = 1 deja el eje sin deformación; puedes explorar hasta λ = 5, como en el experimento del notebook.",
        )
        incluir_he = st.checkbox("Comparar con ecualización clásica", value=True)

    with st.expander("Parámetros de W(r): dos intervalos", expanded=("W(r)" in modo)):
        st.caption("Fuera de ambos intervalos, W(r) = 1. Los intervalos deben estar separados.")
        w1, w2, w3, w4, w5, w6 = st.columns(6)
        with w1:
            a1 = st.number_input("a₁", min_value=0, max_value=255, value=0, step=1)
        with w2:
            b1 = st.number_input("b₁", min_value=0, max_value=255, value=150, step=1)
        with w3:
            alpha = st.number_input("α", min_value=0.01, max_value=20.0, value=2.2, step=0.1)
        with w4:
            a2 = st.number_input("a₂", min_value=0, max_value=255, value=190, step=1)
        with w5:
            b2 = st.number_input("b₂", min_value=0, max_value=255, value=255, step=1)
        with w6:
            beta = st.number_input("β", min_value=0.01, max_value=20.0, value=0.5, step=0.1)

    procesar = st.form_submit_button("Procesar imagen", type="primary", use_container_width=True)

if procesar:
    st.session_state.pop("resultado_modelo", None)
    try:
        resultado = procesar_imagen(
            imagen=imagen,
            metodo=metodo,
            numero_puntos=numero_puntos,
            modo=modo,
            k=float(k),
            lambda_param=float(lambda_param),
            a1=int(a1),
            b1=int(b1),
            alpha=float(alpha),
            a2=int(a2),
            b2=int(b2),
            beta=float(beta),
        )
        resultado["identificador_imagen"] = identificador_imagen
        resultado["nombre_archivo"] = archivo.name
        resultado["metodo"] = metodo
        resultado["numero_puntos"] = numero_puntos
        resultado["modo"] = modo
        resultado["k"] = float(k)
        resultado["lambda"] = float(lambda_param)
        resultado["incluir_he"] = incluir_he
        resultado["parametros_W"] = {"a1": int(a1), "b1": int(b1), "alpha": float(alpha), "a2": int(a2), "b2": int(b2), "beta": float(beta)}
        st.session_state["resultado_modelo"] = resultado
        st.success("Procesamiento terminado.")
    except Exception as exc:
        st.error(f"No se pudo procesar la imagen: {exc}")
        st.stop()

resultado = st.session_state.get("resultado_modelo")
if resultado is None or resultado.get("identificador_imagen") != identificador_imagen:
    st.info("Ajusta los parámetros y pulsa «Procesar imagen» para ver la transformación.")
    st.stop()

st.divider()
st.header("Resultados")

c1, c2, c3 = st.columns(3 if resultado["incluir_he"] else 2, gap="large")
with c1:
    st.image(resultado["imagen_original"], caption="Original", use_container_width=True)
with c2:
    st.image(resultado["imagen_resultado"], caption=f"Modelo: {resultado['metodo']} · {resultado['modo']}", use_container_width=True)
if resultado["incluir_he"]:
    with c3:
        st.image(resultado["imagen_he"], caption="Referencia: ecualización clásica", use_container_width=True)

metric_cols = st.columns(4)
metric_names = ["Desviación estándar", "Rango dinámico", "Entropía (bits)", "Media"]
for col, name in zip(metric_cols, metric_names):
    origen = resultado["metricas_original"][name]
    transformada = resultado["metricas_resultado"][name]
    fmt = ".2f" if name in ("Desviación estándar", "Entropía (bits)", "Media") else "d"
    col.metric(
        name,
        format(transformada, fmt),
        delta=f"{transformada - origen:+.2f}" if fmt == ".2f" else f"{transformada - origen:+d}",
        help="El cambio se calcula respecto de la imagen original.",
    )

if resultado["incluir_he"]:
    with st.expander("Métricas de la ecualización clásica"):
        st.dataframe(
            {
                "Métrica": metric_names,
                "Original": [resultado["metricas_original"][n] for n in metric_names],
                "Modelo": [resultado["metricas_resultado"][n] for n in metric_names],
                "Ecualización clásica": [resultado["metricas_he"][n] for n in metric_names],
            },
            hide_index=True,
            use_container_width=True,
        )

pestanas = st.tabs(["Histogramas reales", "Histograma objetivo", "Curva T(r)", "W(r)"])

with pestanas[0]:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(np.arange(256), resultado["hist_original"], label="Original", linewidth=1.5)
    ax.plot(np.arange(256), resultado["hist_resultado_real"], label="Imagen procesada", linewidth=1.5)
    if resultado["incluir_he"]:
        ax.plot(np.arange(256), resultado["hist_he_real"], label="Ecualización clásica", linewidth=1.2, alpha=0.85)
    ax.set(xlabel="Nivel de intensidad", ylabel="Frecuencia de píxeles", xlim=(0, 255), title="Histogramas de las imágenes resultantes")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with pestanas[1]:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(np.arange(256), resultado["hist_original"], label="Histograma original", alpha=0.8)
    ax.plot(np.arange(256), resultado["hist_interpolado"], label=f"Interpolación {resultado['metodo'].lower()}", linewidth=1.7)
    ax.plot(np.arange(256), resultado["hist_objetivo"], label="Histograma objetivo después de parámetros", linewidth=1.7)
    ax.set(xlabel="Nivel de intensidad", ylabel="Frecuencia (normalizada a la misma suma)", xlim=(0, 255), title="Construcción del histograma objetivo")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("El histograma objetivo es el histograma matemático usado para construir T(r); puede diferir del histograma medido en la imagen transformada por la cuantización a niveles enteros.")

with pestanas[2]:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    niveles = np.arange(256)
    ax.plot(niveles, niveles, "--", label="Identidad", alpha=0.7)
    ax.plot(niveles, resultado["T"], label="T(r) del modelo", linewidth=2)
    if resultado["incluir_he"]:
        ax.plot(niveles, resultado["T_he"], label="Ecualización clásica", linewidth=1.5, alpha=0.85)
    ax.set(xlabel="Intensidad original r", ylabel="Intensidad transformada T(r)", xlim=(0, 255), ylim=(0, 255), title="Función de transformación")
    ax.grid(alpha=0.2)
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with pestanas[3]:
    niveles = np.arange(256)
    W = np.ones(256, dtype=float)
    parametros_W = resultado["parametros_W"]
    a1_g, b1_g, alpha_g = parametros_W["a1"], parametros_W["b1"], parametros_W["alpha"]
    a2_g, b2_g, beta_g = parametros_W["a2"], parametros_W["b2"], parametros_W["beta"]
    W[(niveles >= a1_g) & (niveles <= b1_g)] = alpha_g
    W[(niveles >= a2_g) & (niveles <= b2_g)] = beta_g
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(niveles, W, linewidth=2)
    ax.set(xlabel="Nivel de intensidad r", ylabel="W(r)", xlim=(0, 255), title="Función de peso con dos intervalos")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

buffer = io.BytesIO()
Image.fromarray(resultado["imagen_resultado"]).save(buffer, format="PNG")
st.download_button(
    "Descargar imagen procesada (PNG)",
    data=buffer.getvalue(),
    file_name=f"procesada_{resultado['nombre_archivo'].rsplit('.', 1)[0]}.png",
    mime="image/png",
    type="primary",
)

st.caption(
    "Nota metodológica: se conservaron las dos construcciones de transformación que aparecen en el notebook: "
    "CDF inversa para el modelo base y λ global, y 255·CDF del histograma objetivo para k/W/k+W. "
    "Esto evita cambiar silenciosamente el procedimiento matemático existente."
)
