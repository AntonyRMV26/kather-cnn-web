"""Aplicación pública educativa: clasificador CNN Kather (8 clases)."""
from io import BytesIO
from pathlib import Path
import json
import warnings

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inferencia import Clasificador

BASE = Path(__file__).resolve().parent
MAX_BYTES = 8 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000

st.set_page_config(page_title="Kather CNN | Clasificador histológico", page_icon="🔬", layout="centered")

st.markdown("""<style>
   .block-container { max-width: 820px; padding-top: 2.0rem; padding-bottom: 3.5rem; }
   .stButton>button[kind="primary"] { font-weight: 650; border-radius: .7rem; }
   .kather-head { margin-bottom: 1.2rem; }
   .kather-tag { color: #0f766e; font-size: .83rem; font-weight: 700; letter-spacing: .04em; }
   .kather-title {font-size: 2.0rem; font-weight: 750; line-height: 1.2; margin-top: .35rem; }
   .kather-sub {color: #64748b; font-size: 1rem; margin-top: .4rem; }
   @media (prefers-color-scheme: dark) { .kather-sub {color: #94a3b8;} }
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Cargando modelo de clasificación…")
def obtener_modelo():
    return Clasificador()

def abrir_imagen(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise ValueError("El archivo supera el límite de 8 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(BytesIO(data))
            if im.format not in ("JPEG", "PNG"):
                raise ValueError("Solo se admiten imágenes JPG o PNG.")
            im.load()
            if im.width > 5000 or im.height > 5000:
                raise ValueError("La imagen es demasiado grande (máximo 5000 píxeles por lado).")
            return im.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("No se pudo leer la imagen. Utiliza un JPG o PNG válido.") from exc

st.markdown("""<div class="kather-head">
<div class="kather-tag">UNCP · INGENIERÍA DE CONOCIMIENTOS</div>
<div class="kather-title">🔬 Kather CNN</div>
<div class="kather-sub">Clasificación de fragmentos histológicos colorrectales mediante redes neuronales convolucionales.</div>
</div>""", unsafe_allow_html=True)

st.info("Prototipo educativo, no herramienta diagnóstica. Sube únicamente imágenes histológicas colorrectales similares a las del conjunto Kather, sin datos personales ni clínicos identificables.", icon="ℹ️")

st.subheader("1. Selecciona una imagen")
archivo = st.file_uploader("Imagen histológica (JPG o PNG; máximo 8 MB)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
imagen = None
error_imagen = None
if archivo is not None:
    try:
        imagen = abrir_imagen(archivo.getvalue())
        st.image(imagen, caption=f"Imagen seleccionada · {imagen.width} × {imagen.height} píxeles", use_container_width=True)
        st.caption("La imagen se convierte a RGB y se ajusta a 150 × 150 píxeles antes de predecir. La normalización se realiza dentro de la CNN.")
    except ValueError as e:
        error_imagen = str(e)
        st.error(error_imagen)

clasificar = st.button("🔎 Clasificar imagen", type="primary", use_container_width=True, disabled=imagen is None)

if clasificar and imagen is not None:
    try:
        with st.spinner("Analizando imagen…"):
            resultado = obtener_modelo().predecir(imagen)
    except Exception as exc:
        st.error("No se pudo ejecutar el modelo. Revisa los registros de Streamlit y la instalación de LiteRT.")
        with st.expander("Detalle técnico del error"):
            st.code(f"{type(exc).__name__}: {exc}")
    else:
        st.divider()
        st.subheader("2. Resultado de la CNN")
        st.success(f"Categoría predicha: **{resultado['nombre']}**")
        st.metric("Probabilidad estimada para esta categoría", f"{resultado['probabilidad']*100:.1f}%")
        st.write(resultado["explicacion"])
        st.caption("La probabilidad softmax no representa certeza diagnóstica ni un nivel de fiabilidad clínica.")

        st.markdown("**Categorías con mayor probabilidad**")
        for item in resultado["clases"][:3]:
            st.write(f"{item['nombre']} — {item['probabilidad']*100:.1f}%")
            st.progress(min(100, max(0, round(item['probabilidad']*100))))

        with st.expander("Ver las ocho categorías"):
            for item in resultado["clases"]:
                st.write(f"{item['nombre']} — {item['probabilidad']*100:.2f}%")

        st.warning("Interpretación: se clasificó un recorte según patrones visuales presentes en el dataset. No permite diagnosticar cáncer, determinar pronósticos ni sustituir la evaluación por un patólogo.", icon="⚠️")

st.divider()
with st.expander("Acerca del modelo y sus limitaciones"):
    st.markdown("""
    **Modelo:** CNN Optimizada del trabajo universitario, exportada a TensorFlow Lite (float16).

    **Datos:** Kather-texture-2016, 5,000 recortes en ocho categorías, con distribución balanceada.

    **Evaluación interna del proyecto:** Macro F1 de prueba ≈ **0.9227** (92.27%), con división estratificada por imagen. Este resultado **no** constituye validación independiente por paciente o lámina.

    **Importante:** la CNN produce una clase para cualquier imagen que pueda procesar; una predicción alta no demuestra que la imagen pertenezca al dominio histológico de entrenamiento.

    **Privacidad:** esta demostración no guarda imágenes en una base de datos, pero cualquier archivo enviado a una aplicación pública se procesa en su servidor. No subas información real de pacientes.
    """)
    st.markdown("Fuente: [Dataset Kather en Hugging Face](https://huggingface.co/datasets/1aurent/Kather-texture-2016).")

st.caption("Proyecto académico · Ingeniería de Conocimientos · Universidad Nacional del Centro del Perú")
