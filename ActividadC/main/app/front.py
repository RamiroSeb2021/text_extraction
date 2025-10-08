import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io, os, sys

# --- Importar módulos personalizados ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from align_images_Robust import align_images_robust
from ocr_form8 import preprocess_image_for_ocr
from preProcesTest import ocr_spanish_lines

# --- Configuración de página ---
st.set_page_config(page_title="Text Extraction", layout="wide")

# --- Encabezado ---
st.markdown(
    """
    <h1 style='text-align:center;'>
        Text <span style='color:#ff4b4b;'>extraction</span>
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("### Importe aquí los archivos que desea alinear")

# --- Carga múltiple de archivos ---
uploaded_files = st.file_uploader(
    "📂 Arrastre o seleccione sus archivos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

template_path = "plantilla/Plantilla_0.jpeg"

# --- Si no hay archivos ---
if not uploaded_files:
    st.info("⚠️ Aún no se han cargado imágenes.")
    st.stop()

# --- Selección de archivo ---
filenames = [f.name for f in uploaded_files]
selected_file = st.selectbox("Seleccione una imagen para procesar:", ["-- Seleccione --"] + filenames)

# --- Solo ejecutar si el usuario escoge un archivo real ---
if selected_file != "-- Seleccione --":
    # Botón para iniciar el procesamiento
    if st.button("▶ Procesar imagen seleccionada"):
        st.markdown("### 🧩 Proceso de alineación")

        # Buscar archivo seleccionado
        file_bytes = next(f for f in uploaded_files if f.name == selected_file).read()
        np_img = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        col1.markdown("#### Imagen original")
        col1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # --- Alineación con plantilla ---
        with st.spinner("⏳ Alineando imagen..."):
            template = cv2.imread(template_path)
            aligned = align_images_robust(image, template)

        col2.markdown("#### Imagen alineada")
        col2.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("### 📜 Información extraída")

        # --- OCR ---
        try:
            with st.spinner("🔍 Ejecutando OCR..."):
                bin_block = preprocess_image_for_ocr(aligned)
                text = ocr_spanish_lines(bin_block, psm=6, conf_min=50)
            st.text_area("Texto extraído", text, height=300)
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {e}")

else:
    st.warning("⬆️ Seleccione una imagen de la lista para continuar.")
