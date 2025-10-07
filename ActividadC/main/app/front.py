import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys, os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from align_document import align_images_robust
from ocr_form8 import preprocess_image_for_ocr, remove_horizontal_lines, mask_center_band
from preProcessCodeImg import preprocess_variant, ocr_with_conf, detect_white_box
from preProcesTest import read_numeric_code, preprocess_block_for_ocr, ocr_spanish_lines

# --- Configuración de página ---
st.set_page_config(page_title="Text Extraction", layout="wide")
st.title("Text extraction")
st.markdown("### Importe aquí los archivos que se desea alinear")

# --- Carga múltiple de archivos ---
uploaded_files = st.file_uploader(
    "Drag and drop or browse files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

template_path = "plantilla/Plantilla_0.jpeg"

# --- Selección de archivo ---
if uploaded_files:
    filenames = [f.name for f in uploaded_files]
    selected_file = st.selectbox("Seleccione una imagen para procesar:", filenames)

    if selected_file:
        # Buscar archivo seleccionado
        file_bytes = next(f for f in uploaded_files if f.name == selected_file).read()
        np_img = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Mostrar imágenes
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

        # --- Alineación con plantilla ---
        template = cv2.imread(template_path)
        aligned = align_images_robust(image, template)
        col2.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB), caption="Imagen alineada", use_container_width=True)

        st.markdown("### Información extraída")

        try:
            # (Ejemplo simplificado, ajusta con tus funciones OCR)
            bin_block = preprocess_image_for_ocr(aligned)
            text = ocr_spanish_lines(bin_block, psm=6, conf_min=50)
            st.text_area("Texto extraído", text, height=300)
        except Exception as e:
            st.error(f"Error durante el procesamiento: {e}")
