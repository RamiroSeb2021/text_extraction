import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import numpy as np
from pyimagesearch.alignment.align_images import align_images
import pytesseract
import tempfile
import os

# Configurar Tesseract (ajusta ruta según tu instalación)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("🧾 OCR Form Alignment App")
st.write("Sube una imagen y se alineará con la plantilla para extraer texto automáticamente.")

# Subir archivos
uploaded_img = st.file_uploader("Imagen de entrada", type=["jpg", "png", "jpeg"])
template_img = st.file_uploader("Plantilla", type=["jpg", "png", "jpeg"])

if uploaded_img and template_img:
    # Guardar en temporales para OpenCV
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, uploaded_img.name)
    tpl_path = os.path.join(temp_dir, template_img.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_img.read())
    with open(tpl_path, "wb") as f:
        f.write(template_img.read())

    image = cv2.imread(img_path)
    template = cv2.imread(tpl_path)

    st.subheader("🔍 Imagen Original")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Procesar
    try:
        aligned, _, _, _ = align_images(image, template, use_sift=True)
        st.subheader("📐 Imagen Alineada")
        st.image(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

        # OCR simple
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        st.subheader("🧠 Texto Detectado:")
        st.text(text)

    except Exception as e:
        st.error(f"Error durante alineación/OCR: {e}")
