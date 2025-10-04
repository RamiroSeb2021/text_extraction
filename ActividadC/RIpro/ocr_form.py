#!/usr/bin/env python3
import re
import numpy as np
import cv2
import pytesseract
from pyimagesearch.alignment.align_images import align_images

# -----------------------------------------------------------------------------
# Configuración de Tesseract
# -----------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------------------------------------------------------
# Función principal
# -----------------------------------------------------------------------------
def main():
    """
    Carga las imágenes, realiza la alineación y extrae texto mediante OCR.

    Pasos:
    -------
    1. Carga la plantilla (`Template.jpg`) y la imagen de entrada (`input.jpg`).
    2. Usa `align_images` para corregir perspectiva y obtener la versión alineada.
    3. Define las regiones de interés (ROI) donde se espera texto.
    4. Aplica preprocesamiento y OCR a cada ROI.
    5. Muestra el texto detectado y dibuja las regiones sobre la imagen.

    Dependencias:
    --------------
    - La función `align_images` del módulo `pyimagesearch.alignment.align_images`.
    - Tesseract debe estar correctamente configurado en el sistema.
    """

    # -------------------------------------------------------------------------
    # 1. Cargar imágenes
    # -------------------------------------------------------------------------
    template = cv2.imread("images/Template.jpg")
    image = cv2.imread("images/input.jpg")

    if template is None or image is None:
        raise FileNotFoundError("No se encontraron las imágenes. Verifica las rutas.")

    # -------------------------------------------------------------------------
    # 2. Alinear la imagen respecto a la plantilla
    # -------------------------------------------------------------------------
    aligned, _, _, _ = align_images(image, template, use_sift=True)

    # -------------------------------------------------------------------------
    # 3. Definir las regiones de interés (ROIs)
    #    Cada campo: (x, y, w, h)
    # -------------------------------------------------------------------------
    fields = {
        "nombre": (490, 255, 360, 105),
        "informacion": (336, 375, 684, 205),
        "autor": (400, 660, 550, 100),
        "id_arbol": (10, 40, 240, 180)
    }

    results = {}

    # -------------------------------------------------------------------------
    # 4. Procesar cada campo y extraer texto
    # -------------------------------------------------------------------------
    for field, (x, y, w, h) in fields.items():
        roi = aligned[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # OCR diferenciado: numérico para ID, texto general para otros
        if field == "id_arbol":
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -c tessedit_char_whitelist=0123456789"
            ).strip()
        else:
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            text = pytesseract.image_to_string(thresh, lang="eng").strip()

        # Guardar resultado
        results[field] = text

        # Dibujar rectángulo y etiqueta
        cv2.rectangle(aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            aligned,
            field,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # -------------------------------------------------------------------------
    # 5. Mostrar resultados
    # -------------------------------------------------------------------------
    for field, text in results.items():
        print(f"[{field.upper()}]: {text}")

    cv2.imshow("Aligned + OCR", aligned)
    cv2.waitKey(0)


# -----------------------------------------------------------------------------
# Ejecución directa
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
