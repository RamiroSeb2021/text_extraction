#!/usr/bin/env python3
"""
OCR Form Processing Script (bloque grande limpio + CÓDIGO en crudo)
- Remueve líneas horizontales (subrayado) para mejorar el título.
- Lee el TÍTULO con un sub-ROI superior en modo línea.
"""

import argparse
import re
from collections import namedtuple

import cv2
import imutils
import numpy as np
import pytesseract

#from pyimagesearch.alignment import align_images
from align_images_Robust import align_images_robust as align_images_Robust


# -------------------- Utilidades generales -------------------- #

def cleanup_text(text: str) -> str:
    lines = text.split('\n')
    return '\n'.join([l.strip() for l in lines if l.strip()])


def preprocess_image_for_ocr(image):
    """Preprocesado general (para bloque principal)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.fastNlMeansDenoising(gray, None, h=15,
                                    templateWindowSize=7, searchWindowSize=21)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    # despeckle
    inv = 255 - th
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    clean = np.full_like(th, 255)
    min_area = 35  # si hay ruido, sube a 60–80
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 0
    return clean


def extract_text_from_image(image, config_string="--oem 1 --psm 6"):
    """OCR genérico con preprocesado general (vista global/diagnóstico)."""
    processed_image = preprocess_image_for_ocr(image)
    raw_text = pytesseract.image_to_string(processed_image, config=config_string)
    return cleanup_text(raw_text), processed_image


# -------------------- Limpiezas específicas del BLOQUE -------------------- #

def remove_horizontal_lines(bin_white_bg, min_frac=0.18):
    """
    Quita líneas horizontales largas (como el subrayado del título).
    bin_white_bg: imagen binaria con texto negro (0) sobre blanco (255).
    """
    h, w = bin_white_bg.shape[:2]
    klen = max(15, int(w * min_frac))              # tamaño mínimo de línea
    tmp = 255 - bin_white_bg                       # texto como 255
    horiz = cv2.morphologyEx(tmp, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)),
                             iterations=1)
    tmp = cv2.subtract(tmp, horiz)
    return 255 - tmp


def mask_center_band(img, left_margin=0.08, right_margin=0.08):
    """Deja solo la franja central del ROI (evita tornillos/ruido en bordes)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    x0 = int(w * left_margin)
    x1 = int(w * (1 - right_margin))
    mask[:, x0:x1] = 255
    return cv2.bitwise_and(img, img, mask=mask)


def ocr_block_roi(roi_bgr, visualize=False):
    """
    OCR del bloque (Nombre científico / Lugar / Altura):
    - Preprocesa, elimina subrayados y máscara central.
    """
    proc = preprocess_image_for_ocr(roi_bgr)              # texto negro sobre blanco
    proc = remove_horizontal_lines(proc, min_frac=0.18)   # quita subrayados
    proc = mask_center_band(proc, left_margin=0.08, right_margin=0.08)

    txt = pytesseract.image_to_string(
        proc, config="-l spa --oem 1 --psm 6 -c preserve_interword_spaces=1"
    )
    if visualize:
        cv2.imshow("bloque_preproc", imutils.resize(proc, width=320))
    return cleanup_text(txt)


def ocr_title_from_roi(roi_bgr, visualize=False):
    """
    Extrae el TÍTULO desde una franja superior del ROI en modo línea.
    """
    h, w = roi_bgr.shape[:2]
    top = roi_bgr[: int(h * 0.28), :]               # top ~28% del bloque
    proc = preprocess_image_for_ocr(top)
    proc = remove_horizontal_lines(proc, min_frac=0.12)   # subrayado justo debajo del título
    txt = pytesseract.image_to_string(
        proc, config="-l spa --oem 1 --psm 7 -c preserve_interword_spaces=1"
    )
    txt = cleanup_text(txt)
    if visualize:
        cv2.imshow("titulo_preproc", imutils.resize(proc, width=320))
    # criterio simple para aceptar título
    ok = len(txt) >= 3 and sum(ch.isalpha() for ch in txt) / max(1, len(txt)) >= 0.6
    return txt if ok else ""


# -------------------- Script principal -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image to be OCR'd")
    ap.add_argument("-t", "--template", required=True, help="path to template image")
    ap.add_argument("-a", "--align", type=int, default=1, help="align before OCR")
    ap.add_argument("-c", "--config", default="--oem 1 --psm 6",
                    help='Tesseract config for GLOBAL view (default: "--oem 1 --psm 6")')
    ap.add_argument("-o", "--output", help="path to save global text")
    ap.add_argument("-v", "--visualize", type=int, default=0, help="visualize windows")
    args = vars(ap.parse_args())

    # Cargar imágenes
    print("[INFO] loading template image...")
    template = cv2.imread(args["template"])
    if template is None:
        print(f"[ERROR] No se pudo cargar template: {args['template']}")
        return

    print("[INFO] loading scanned image...")
    scanned = cv2.imread(args["image"])
    if scanned is None:
        print(f"[ERROR] No se pudo cargar imagen: {args['image']}")
        return

    # Alineación
    if args["align"]:
        print("[INFO] aligning images...")
        aligned = align_images_Robust(scanned, template, debug=args["visualize"])
        if aligned is None:
            print("[ERROR] No se pudo alinear. Uso imagen original.")
            aligned = scanned
    else:
        print("[INFO] skipping image alignment...")
        aligned = scanned

    # OCR global (diagnóstico)
    print("[INFO] performing OCR (global)...")
    extracted_text, processed_image = extract_text_from_image(
        aligned, config_string=args["config"]
    )
    # print(extracted_text)  # comenta si no quieres verlo

    # ROIs relativas: (x_pct, y_pct, w_pct, h_pct)
    OCRLocation = namedtuple("OCRLocation", ["id", "rel_bbox", "filter_keywords"])
    rois_rel = [
        OCRLocation("codigo", (0.000, 0.040, 0.15, 0.14), ["codigo", ":"]),  # CÓDIGO (crudo)
        OCRLocation("bloque_principal", (0.15, 0.20, 0.75, 0.60), []),       # Título + 3 líneas
    ]

    h_al, w_al = aligned.shape[:2]

    def rel_to_abs(rel, w, h):
        x_pct, y_pct, w_pct, h_pct = rel
        return (int(x_pct * w), int(y_pct * h), int(w_pct * w), int(h_pct * h))

    vis = aligned.copy()
    results = {}

    for loc in rois_rel:
        x, y, ww, hh = rel_to_abs(loc.rel_bbox, w_al, h_al)
        pad_top, pad_bottom, pad_left, pad_right = (2, 2, 2, 2)
        x0 = max(0, x - pad_left)
        y0 = max(0, y - pad_top)
        x1 = min(w_al, x + ww + pad_right)
        y1 = min(h_al, y + hh + pad_bottom)
        roi = aligned[y0:y1, x0:x1]

        if loc.id == "bloque_principal":
            # Título desde franja superior (modo línea)
            titulo = ocr_title_from_roi(roi, visualize=bool(args["visualize"]))
            # Resto del bloque para las 3 líneas
            txt = ocr_block_roi(roi, visualize=bool(args["visualize"]))

            # Regex robustas por campo
            re_nom   = re.compile(r"nombre\s+cient[ií]fico\s*:?\s*(.*)", re.I)
            re_lugar = re.compile(r"lugar\s+originari[oa]\s*:?\s*(.*)", re.I)
            re_alt   = re.compile(r"altura\s+del\s+ejemplar\s*:?\s*(.*)", re.I)

            nombre_cientifico = ""
            lugar_originario  = ""
            altura_ejemplar   = ""

            for line in txt.split("\n"):
                s = line.strip()
                if not s:
                    continue
                m = re_nom.search(s)
                if m:
                    nombre_cientifico = m.group(1).strip(); continue
                m = re_lugar.search(s)
                if m:
                    lugar_originario = m.group(1).strip(); continue
                m = re_alt.search(s)
                if m:
                    altura_ejemplar = m.group(1).strip(); continue

            results["titulo"] = titulo
            results["nombre_cientifico"] = nombre_cientifico
            results["lugar_originario"] = lugar_originario
            results["altura_ejemplar"] = altura_ejemplar

        else:
            # CÓDIGO: OCR CRUDO (sin preprocesado), upsample + inversión automática
            roi_up = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            txt_code = pytesseract.image_to_string(
                roi_up,
                config="--oem 1 --psm 7 -c tessedit_do_invert=1 -c preserve_interword_spaces=1"
            )
            results["codigo"] = cleanup_text(txt_code)
            if args["visualize"]:
                cv2.imshow("codigo_raw_roi", imutils.resize(roi_up, width=220))

        # Dibujo de ROIs
        if args["visualize"]:
            color = (0, 0, 255) if loc.id == "codigo" else (0, 255, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.putText(vis, loc.id, (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resultados
    print("[INFO] Campos OCR:")
    for k in ("codigo", "titulo", "nombre_cientifico", "lugar_originario", "altura_ejemplar"):
        if k in results:
            print(f"- {k}: {results[k]}")

    # Guardar texto global si se solicita (diagnóstico)
    if args["output"]:
        print(f"[INFO] saving extracted text to {args['output']}...")
        with open(args["output"], 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print("[INFO] text saved successfully!")

    # Ventanas
    if args["visualize"]:
        template_display  = imutils.resize(template,  width=600)
        scanned_display   = imutils.resize(scanned,   width=600)
        aligned_display   = imutils.resize(aligned,   width=600)
        processed_display = imutils.resize(processed_image, width=600)
        vis_display       = imutils.resize(vis,       width=600)
        cv2.imshow("1. Template", template_display)
        cv2.imshow("2. Scanned", scanned_display)
        cv2.imshow("3. Aligned", aligned_display)
        cv2.imshow("4. Processed for OCR (Para OCR)", processed_display)
        cv2.imshow("5. Aligned + ROIs", vis_display)
        print("[INFO] Press 'q' or ESC to close windows...")
        while True:
            k = cv2.waitKey(20) & 0xFF
            if k in (27, ord('q')):
                break
        cv2.destroyAllWindows()

    print("[INFO] OCR process completed successfully!")


if __name__ == "__main__":
    main()
