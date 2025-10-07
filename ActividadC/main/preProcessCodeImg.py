import cv2, numpy as np, pytesseract, os

# Imports opcionales
try:
    import pandas as pd
except ImportError:
    pd = None

# CHUNK 2 — Utilidades: HSV crop, CLAHE, umbrales, OCR con confianza

def crop_green_hsv(img, lower=(35,25,25), upper=(85,255,255), ksize=5, it_close=2, it_open=1, pad_frac=0.04):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=it_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=it_open)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img.copy()
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = int(max(w,h)*pad_frac)
    H,W = img.shape[:2]
    x0,y0 = max(0,x-pad), max(0,y-pad)
    x1,y1 = min(W,x+w+pad), min(H,y+h+pad)
    return img[y0:y1, x0:x1]

def apply_clahe(gray, clip=3.0, tiles=8):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles,tiles))
    return clahe.apply(gray)

def thresh_otsu(gray, invert=False):
    ttype = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, th = cv2.threshold(gray, 0, 255, ttype + cv2.THRESH_OTSU)
    return th

def thresh_adapt(gray, block=31, C=5, invert=False):
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, C)
    return 255 - th if invert else th

def rm_small_blobs(bin_img, min_area=40):
    inv = 255 - bin_img  # tinta como blanco para etiquetar
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep = np.zeros_like(bin_img)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels==i] = 255
    return keep

def ocr_with_conf(img_bin, psm=7, whitelist="0123456789"):
    cfg = f'--oem 1 --psm {psm} -c tessedit_char_whitelist={whitelist}'
    data = pytesseract.image_to_data(img_bin, config=cfg, output_type=pytesseract.Output.DICT)
    text = "".join([w for w in data["text"] if w.strip()]) or ""
    confs = [c for c in data["conf"] if isinstance(c,(int,float)) and c>=0]
    mean_conf = float(np.mean(confs)) if confs else -1.0
    return text.strip(), mean_conf

def preprocess_variant(img, variant):
    # upscale ligero siempre ayuda
    img2 = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    if variant["crop_green"]:
        img2 = crop_green_hsv(img2)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if variant["clahe"]:
        gray = apply_clahe(gray, clip=variant.get("clahe_clip",3.0), tiles=variant.get("clahe_tiles",8))

    if variant["thr_type"] == "otsu":
        th = thresh_otsu(gray, invert=variant["invert"])
    elif variant["thr_type"] == "adapt":
        th = thresh_adapt(gray, block=variant.get("block",31), C=variant.get("C",5), invert=variant["invert"])
    else:
        raise ValueError("thr_type desconocido")

    if variant["rm_blobs"]:
        th = rm_small_blobs(th, min_area=variant.get("min_area",40))

    return th

import cv2
import numpy as np

def detect_white_box(roi_bgr, margin=4, min_area_frac=0.02,
                     min_extent=0.60, min_solidity=0.90,
                     ar_min=0.5, ar_max=1.8, debug=False):
    """
    Detecta el rectángulo blanco principal dentro de un ROI y lo recorta.
    Retorna: crop, (x0,y0,x1,y1), mask_bin (misma forma que ROI en gris)
    - margin: píxeles extra alrededor del bbox
    - min_area_frac: área mínima del contorno relativo al ROI
    - min_extent: área_contorno / área_bbox  (qué tan lleno está el rectángulo)
    - min_solidity: área_contorno / área_casco_convexo (qué tan sólido)
    - ar_min, ar_max: rango de aspect ratio (w/h) aceptable
    """
    # 1) gris + Otsu para que fondo sea blanco (255)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr.copy()
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2) limpieza morfológica: quitar punticos y cerrar huecos
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)), iterations=1)

    # 3) contornos externos sobre zonas blancas
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = th.shape
    area_min_abs = min_area_frac * (H * W)

    best_score, best_bbox, best_mask = -1, None, None

    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min_abs:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        ar = w / float(h)  # aspect ratio
        if not (ar_min <= ar <= ar_max):
            continue

        bbox_area = w * h
        extent = area / float(bbox_area + 1e-6)  # cuán lleno está el bbox

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area

        # puntuación simple: prioriza área grande y buena rectangularidad
        score = (area / (H*W)) + 0.5*extent + 0.5*solidity

        if extent >= min_extent and solidity >= min_solidity and score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    # 4) si no se encontró nada decente: fallback al ROI completo
    if best_bbox is None:
        x0, y0, x1, y1 = 0, 0, W, H
        crop = roi_bgr.copy()
        mask_bin = th
    else:
        x, y, w, h = best_bbox
        x0 = max(0, x - margin); y0 = max(0, y - margin)
        x1 = min(W, x + w + margin); y1 = min(H, y + h + margin)
        crop = roi_bgr[y0:y1, x0:x1]

        mask_bin = np.zeros_like(th)
        mask_bin[y0:y1, x0:x1] = 255  # máscara del recorte (para depurar)

    if debug:
        vis = roi_bgr.copy()
        if roi_bgr.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (x0,y0), (x1,y1), (0,0,255), 2)
        cv2.imshow("ROI", roi_bgr)
        cv2.imshow("Binario limpio", th)
        cv2.imshow("Detección rectángulo (bbox)", vis)
        cv2.imshow("Recorte", crop)
        cv2.waitKey(0); cv2.destroyAllWindows()

    return crop, (x0, y0, x1, y1), mask_bin
