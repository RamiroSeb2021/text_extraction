from ocr_form8 import (
cleanup_text,
preprocess_image_for_ocr,
remove_horizontal_lines,
mask_center_band,
ocr_block_roi,
ocr_title_from_roi,
extract_text_from_image
)
from align_images_Robust import align_images_robust as align_images_robust
import cv2
from collections import namedtuple
from preProcessCodeImg import (preprocess_variant, 
                               ocr_with_conf,
                               detect_white_box)
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print(pytesseract.get_tesseract_version())
import numpy as np

# incluye letras españolas + dígitos + puntuación básica
WHITELIST_ES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁÉÍÓÚÜÑáéíóúüñ0123456789.-,:;()°/ "

def remove_speckles_opening(bin_white_bg, k=3, it=1):
    """
    Opening morfológico para quitar punticos.
    k=3 o 5; it=1. Si te come detalles, reduce k o it.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(bin_white_bg, cv2.MORPH_OPEN, kernel, iterations=it)

def remove_speckles_cc(bin_white_bg, min_area=150):
    """Quita manchas chicas por área (texto=0, fondo=255)."""
    inv = 255 - bin_white_bg
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    clean_inv = np.zeros_like(inv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_inv[labels == i] = 255
    return 255 - clean_inv

def preprocess_block_for_ocr(roi_bgr, title = False):
    # 1) binariza+desruida
    bin_img = preprocess_image_for_ocr(roi_bgr)
    # 2) quita subrayados (líneas horizontales)
    bin_img = remove_horizontal_lines(bin_img, min_frac=0.12)
    # 3) enmascara bordes laterales (evita manchas/tornillos)
    bin_img = mask_center_band(bin_img, left_margin=0.06, right_margin=0.06)
    # 4) quita punticos/chirridos
    if title:
        bin_img = remove_speckles_cc(bin_img, min_area=150)
    # 5) upsample (mejora DPI para Tesseract)
    bin_img = cv2.resize(bin_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return bin_img

def ocr_spanish_lines(img_bin, psm=6, conf_min=55):
    cfg = (
        f"-l spa --oem 1 --psm {psm} "
        f"-c preserve_interword_spaces=1 "
        f"-c tessedit_char_whitelist={WHITELIST_ES}"
        f"-c tessedit_do_invert=0"
    )
    data = pytesseract.image_to_data(img_bin, config=cfg, output_type=pytesseract.Output.DICT)

    # reconstruir por línea y filtrar por confianza
    lines = {}
    for i, w in enumerate(data["text"]):
        if not w or not w.strip(): 
            continue
        try:
            conf = float(data["conf"][i])
        except:
            conf = -1
        if conf < conf_min:     # <--- filtra tokens basura
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((data["word_num"][i], w))

    out_lines = []
    for k in sorted(lines.keys()):
        words = [w for _, w in sorted(lines[k], key=lambda t: t[0])]
        out_lines.append(" ".join(words))

    text = "\n".join(out_lines).strip()
    return text

import cv2
import numpy as np
import pytesseract

# Ruta absoluta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print(pytesseract.get_tesseract_version())


def read_numeric_code(roi_bgr, trim_border=4, mask_top_frac=0.22, debug=False):
    """
    Lee un código numérico tipo '344' dentro de un recuadro blanco.
    - trim_border: pixeles a recortar en cada lado para quitar el marco negro
    - mask_top_frac: fracción superior del recorte a ignorar (manchas/rayas)
    """
    # 1) Gris + Otsu (por si viene en color)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if roi_bgr.ndim == 3 else roi_bgr.copy()
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # fondo=blanco, dígitos=negro

    # 2) Recortar borde negro (marco)
    H, W = th.shape
    x0 = max(0, trim_border); y0 = max(0, trim_border)
    x1 = max(x0+1, W - trim_border); y1 = max(y0+1, H - trim_border)
    th = th[y0:y1, x0:x1]

    # 3) Enmascarar franja superior con ruido
    h2, w2 = th.shape
    top_cut = int(h2 * mask_top_frac)
    th[:top_cut, :] = 255  # la parte superior se vuelve blanca

    # 4) Closing suave para unir trazos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5) Upscale x3
    up = cv2.resize(th, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 6) OCR estrictamente numérico (una sola palabra)
    config = (
        "-l eng --oem 1 --psm 8 "
        "-c tessedit_char_whitelist=0123456789 "
        "-c classify_bln_numeric_mode=1 "
        "-c tessedit_do_invert=0"
    )
    txt = pytesseract.image_to_string(up, config=config).strip()

    # Fallbacks si viene vacío
    if not txt:
        config2 = (
            "-l eng --oem 1 --psm 7 "
            "-c tessedit_char_whitelist=0123456789 "
            "-c classify_bln_numeric_mode=1 "
            "-c tessedit_do_invert=0"
        )
        txt = pytesseract.image_to_string(up, config=config2).strip()
    if not txt:
        config3 = (
            "-l eng --oem 1 --psm 13 "
            "-c tessedit_char_whitelist=0123456789 "
            "-c classify_bln_numeric_mode=1 "
            "-c tessedit_do_invert=0"
        )
        txt = pytesseract.image_to_string(up, config=config3).strip()

    if debug:
        cv2.imshow("roi_bgr", roi_bgr if roi_bgr.ndim==3 else cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR))
        cv2.imshow("bin(th)", th)
        cv2.imshow("up", up)
        cv2.waitKey(0); cv2.destroyAllWindows()

    return txt

def preprocess_block_for_ocr(image):
    """Aplica preprocesamiento al bloque antes de OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img


def ocr_spanish_lines_v2(image, psm=6, conf_min=50):
    """Realiza OCR línea por línea con filtro de confianza"""
    custom_config = f'--oem 3 --psm {psm} -l spa'
    data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

    lines = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i].isdigit() else 0
        if text and conf >= conf_min:
            lines.append(text)

    return " ".join(lines)


def read_numeric_code(image):
    """Extrae números (por ejemplo, códigos o valores específicos)"""
    custom_config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text.strip()

if __name__ == "__main__":
    imagen = r"images/input.jpg"
    #imagen = r"images/input.jpeg"
    #imagen = r"images/urapan.jpeg"

    #imagen = r"images/liqui.jpeg"
    template = r"plantilla/Plantilla_0.jpeg" 

    img = cv2.imread(imagen)
    tmp = cv2.imread(template)
    aligned = align_images_robust(img, tmp)

    img_cut = []

    # ROIs relativas: (x_pct, y_pct, w_pct, h_pct)
    OCRLocation = namedtuple("OCRLocation", ["id", "rel_bbox", "filter_keywords"])
    rois_rel = [
        OCRLocation("codigo", (0.014, 0.040, 0.15, 0.16), ["codigo", ":"]),  # CÓDIGO (crudo)
        OCRLocation("bloque_principal", (0.15, 0.20, 0.75, 0.60), []),       # Título + 3 líneas
    ]

    h_al, w_al = aligned.shape[:2]

    def rel_to_abs(rel, w, h):
        x_pct, y_pct, w_pct, h_pct = rel
        return (int(x_pct * w), int(y_pct * h), int(w_pct * w), int(h_pct * h))

    vis = aligned.copy()
    #results = {}

    for loc in rois_rel:
        x, y, ww, hh = rel_to_abs(loc.rel_bbox, w_al, h_al)
        pad_top, pad_bottom, pad_left, pad_right = (2, 2, 2, 2)
        x0 = max(0, x - pad_left)
        y0 = max(0, y - pad_top)
        x1 = min(w_al, x + ww + pad_right)
        y1 = min(h_al, y + hh + pad_bottom)
        roi = aligned[y0:y1, x0:x1]
        img_cut.append(roi)
        



    def ocr_spanish_lines_v2(img_bin, psm=6, whitelist=WHITELIST_ES):
        """
        Realiza OCR en español y reconstruye el texto por líneas (manteniendo espacios).
        img_bin: imagen binaria (texto negro, fondo blanco).
        psm: 6 para párrafo; 7 si título; 8 para 'una palabra'.
        Retorna: (texto_con_saltos, conf_media)
        """
        cfg = (
            f"-l spa --oem 1 --psm {psm} "
            f"-c preserve_interword_spaces=1 "
            f"-c tessedit_char_whitelist=\"{whitelist}\" "
            f"-c tessedit_do_invert=0"
        )

        data = pytesseract.image_to_data(img_bin, config=cfg, output_type=pytesseract.Output.DICT)

        # Reagrupar por línea (block, par, line) y ordenar por word_num
        lines = {}
        for i, word in enumerate(data["text"]):
            if not word or not word.strip():
                continue
            conf = data["conf"][i]
            try:
                conf = float(conf)
            except:
                conf = -1.0
            if conf < 0:
                continue

            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append((data["word_num"][i], word))

        # Orden y unión con espacios, preservando salto de línea
        ordered_keys = sorted(lines.keys())
        text_lines = []
        for k in ordered_keys:
            words_sorted = [w for _, w in sorted(lines[k], key=lambda t: t[0])]
            text_lines.append(" ".join(words_sorted))

        # De-hifen automático (ej. "rosa-" + salto → "rosa")
        text = "\n".join(text_lines)
        text = text.replace("-\n", "")  # une palabras cortadas por guion al final de línea

        # confianza media
        confs = []
        for c in data["conf"]:
            try:
                cf = float(c)
                if cf >= 0: confs.append(cf)
            except:
                pass
        mean_conf = float(np.mean(confs)) if confs else -1.0

        return text.strip(), mean_conf

        
    # Procesar bloque

    # B) Robusto (binariza + quita subrayado + bordes + punticos)
    bin_block = preprocess_image_for_ocr(img_cut[1])

    config = "--oem 1 --psm 6 -l spa"  
    texto = pytesseract.image_to_string(bin_block, config=config)
    print("extraccion texto bloque normal")
    print(texto, "\n")
    print("Texto con preprocesamiento:")

    # ========= USO CON TU ROI =========
    roi_bgr = img_cut[1]                         # bloque principal (BGR crudo)
    bin_block_title = preprocess_block_for_ocr(roi_bgr, title=True)
    bin_block_ = preprocess_block_for_ocr(roi_bgr)

    # separa título (~top 28%) y cuerpo (resto)
    h_t = bin_block_title.shape[0]
    h_b = bin_block_.shape[0]
    bin_top  = bin_block_title[:int(h_t*0.28), :]
    bin_body = bin_block_[int(h_b*0.28):, :]

    # OCR
    titulo   = ocr_spanish_lines(bin_top,  psm=7, conf_min=55)  # una línea
    bloque   = ocr_spanish_lines(bin_body, psm=6, conf_min=25)  # párrafo

    print("Título:\n", titulo)
    print("Bloque:\n", bloque, "\n")


    # # 2) (Opcional) Quitar subrayados largos
    # clean = remove_horizontal_lines(clean_img, min_frac=0.18)

    # # 3) >>> Aquí va mask_center_band <<<
    # center = mask_center_band(clean, left_margin=0.08, right_margin=0.08)

    # # 4) (Opcional) Remover punticos residuales
    # center_np = remove_speckles_opening(center, k=3, it=1)


    # Procesar codigo

    variant2 = {"name":"clahe_otsu",
                "crop_green":False,
                "clahe":True,
                "thr_type":"otsu",
                "invert":False,
                "rm_blobs":True}

    CodeProcessed_img = preprocess_variant(img_cut[0], variant2)
    roi_codigo = CodeProcessed_img  # tu ROI del código
    crop_code, (x0,y0,x1,y1), mask = detect_white_box(roi_codigo, margin=4, debug=False)
    crop_code = remove_speckles_cc(crop_code)
    cv2.imshow("Rectángulo blanco recortado", crop_code)
    cv2.imwrite("codigo.jpg", crop_code)
    # Extraer texto codigo
    text2 = read_numeric_code(crop_code, trim_border=5, mask_top_frac=0.25, debug=False)
    text, conf = ocr_with_conf(crop_code, psm=6, whitelist="0123456789")
    print("El número que reconocio es:",text)
    print("El número2 que reconocio es:",text2)
    cv2.imshow("1 binario", bin_block)
    cv2.imshow("2 binario", img_cut[0])
    #cv2.imshow("1 codigo", CodeProcessed_img)
    # cv2.imshow("2 sin subrayados", clean)
    # cv2.imshow("3 centrado sin bordes", center)
    # cv2.imshow("4 centrado + opening", center_np)
    #cv2.imshow("5 bloque", img_cut[0])
    #cv2.imshow("6 codigo", img_cut[1])
    cv2.waitKey(0); cv2.destroyAllWindows()

