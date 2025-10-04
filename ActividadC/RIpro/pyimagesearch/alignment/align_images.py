#!/usr/bin/env python3
"""
align_images.py
Alinea `input` respecto a `template` usando detectores/descriptores (SIFT u ORB),
emparejamiento con Lowe ratio test, cálculo de homografía con RANSAC y warp.
Guarda: aligned.png, matches.png, overlay.png
"""

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

# ---------------------------
# 1. Crear detector dinámico
# ---------------------------
def create_detector(use_sift=True, max_features=5000):
    """Crear detector (SIFT si está disponible, si no ORB)."""
    if use_sift:
        try:
            return cv2.SIFT_create(nfeatures=max_features)
        except Exception:
            print("[WARN] SIFT no disponible -> usando ORB")
    return cv2.ORB_create(nfeatures=max_features)

# ---------------------------
# 2. Emparejar descriptores
# ---------------------------
def match_descriptors(desc1, desc2, use_sift=True, ratio=0.75):
    """Empareja descriptores con FLANN (SIFT) o BF Hamming (ORB)."""
    if desc1 is None or desc2 is None:
        return []

    if use_sift:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(np.float32(desc1), np.float32(desc2), k=2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)
    return good

# ---------------------------
# 3. Alineación de imágenes
# ---------------------------
def align_images(image, template, use_sift=True, max_features=5000,
                 keep_percent=0.15, ratio=0.75, ransac_thresh=5.0, debug=False):
    """Alinea image -> template. Devuelve aligned, homography, matches_img, overlay."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    detector = create_detector(use_sift=use_sift, max_features=max_features)
    kps1, desc1 = detector.detectAndCompute(img_gray, None)
    kps2, desc2 = detector.detectAndCompute(tpl_gray, None)

    if len(kps1) < 4 or len(kps2) < 4:
        raise RuntimeError("Muy pocos keypoints detectados. Prueba aumentar max_features.")

    matches = match_descriptors(desc1, desc2, use_sift=use_sift, ratio=ratio)
    if len(matches) < 4:
        raise RuntimeError(f"Pocos matches buenos ({len(matches)}).")

    matches = sorted(matches, key=lambda x: x.distance)
    num_keep = max(4, int(len(matches) * keep_percent))
    matches_to_use = matches[:num_keep]

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches_to_use]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches_to_use]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("No se pudo calcular homografía.")

    h_tpl, w_tpl = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w_tpl, h_tpl))

    matches_img = cv2.drawMatches(image, kps1, template, kps2, matches_to_use, None,
                                  matchColor=(0,255,0), singlePointColor=(255,0,0),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    overlay = cv2.addWeighted(template, 0.5, aligned, 0.5, 0)

    if debug:
        print(f"[INFO] Keypoints img: {len(kps1)}, tpl: {len(kps2)}")
        print(f"[INFO] Matches: {len(matches)} -> usando {len(matches_to_use)}")
        print(f"[INFO] Homography:\n{H}")

    return aligned, H, matches_img, overlay

# ---------------------------
# 4. Mostrar imágenes
# ---------------------------
def imshow_local(title, img):
    """Mostrar imagen con matplotlib (más estable en Windows)."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.title(title)
    plt.axis('off')
    plt.imshow(img_rgb)
    plt.show()

# ---------------------------
# 5. Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Imagen a alinear")
    ap.add_argument("-t", "--template", required=True, help="Plantilla")
    ap.add_argument("-o", "--out", default="aligned.png", help="Salida aligned")
    ap.add_argument("--matches", default="matches.png", help="Salida matches")
    ap.add_argument("--overlay", default="overlay.png", help="Salida overlay")
    ap.add_argument("--use_sift", action="store_true", help="Usar SIFT (si disponible)")
    ap.add_argument("--max_features", type=int, default=5000)
    ap.add_argument("--keep_percent", type=float, default=0.15)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac", type=float, default=5.0)
    ap.add_argument("--show", action="store_true", help="Mostrar resultados")
    ap.add_argument("--debug", action="store_true", help="Mensajes extra")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    template = cv2.imread(args.template)
    if image is None or template is None:
        raise SystemExit("No pude leer las imágenes. Revisa las rutas.")

    aligned, H, matches_img, overlay = align_images(
        image, template,
        use_sift=args.use_sift,
        max_features=args.max_features,
        keep_percent=args.keep_percent,
        ratio=args.ratio,
        ransac_thresh=args.ransac,
        debug=args.debug
    )

    cv2.imwrite(args.out, aligned)
    cv2.imwrite(args.matches, matches_img)
    cv2.imwrite(args.overlay, overlay)

    print(f"[OK] aligned -> {args.out}")
    print(f"[OK] matches -> {args.matches}")
    print(f"[OK] overlay -> {args.overlay}")

    if args.show:
        imshow_local("template", template)
        imshow_local("aligned", aligned)
        imshow_local("matches", matches_img)
        imshow_local("overlay", overlay)

if __name__ == "__main__":
    main()
