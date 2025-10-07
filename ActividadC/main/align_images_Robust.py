import cv2
import numpy as np

def _preprocess_gray(img):
    """Gris + CLAHE + blur suave para mejorar contraste de texto y bordes."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return g

def _make_rect_mask_like(img, margin=10):
    """Máscara rectangular (útil si ya tienes la placa recortada)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[margin:h-margin, margin:w-margin] = 255
    return mask

def _detect_holes_mask(img_gray):
    """Detecta círculos metálicos (tornillos) para usar como landmarks / máscara."""
    # Paramétrico: puedes ajustar según tu resolución
    circles = cv2.HoughCircles(
        img_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=img_gray.shape[0]//6,
        param1=120, param2=20, minRadius=6, maxRadius=60
    )
    pts = None
    mask = np.zeros_like(img_gray, np.uint8)
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        pts = np.array([[c[0], c[1]] for c in circles], dtype=np.float32)
        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), int(r*1.6), 255, -1)  # máscara centrada en orificios
    return pts, mask

def _match_keypoints(imgA, imgB, maskA=None, maskB=None, max_features=4000, use_sift=False):
    """KNN + ratio test. Devuelve ptsA, ptsB filtrados para homografía."""
    gA = _preprocess_gray(imgA)
    gB = _preprocess_gray(imgB)

    if use_sift and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    else:
        # AKAZE es muy fuerte en superficies con poco texto; si no está, usa ORB “potente”
        if hasattr(cv2, "AKAZE_create"):
            detector = cv2.AKAZE_create()
            norm = cv2.NORM_HAMMING
        else:
            detector = cv2.ORB_create(
                nfeatures=max_features, scaleFactor=1.2, nlevels=8,
                edgeThreshold=15, patchSize=31, WTA_K=2
            )
            norm = cv2.NORM_HAMMING

    kpsA, descA = detector.detectAndCompute(gA, maskA)
    kpsB, descB = detector.detectAndCompute(gB, maskB)
    if descA is None or descB is None:
        return None, None

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(descA, descB, k=2)

    ptsA, ptsB = [], []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:  # ratio test de Lowe
            ptsA.append(kpsA[m.queryIdx].pt)
            ptsB.append(kpsB[m.trainIdx].pt)
    if len(ptsA) < 4:
        return None, None
    return np.float32(ptsA), np.float32(ptsB)

def _homography_ransac(ptsA, ptsB):
    H, inliers = cv2.findHomography(
        ptsA, ptsB, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=5000, confidence=0.995
    )
    n_inl = int(inliers.sum()) if inliers is not None else 0
    return H, n_inl

def _ecc_refine(src_gray, dst_gray, init_warp=None, mode="H"):
    """Refina con ECC. mode: 'E' Euclid., 'A' Afín, 'H' Homografía."""
    warp_mode = {
        "E": cv2.MOTION_EUCLIDEAN,
        "A": cv2.MOTION_AFFINE,
        "H": cv2.MOTION_HOMOGRAPHY
    }[mode]

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, dtype=np.float32)
        if init_warp is not None and init_warp.shape == (3,3):
            warp = init_warp.astype(np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)
        if init_warp is not None and init_warp.shape == (2,3):
            warp = init_warp.astype(np.float32)

    try:
        cc, warp = cv2.findTransformECC(
            dst_gray, src_gray, warp, warp_mode, criteria, None, 5
        )
    except cv2.error:
        return None

    return warp

def align_images_robust(image, template, use_sift=False, try_holes=True, debug=False):
    """
    1) Keypoints + Homografía (AKAZE/SIFT/ORB) con máscara opcional
    2) Fallback ECC (afin u homografía)
    3) Anclaje por orificios (si se detectan) + refinamiento ECC
    """
    h, w = template.shape[:2]
    gT = _preprocess_gray(template)
    gI = _preprocess_gray(image)

    # Máscaras (rectangular por defecto) para evitar fondo
    maskT = _make_rect_mask_like(template, margin=12)
    maskI = _make_rect_mask_like(image, margin=12)

    # (Opcional) generar máscaras de orificios (muy útil en tu placa)
    ptsT_holes, maskT_holes = _detect_holes_mask(gT) if try_holes else (None, None)
    ptsI_holes, maskI_holes = _detect_holes_mask(gI) if try_holes else (None, None)
    if maskT_holes is not None: maskT = cv2.bitwise_or(maskT, maskT_holes)
    if maskI_holes is not None: maskI = cv2.bitwise_or(maskI, maskI_holes)

    # --- 1) KEYPOINTS + HOMOGRAFÍA ---
    ptsA, ptsB = _match_keypoints(image, template, maskI, maskT, use_sift=use_sift)
    H, inl = (None, 0)
    if ptsA is not None:
        H, inl = _homography_ransac(ptsA, ptsB)

    if H is not None and inl >= 10:
        aligned = cv2.warpPerspective(image, H, (w, h))
        # Refinar con ECC homografía (opcional)
        warp = _ecc_refine(_preprocess_gray(aligned), gT, init_warp=np.eye(3), mode="H")
        if warp is not None:
            aligned = cv2.warpPerspective(aligned, warp, (w, h))
        if debug: print(f"[Keypoints] inliers={inl}")
        return aligned

    # --- 2) FALLBACK ECC (AFÍN u HOMOGRAFÍA) ---
    warp = _ecc_refine(gI, gT, mode="A")
    if warp is not None:
        aligned = cv2.warpAffine(image, warp, (w, h))
        # pequeño refinamiento homográfico
        warpH = _ecc_refine(_preprocess_gray(aligned), gT, mode="H")
        if warpH is not None:
            aligned = cv2.warpPerspective(aligned, warpH, (w, h))
        if debug: print("[ECC] affine/homography refine OK")
        return aligned

    # --- 3) ORIFICIOS COMO ANCLAJES + ECC ---
    if try_holes and ptsT_holes is not None and ptsI_holes is not None:
        # emparejar por distancia (greedy) si # de círculos coincide
        if len(ptsT_holes) >= 2 and len(ptsI_holes) >= 2:
            # elegir los dos más lejanos en cada imagen (suelen ser los tornillos)
            def two_farthest(pts):
                dmax, pair = -1, (0,1)
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        d = np.linalg.norm(pts[i]-pts[j])
                        if d > dmax: dmax, pair = d, (i,j)
                return np.vstack([pts[pair[0]], pts[pair[1]]]).astype(np.float32)

            A = two_farthest(ptsI_holes)
            B = two_farthest(ptsT_holes)
            M, _ = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if M is not None:
                approx = cv2.warpAffine(image, M, (w, h))
                # refinar con ECC homografía
                warpH = _ecc_refine(_preprocess_gray(approx), gT, mode="H")
                if warpH is not None:
                    approx = cv2.warpPerspective(approx, warpH, (w, h))
                if debug: print("[Holes+ECC] OK")
                return approx

    # Si todo falla, devuelve None
    if debug: print("Alignment failed")
    return None
