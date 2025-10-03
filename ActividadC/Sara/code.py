import cv2
import numpy as np
import pytesseract
import os 
import imutils 

# 1. Configurar la ruta a tesseract.exe (Solo se hace una vez)
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception as e:
    print(f"ADVERTENCIA: No se pudo configurar Tesseract. Verifique la ruta.\nError: {e}")

# 2. Cargar plantilla (Solo se carga una vez)
PLANTILLA = cv2.imread("plantilla3.jpg", 0) 
if PLANTILLA is None:
    print("Error: No se pudo cargar la imagen de plantilla. ¡Verifique ruta y nombre!")
    exit()

# 3. Configuración SIFT (Detector robusto) y Matcher
SIFT_DETECTOR = cv2.SIFT_create() 
KP_PLANTILLA, DES_PLANTILLA = SIFT_DETECTOR.detectAndCompute(PLANTILLA, None)

# SIFT usa la norma L2 y knnMatch para el Ratio Test.
BF_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) 

# =====================================================================
# FUNCIÓN CLAVE: PROCESAR UNA SOLA IMAGEN
# =====================================================================

def procesar_imagen_y_extraer_datos(ruta_imagen_foto, plantilla, kp_plantilla, des_plantilla, matcher, feature_detector):
    """
    Carga, aplica Ecualización, alinea con SIFT + Umbrales estrictos, recorta, aplica OCR y visualiza.
    """
    print(f"\n--- Procesando: {ruta_imagen_foto} ---")
    
    foto_gris = cv2.imread(ruta_imagen_foto, 0)
    foto_color = cv2.imread(ruta_imagen_foto) 
    
    if foto_gris is None or foto_color is None:
        print(f"Error: No se pudo cargar la imagen {ruta_imagen_foto}. Saltando.")
        return None
    
    # 4. Pre-procesamiento: Ecualización de Histograma
    foto_ecualizada = cv2.equalizeHist(foto_gris)
        
    kp_foto, des_foto = feature_detector.detectAndCompute(foto_ecualizada, None)

    if des_foto is None or DES_PLANTILLA is None:
        print(f"Advertencia: No se detectaron descriptores suficientes en la imagen {ruta_imagen_foto}. Saltando.")
        cv2.imshow(f"FALLO DE DESCRIPTORES - {os.path.basename(ruta_imagen_foto)} (Original)", foto_color)
        print("Presione una tecla para cerrar y continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    # 5. Hacer matching de características con knnMatch (k=2)
    matches = matcher.knnMatch(DES_PLANTILLA, des_foto, k=2) 

    # 6. Aplicar el Filtro de Ratio de Lowe (UMBRAL EXTREMADAMENTE ESTRICTO)
    good_matches = []
    UMBRAL_RATIO = 0.65 

    for m, n in matches:
        if m.distance < UMBRAL_RATIO * n.distance:
            good_matches.append(m)

    # 7. Calcular Homografía
    if len(good_matches) > 4: 
        
        src_pts = np.float32([kp_plantilla[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_foto[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        RANSAC_REPROJECTION_THRESHOLD = 2.0 
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD)
        
        if M is None: 
            print(f"Fallo de alineación: RANSAC no pudo encontrar una homografía válida para {ruta_imagen_foto}.")
            cv2.imshow(f"FALLO RANSAC - {os.path.basename(ruta_imagen_foto)} (Original)", foto_color)
            print("Presione una tecla para cerrar y continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return None

        h, w = plantilla.shape
        
        alineada_color = cv2.warpPerspective(foto_color, M, (w, h))
        alineada_gris = cv2.warpPerspective(foto_ecualizada, M, (w, h)) 

        # =================================================================
        # 8. VISUALIZACIÓN Y RECORTES (¡COORDENADAS FINALMENTE AJUSTADAS!)
        # x_inicio: 800 (para capturar el inicio del texto)
        # x_fin: 1050 (para capturar el final largo)
        # =================================================================
        
        # Definición de recortes 
        # [y_inicio, y_fin, x_inicio, x_fin, Color (BGR)]
        recortes = {
            # x_inicio ajustado a 800 (Más margen a la izquierda)
            "nombre": [355, 426, 710, 1050, (0, 255, 0)], 
            # x_inicio ajustado a 810 (Más margen a la izquierda)
            "origen": [448, 492, 717, 1050, (255, 0, 0)], 
            # x_inicio ajustado a 840 (Más margen a la izquierda)
            "altura": [521, 570, 765, 1050, (0, 0, 255)], 
        }
        
        datos_extraidos = {}
        
        for key, coords in recortes.items():
            y1, y2, x1, x2, color = coords
            
            # Ajuste de límites para evitar errores de índice
            y1 = max(0, y1)
            y2 = min(alineada_color.shape[0], y2)
            x1 = max(0, x1)
            x2 = min(alineada_color.shape[1], x2)
            
            cv2.rectangle(alineada_color, (x1, y1), (x2, y2), color, 2)
            recorte_img = alineada_gris[y1:y2, x1:x2]
            
            # Aplicar OCR
            texto = pytesseract.image_to_string(recorte_img, lang="spa").strip()
            datos_extraidos[key] = texto
            
            cv2.imshow(f"Recorte {key.capitalize()} - {os.path.basename(ruta_imagen_foto)}", recorte_img)

        cv2.imshow(f"Original - {os.path.basename(ruta_imagen_foto)}", foto_color)
        cv2.imshow(f"ALINEADA con Recortes - {os.path.basename(ruta_imagen_foto)}", alineada_color)
        
        print(f"Datos extraídos: Nombre={datos_extraidos['nombre']}, Origen={datos_extraidos['origen']}, Altura={datos_extraidos['altura']}")
        print("Presione una tecla para cerrar las ventanas y continuar con la siguiente imagen...")
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        
        return {
            "archivo": ruta_imagen_foto,
            "nombre_cientifico": datos_extraidos['nombre'],
            "lugar_originario": datos_extraidos['origen'],
            "altura_ejemplar": datos_extraidos['altura']
        }

    else:
        print(f"Fallo de alineación: Solo se encontraron {len(good_matches)} buenas coincidencias para {ruta_imagen_foto}. No se pudo alinear.")
        cv2.imshow(f"FALLO DE MATCHES - {os.path.basename(ruta_imagen_foto)} (Original)", foto_color)
        print("Presione una tecla para cerrar y continuar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

# =====================================================================
# PARTE PRINCIPAL: PROCESAR TODAS LAS IMÁGENES
# =====================================================================

# 1. Crear una lista con las rutas de TODAS tus imágenes
rutas_de_imagenes = [
    "foto.jpeg",  
    "ejemplo2.jpeg",  # Imagen de Sauco
    # Agrega el resto de tus N imágenes aquí
]

resultados_totales = []

# 2. Iterar sobre la lista y procesar cada imagen
for ruta in rutas_de_imagenes:
    datos_extraidos = procesar_imagen_y_extraer_datos(
        ruta, 
        PLANTILLA, 
        KP_PLANTILLA, 
        DES_PLANTILLA, 
        BF_MATCHER,
        SIFT_DETECTOR 
    )
    if datos_extraidos:
        resultados_totales.append(datos_extraidos)

# 3. Mostrar el resumen final
print("\n====================================")
print("       RESUMEN DE RESULTADOS")
print("====================================")

if not resultados_totales:
    print("No se pudieron extraer datos de ninguna imagen procesada exitosamente.")
else:
    for res in resultados_totales:
        print(f"\nArchivo: {res['archivo']}")
        print(f"Nombre científico: {res['nombre_cientifico']}")
        print(f"Lugar originario: {res['lugar_originario']}")
        print(f"Altura del ejemplar: {res['altura_ejemplar']}")
