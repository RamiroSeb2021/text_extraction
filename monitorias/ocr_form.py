import cv2
import pytesseract
from pyimagesearch.alignment.align_images import align_images

# Si estás en Windows, ajusta el path a tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def main():
    # Cargar imágenes
    template = cv2.imread("images/Template2.jpg")
    image = cv2.imread("images/input2.jpg")

    # Alinear
    aligned, _, _, _ = align_images(image, template, use_sift=True)

    # Definir campos (x, y, w, h)
    fields = {
        "nombre": (340, 255, 682, 335),# azul
        "autor": (400, 660, 550, 100),      # negro
        "id_arbol": (10, 40, 240, 180)      # id arbol
    }

    results = {}
    for field, (x, y, w, h) in fields.items():
        roi = aligned[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if field == "id_arbol":
            _, thresh = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # Forzar solo números
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 -c tessedit_char_whitelist=0123456789"
            ).strip()
        else:
            _, thresh = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, lang="eng").strip()



        
        results[field] = text

        # Dibujar rectángulo
        cv2.rectangle(aligned, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(aligned, field, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Mostrar resultados
    for field, text in results.items():
        print(f"[{field.upper()}]: {text}")

    cv2.imshow("Aligned + OCR", aligned)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
