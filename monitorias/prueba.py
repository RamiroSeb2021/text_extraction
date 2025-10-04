import pytesseract
import cv2

# Configurar la ruta al ejecutable de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Cargar la imagen
img = cv2.imread("images/input.jpg")

# Reconocimiento OCR en español
text = pytesseract.image_to_string(img, lang="eng")

print("Texto detectado:")
print(text)
