import cv2
from pyimagesearch.alignment.align_images import align_images

def main():
    template = cv2.imread("images/template.jpg")
    image = cv2.imread("images/input.jpg")

    aligned, H = align_images(image, template)
    cv2.imshow("Template", template)
    cv2.imshow("Aligned", aligned)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
