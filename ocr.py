import cv2
import numpy as np
import pytesseract

invoice_info = {}


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # image = cv2.medianBlur(image, 3)
    return image


def remove_borders(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()

    cntrsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cntr = cntrsSorted[-1]
    # largest_contour = max(contours, key=cv2.contourArea)
    x, y, h, w = cv2.boundingRect(cntr)

    img_copy = cv2.drawContours(
        img_copy, contours, -1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA
    )
    cv2.imwrite("img_copy.png", img_copy)

    crop = image[y : y + h, x : x + w]
    return crop


img = cv2.imread("image4.png")


img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
no_noise = noise_removal(img)

colors = [0, 255, 0]

top, bottom, left, right = [150] * 4


border_img = cv2.copyMakeBorder(
    no_noise, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colors
)
no_borders = remove_borders(img)

cv2.imwrite("test.png", no_borders)


text = pytesseract.image_to_string(no_borders, lang="eng")


# making an list from the data received from the tesseract
invoice_data = text.split("\n")

# removing any unnecessary spaces and gaps
cleaned_data = [item for item in invoice_data if item != " " and item != ""]

# print(cleaned_data)

for i, data in enumerate(cleaned_data):
    if i == 1:
        invoice_info["Company Name"] = data

    if "Invoice No" in data:
        invoice_info["Invoice No"] = data
    # print(data)
# for data in cleaned_data:
# print(invoice_info)
