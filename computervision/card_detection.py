import cv2

min_a = 70000 # Minimum area for a contour to be considered a card
max_a = 120000

def preprocess(img):
    """Return a thresholded version of the inital image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray,(5,5),0)
    
    # Thresholding
    _, ths = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # cv2.imshow("Thresholded Image", ths)
    # cv2.waitKey(0)
    return ths 

def find_cards(prep_img):
    """Detect all the cards in the image"""
    contours, hierarchy = cv2.findContours(prep_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], 0
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_cont = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > min_a:
            perimeter = cv2.arcLength(contour, True)
            approxpoly = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approxpoly) == 4:
                card_cont.append(contour)

    # image_with_contours = cv2.imread("/Users/alessandroorsi/blackjack01/computervision/images/Games/test1.jpg")

    # Evidenzia contorni carte
    # cv2.drawContours(image_with_contours, card_cont, -1, (0, 255, 0), 3)
    # cv2.imwrite("image_with_contours.jpg", image_with_contours)
    # cv2.imshow("Carte Rilevate", image_with_contours)
    # cv2.waitKey(0)


    return card_cont, len(card_cont)