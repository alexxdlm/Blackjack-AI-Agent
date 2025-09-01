import os 
import cv2
import numpy as np

def preprocess_card(contour, img):
    """Get suit and rank image with center of card"""
    card_contour = contour[0]
    card_corners = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    x, y, w, h = cv2.boundingRect(card_contour)
    average = np.sum(card_corners, axis=0)/len(card_corners)
    x = int(average[0][0])
    y = int(average[0][1])
    card_img = transform_card(img, card_corners, w, h)
    corner_img = card_img[0:84, 0:32]
    corner_img_zoom = cv2.resize(corner_img, (0,0), fx=4, fy=4)
    white_level = corner_img_zoom[15,int((32*4)/2)]
    thresh_level = white_level - 30
    if (thresh_level <= 0):
        thresh_level = 1
    _, ths = cv2.threshold(corner_img_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)

    rank_img = ths[20:185, 0:128]
    suit_img = ths[186:336, 0:128]

    # rank
    r_contours, _ = cv2.findContours(rank_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    r_contours = sorted(r_contours, key=cv2.contourArea,reverse=True)
    if len(r_contours) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(r_contours[0])
        rank = rank_img[y1:y1+h1, x1:x1+w1]
        rank_img = cv2.resize(rank, (70,125), 0, 0)


    # suit
    s_contours, _ = cv2.findContours(suit_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    s_contours = sorted(s_contours, key=cv2.contourArea,reverse=True)
    if len(s_contours) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(s_contours[0])
        suit = suit_img[y2:y2+h2, x2:x2+w2]
        suit_img = cv2.resize(suit, (70, 100), 0, 0)

    return rank_img, suit_img, x, y


def match_cards(rank_img, suit_img, templates_path, img, x, y):
    """Classify the card"""
    templates_number = os.listdir(f"{templates_path}/numbers")
    templates_suits = os.listdir(f"{templates_path}/symbols")
    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    if rank_img.any() and suit_img.any():
        for numbers in templates_number:
            template = cv2.imread(f"{templates_path}/numbers/{numbers}")
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            diff = cv2.absdiff(rank_img, template)
            num_diff = int(np.sum(diff)/ 255)

            if num_diff < best_rank_match_diff:
                best_rank_match_diff = num_diff
                best_rank_match_name = numbers.split(".")[0]

        for suits in templates_suits:
            template = cv2.imread(f"{templates_path}/symbols/{suits}")
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            diff = cv2.absdiff(suit_img, template)
            suit_diff = int(np.sum(diff)/ 255)

            if suit_diff < best_suit_match_diff:
                best_suit_match_diff = suit_diff
                best_suit_match_name = suits.split(".")[0]

    img = write_card_pred(x, y, img, best_rank_match_name, best_suit_match_name)

    # cv2.imshow("Card Prediction", img)
    # cv2.waitKey(0)

    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff, img



def transform_card(img, corners, w, h):
    """Handle orientation and perspective of the card in order to get a rectangular
    standardized format"""
    # corners we want tl, tr, br, bl format
    rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(corners, axis = 2)

    tl = corners[np.argmin(s)]
    br = corners[np.argmax(s)]

    diff = np.diff(corners, axis = 2)
    tr = corners[np.argmin(diff)]
    bl = corners[np.argmax(diff)]

    # handle different orientation
    if w <= 0.8*h:
        # vertically oriented
        rect[0] = tl
        rect[1] = tr
        rect[2] = br
        rect[3] = bl

    if w >= 1.2*h:
        # horizontally oriented
        rect[0] = bl
        rect[1] = tl
        rect[2] = tr
        rect[3] = br
    
    if w > 0.8*h and w < 1.2*h:
        # dimaond oriented to the left
        if corners[1][0][1] <= corners[3][0][1]:
            rect[0] = corners[1][0] 
            rect[1] = corners[0][0] 
            rect[2] = corners[3][0] 
            rect[3] = corners[2][0] 

        # diamond oriented to the right
        if corners[1][0][1] > corners[3][0][1]:
            rect[0] = corners[0][0] 
            rect[1] = corners[3][0] 
            rect[2] = corners[2][0] 
            rect[3] = corners[1][0] 


    maxWidth = 200
    maxHeight = 300

    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(rect,dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    return warp


def write_card_pred(cx, cy, img, rk, su):
    """Write name on top of the card"""
    cv2.putText(img, (f"{rk} of {su}"), (round(0.9 * cx), cy - 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
    return img