import cv2
import numpy as np
import math

# cap is pointer to webcam
cap = cv2.VideoCapture(0)

# as long as webcam is open
while(cap.isOpened()):
    # read image

    ret, img = cap.read()

    # get hand data from the rectangle sub window on the screen
    # subscreen of 300, 300 and 100, 100
    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
    # crop image to that rectangle
    crop_img = img[100:300, 100:300]

    # convert to grayscale to find ROI
    # this is going to help in contour extraction as well (since finding contours is hard on an RGB image)
    grey =  cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur to reduce noise and smoothen the image
    # the absolute difference would be high otherwise
    # here we are not interested in the details but the shape of the object
    # if we do not blur, there will be sharp  = movement of the hand by just 1 pixel,
    # the information of the hand would be lost. Blurring will remove this effect
    # and we want a smooth transition for the shape
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)

    # thresholding: Otsu - Binarization

    _,thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    cv2.imshow('Thresholded', thresh1)

    contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # finding contours with max area (hand)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt)

    # drawing contours
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 0)

    # finding convex hull
    hull = cv2.convexHull(cnt, returnPoints = False)

    # finding convexity defects
    # assumption is that any defect is because of the fingers
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    # drawing contours on the defects
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    # hands detected

    # applying Cosine rule to find angle for all defects - fingers (b/w fingers)
    # with angle > 90 and ignore defects
    print(defects.shape[0])

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        # finding length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # applying cosine rule

        angle = math.cos((b**2 - c**2 - a**2)/(2*b*c)) * 57

        # ignoring angles > 90 and highlighting rest with red dots
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 1, [0, 0 , 255], -1)
        

        # define required actions
        if count_defects == 1:
            cv2.putText(img, "1 finger detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 2:
            cv2.putText(img, "2 fingers detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img, "3 fingers detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 4:
            cv2.putText(img, "4 fingers detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img, "Hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            
        # showing appropriate images in windows
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)

        k = cv2.waitKey(10)
        if k == 27:
            break



