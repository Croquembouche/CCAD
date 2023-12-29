from base64 import decode
import numpy as np
import cv2 as cv
import qrcode
import numpy as np
import time
# example data
# data = "TestV-White-Robot"
# # output file name
# filename = "qrcode.png"
# # generate qr code
# img = qrcode.make(data)
# # save img to a file
# img.save(filename)

# Read the video into the program

# start reading the video
capture = cv.VideoCapture('goofflane.avi')
# capture = cv.VideoCapture("/dev/video6")
# frame_width = int(capture.get(3))
# frame_height = int(capture.get(4))
# out = cv.VideoWriter('goofflane.avi',cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

# background subtraction
total = 0
counter = 0
average = 0
backSub = cv.createBackgroundSubtractorKNN(history=1000, dist2Threshold=200, detectShadows=False)
while True:
    ret, frame = capture.read()

    if frame is None:
        break
    counter += 1
    start = time.time()
    fgMask = backSub.apply(frame)     
    
    # set a kernel size
    kernel = np.ones((10,10),np.uint8)
    # remove very small motion with opening
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
    # add small bubbles within the obj with dilation, also expands the area of interest
    fgMask = cv.dilate(fgMask,kernel,iterations = 5)
    
    # crop out the region of interest
    contours, hierarchy = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # draw lane lines
    LL_start = (232,33)
    LL_end = (184,368)
    RL_start = (365, 20)
    RL_end = (376, 359)
    frame = cv.line(frame, LL_start, LL_end, (22, 218, 253), 3)
    frame = cv.line(frame, RL_start, RL_end, (22, 218, 253), 3)
    # find the eqn of the two lines so we can quickly find the y values later when comparing the centroid
    LL_coefficients = np.polyfit(LL_start, LL_end, 1)
    LL_a = LL_coefficients[0]
    LL_b = LL_coefficients[1]
    RL_coefficients = np.polyfit(RL_start, RL_end, 1)
    RL_a = RL_coefficients[0]
    RL_b = RL_coefficients[1]
    # Print the findings
    # print( 'a =', LL_coefficients[0])
    # print ('b =', LL_coefficients[1])
    # find the TL and BR of all bounding boxes
    temp_TL_x=999
    temp_TL_y=999
    temp_BR_x=0
    temp_BR_y=0
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        # cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if x < temp_TL_x:
            temp_TL_x = x 
        if y < temp_TL_y:
            temp_TL_y = y
        if x+w > temp_BR_x:
            temp_BR_x = x+w 
        if y+h > temp_BR_y:
            temp_BR_y = y+h  

        # see if the vehicle is at the center of the lane
        # i know the eqn of the two lines, i know the y, find the x, y=ax+b, x = (y-b)/a
        # LL_x = LL_a*cy+LL_b
        # RL_x = abs(cy-RL_b)/RL_a
        # print(LL_x, RL_x)
        # frame = cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # frame = cv.circle(frame, (LL_x, cy), 5, (0, 0, 255), -1)
        LL_x = int((cy-1652.17)/-6.98)
        RL_x = int(RL_a*cy+RL_b)
        # print(LL_x)
        frame = cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        frame = cv.circle(frame, (RL_x, cy), 5, (255, 0, 0), -1)
        frame = cv.circle(frame, (LL_x, cy), 5, (0, 255, 0), -1)

        threshold = abs(LL_x-RL_x)/10

        distance_to_LL = RL_x - cx
        distance_to_RL = cx-LL_x
        distance_info = str(distance_to_LL) +", "+ str(distance_to_RL)
        frame = cv.putText(frame, str(distance_to_RL), (LL_x, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        frame = cv.putText(frame, str(distance_to_LL), (RL_x, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
    
        
    

    if temp_BR_x < temp_TL_x or temp_BR_y < temp_TL_y:
        # print("Nothing in scene")
        print('')
    else:
        cropped_image = frame[temp_TL_y:temp_BR_y, temp_TL_x:temp_BR_x]

        # apply super-resolution to increase the resolution so we can see the QR code
        # With a higher resolution camera, we won't need to do super resolution
        sr = cv.dnn_superres.DnnSuperResImpl_create()
        path = "ESPCN_x4.pb"
        sr.readModel(path)
        sr.setModel("espcn",4)
        upscaled = sr.upsample(cropped_image)
        cropped_image = sr.upsample(cropped_image)

        
        # cv.imshow('ROI', cropped_image)
        # cv.imshow('Masked', res)


        # try finding the qr code in masked image
        det=cv.QRCodeDetector()
        decoded_text, points, st_code=det.detectAndDecode(cropped_image)
        if points is not None and len(decoded_text) != 0:
        # QR Code detected handling code
            # print("Detected value is: ", val)
            # nrOfPoints = len(points)
    
            # frame = cv.putText(frame, decoded_text, (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
            # for i in range(points):
            #     nextPointIndex = (i+1) % points
            #     cv.line(frame, tuple(points[i][0]), tuple(points[nextPointIndex][0]), (255,0,0), 5)
            print(decoded_text)
            
            
        else:
            # print("QR code not detected")
            print('')
    end = time.time()
    total = total + (end-start)
    average = total/counter
    print("average process time is:", average*1000, " ms.")
        # cv.imshow('Centroid and Distance', frame)
        # keyboard = cv.waitKey(5)
        # if keyboard == 'q' or keyboard == 27:
        #     break

capture.release()
# out.release()
cv.destroyAllWindows()


# run yolov5/ssd512

# get centroid of each object

# calculate distance from centroid to LL and RL

# find the threshold value (RL-RL)/10

# determine if the vehicle is in the center