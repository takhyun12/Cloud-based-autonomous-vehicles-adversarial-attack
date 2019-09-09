import cv2
import cv2 as cv
import numpy as np

# [1] 환경설정
cap = cv2.VideoCapture("video/test2.mp4")
car_cascade = cv2.CascadeClassifier('cars.xml')

# [2] 실시간 비디오 재생
while (True): # 연속적으로 비디오 프레임을 재생함
    ret, frame = cap.read() # 비디오의 정상 동작 여부와 프레임을 읽어옴

    frame = cv2.resize(frame, (450, 260)) # 말끔한 재생을 위한 비디오 크기 리사이징

    canny_frame = cv.Canny(frame, 50, 200, None, 3)  # canny 알고리즘을 통한 외곽선 검출 프레임

    cdstP_frame = cv.cvtColor(canny_frame, cv.COLOR_GRAY2BGR) # 외곽선에 lane을 추출할 프레임 

    line_frame = np.copy(frame)  # 외곽선을 적용한 색상이 있는 프레임

    flip_frame = cv2.flip(frame, 1) # 좌우 반전을 수행시킨 프레임

    # ======================= [ lane object ] =======================

    linesP = cv.HoughLinesP(canny_frame, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP_frame, (l[0], l[1]), (l[2], l[3]), (20, 255, 85), 1, cv.LINE_AA)
            cv.line(line_frame, (l[0], l[1]), (l[2], l[3]), (20, 255, 85), 1, cv.LINE_AA)
            cv.line(flip_frame, (l[0], l[1]), (l[2], l[3]), (20, 255, 85), 1, cv.LINE_AA)

    cv.imshow("source frame", frame)
    cv.imshow("only line", cdstP_frame)
    cv.imshow("line frame", line_frame)

    # HSV 필터를 적용한 프레임
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv_frame, (0, 50, 50), (20, 255, 255))
    #mask2 = cv2.inRange(hsv_frame, (1, 0, 0), (1, 255, 255))
    #mask = cv2.bitwise_or(mask1, mask2)

    attack_frame = cv2.bitwise_and(frame, frame, mask=mask1)
    attack_frame = cv.cvtColor(attack_frame, cv.COLOR_HSV2RGB)

    cv.imshow("lane attack type 1", attack_frame)

    lane_attack_frame = cv.Canny(attack_frame, 50, 200, None, 3)  # canny 알고리즘을 통한 외곽선 검출
    cv.imshow("lane attack type 2", lane_attack_frame)

    # ======================= [ car object ] =======================

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    car_attack_frame = gray_frame.copy()

    kernel = np.array([[-1, -1, -1],
                       [-1, 4, -1],
                       [-1, -1, -1]])

    filter_frame = cv2.filter2D(gray_frame, -1, kernel)

    cars = car_cascade.detectMultiScale(gray_frame, 1.3, 1)

    for (x, y, w, h) in cars:
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(filter_frame, (x, y), (x + w, y + h), (255, 255, 255), -1)

    cv2.imshow('car detect frame', gray_frame)

    car_attack_frame = cv2.inpaint(car_attack_frame , filter_frame, 3, cv2.INPAINT_TELEA)

    '''
    cars = car_cascade.detectMultiScale(car_attack_frame, 1.3, 1)

    for (x, y, w, h) in cars:
        cv2.rectangle(car_attack_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    '''

    cv2.imshow('car_attack_frame', car_attack_frame)

    car_canny_frame = cv.Canny(car_attack_frame, 50, 200, None, 3)  # canny 알고리즘을 통한 외곽선 검출
    cv2.imshow('car_attack_frame (2)', car_canny_frame)

    # ======================= [ flip object ] =======================

    cv2.imshow('flip attack frame', flip_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
