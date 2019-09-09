import win32gui
from PIL import ImageGrab
import numpy
import time
import cv2
import numpy as np
from inputkey import PressKey, ReleaseKey, UP


while (True):
    game_screen = ImageGrab.grab(bbox=(465, 200, 1260, 645))
    frame = numpy.array(game_screen)

    # color filtering through hsv filter
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, (0, 0, 0), (0, 0, 255))
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)
    filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_HSV2RGB)

    canny_frame = cv2.Canny(filtered_frame, 50, 200, None, 3)  # canny 알고리즘을 통한 외곽선 검출 프레임
    cdstP_frame = cv2.cvtColor(canny_frame, cv2.COLOR_GRAY2BGR)  # 외곽선에 lane을 추출할 프레임

    # ======================= [ lane object detection ] =======================

    linesP = cv2.HoughLinesP(canny_frame, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP_frame, (l[0], l[1]), (l[2], l[3]), (20, 255, 85), 1, cv2.LINE_AA)

    # ==========================================================================

    #PressKey(UP)
    cv2.imshow("game", cdstP_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

