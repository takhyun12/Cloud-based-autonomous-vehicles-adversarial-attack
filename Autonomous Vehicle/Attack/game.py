import win32gui
from PIL import ImageGrab
import numpy
import time
import cv2
import numpy as np
from inputkey import PressKey, ReleaseKey, UP
import time

while (True):
    game_screen = ImageGrab.grab(bbox=(465, 200, 1260, 645))
    frame = numpy.array(game_screen)

    #ROI_frame = frame[47:47+269, 4:4+791]  # ROI 영역만 잘라내기

    filtered_frame = np.copy(frame)

    #  BGR 제한 값 설정 (흰색 추출)
    blue_threshold = 200
    green_threshold = 200
    red_threshold = 200
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 필터를 통해 흰색만 추출
    thresholds = (frame[:, :, 0] < bgr_threshold[0]) \
                 | (frame[:, :, 1] < bgr_threshold[1]) \
                 | (frame[:, :, 2] < bgr_threshold[2])
    filtered_frame[thresholds] = [0, 0, 0]

    canny_frame = cv2.Canny(filtered_frame, 50, 200, None, 3)  # canny 알고리즘을 통한 외곽선 검출 프레임
    cdstP_frame = cv2.cvtColor(canny_frame, cv2.COLOR_GRAY2BGR)  # 외곽선에 lane을 추출할 프레임

    # ======================= [ lane object detection ] =======================

    linesP = cv2.HoughLinesP(canny_frame, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            print(l)
            cv2.line(cdstP_frame, (l[0], l[1]), (l[2], l[3]), (20, 125, 185), 1, cv2.LINE_AA)

    # ==========================================================================
    cv2.imshow("game", cdstP_frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

