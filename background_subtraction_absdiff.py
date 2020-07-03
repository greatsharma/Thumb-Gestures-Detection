import cv2
import time
import numpy as np


vidcap = cv2.VideoCapture(0)
first_frame = None

count = 0
while True:
    status, frame = vidcap.read()
    if not status:
        break

    frame = cv2.flip(frame, 1, 0)

    # draw bounding box
    frame_h, frame_w = frame.shape[:2]
    start_coords = (int(frame_w * 0.6), int(frame_h * 0.05))
    end_coords = (int(frame_w * 0.95), int(frame_h * 0.55))
    cv2.rectangle(frame, start_coords, end_coords, (255,0,0), 2)
    cv2.imshow("original", frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (11, 11), 0)

    count += 1
    if count > 30:
        bb = gray_frame[start_coords[1]:end_coords[1], start_coords[0]: end_coords[0]]
        if first_frame is None:
            first_frame = bb
            continue

        frame_delta = cv2.absdiff(first_frame, bb)
        thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
        hull = [cv2.convexHull(c) for c in max_cnt]

        try:
            cv2.drawContours(bb, hull, -1, (0,0,0), thickness=3)
            # (x, y, w, h) = cv2.boundingRect(max_cnt[0])
            # cv2.rectangle(bb, (x, y), (x + w, y + h), (0, 0, 0), 4)
        except Exception:
            pass
        
        combined = np.hstack((bb, thresh))
        cv2.imshow("bb v/s bs", combined)

    if cv2.waitKey(10) == ord("q"):
        break

vidcap.release()
cv2.destroyAllWindows()