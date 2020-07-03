import cv2
import numpy as np


back_subtractor = cv2.createBackgroundSubtractorMOG2()

vidcap = cv2.VideoCapture(0)
while True:
    status, frame = vidcap.read()
    if not status:
        break

    frame = cv2.flip(frame, 1, 0)

    # draw bounding box
    frame_h, frame_w = frame.shape[:2]
    start_coords = (int(frame_w * 0.25), int(frame_h * 0.1))
    end_coords = (int(frame_w * 0.95), int(frame_h * 0.9))
    cv2.rectangle(frame, start_coords, end_coords, (255,0,0), 2)
    cv2.imshow("original", frame)
 
    bb = frame[start_coords[1]:end_coords[1], start_coords[0]: end_coords[0]]
    gray_bb = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
    gray_bb = cv2.GaussianBlur(gray_bb, (15, 15), 0)

    bs = back_subtractor.apply(gray_bb)

    combined = np.hstack((gray_bb, bs))
    cv2.imshow("bb v/s bs", combined)
    if cv2.waitKey(10) == ord("q"):
        break

vidcap.release()
cv2.destroyAllWindows()