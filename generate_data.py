import cv2
import time
import argparse
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument('-dt', '--data_type', required=True,
                help='data type, `bb` or `thresh`')
ap.add_argument('-d', '--direction', required=True,
                help='type of direction, `init`, `left`, `right`, `down` and `up`')
ap.add_argument('-it', '--initial_count', required=True)
args = vars(ap.parse_args())

vidcap = cv2.VideoCapture(0)
first_frame = None

count = 0
img_count = 0
while True:
    status, frame = vidcap.read()
    if not status:
        break

    frame = cv2.flip(frame, 1, 0)

    # draw bounding box
    frame_h, frame_w = frame.shape[:2]
    start_coords = (int(frame_w * 0.5), int(frame_h * 0.05))
    end_coords = (int(frame_w * 0.95), int(frame_h * 0.6))
    cv2.rectangle(frame, start_coords, end_coords, (255,0,0), 2)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7,7), 0)

    count += 1
    if count > 60:
        bb = gray_frame[start_coords[1]:end_coords[1], start_coords[0]: end_coords[0]]

        if first_frame is None:
            first_frame = bb
            print("start")
            continue

        frame_delta = cv2.absdiff(first_frame, bb)
        thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        try:
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
            # hull = [cv2.convexHull(c) for c in max_cnt]
            (x, y, w, h) = cv2.boundingRect(max_cnt[0])

            if args["data_type"] == "bb":
                bb = cv2.resize(bb, (120,120))
                cv2.imwrite(f"data_bb/{args['direction']}/{args['direction']}_{img_count + int(args['initial_count'])}.png", bb)
            else:
                thresh = cv2.resize(thresh, (120, 120))
                cv2.imwrite(f"data_thresh/{args['direction']}/{args['direction']}_{img_count + int(args['initial_count'])}.png", thresh)
    
            img_count += 1
        except Exception:
            pass

    cv2.imshow("out", frame)
    if cv2.waitKey(10) == ord("q"):
        break

vidcap.release()
cv2.destroyAllWindows()