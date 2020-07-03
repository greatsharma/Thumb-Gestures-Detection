import cv2
import time
import joblib
import argparse
import numpy as np
from tensorflow.keras.models import load_model

import utils


ap = argparse.ArgumentParser()
ap.add_argument('-mo', '--mode', required=False,
                help='either `debug` or `prod` (production), default is prod')
args = vars(ap.parse_args())

if not "mode" in args:
    args["mode"] = "prod"

model = load_model("dumps/model.h5")
label2text = joblib.load("dumps/label2text.pkl")

vidcap = cv2.VideoCapture(0)
first_frame = None

tt = 0
frame_count = 0
while True:
    status, frame = vidcap.read()
    if not status:
        break

    frame = cv2.flip(frame, 1, 0)

    frame_count += 1
    tik = time.time()

    # draw bounding box
    frame_h, frame_w = frame.shape[:2]
    start_coords = (int(frame_w * 0.5), int(frame_h * 0.05))
    end_coords = (int(frame_w * 0.95), int(frame_h * 0.6))
    cv2.rectangle(frame, start_coords, end_coords, (255,0,0), 2)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7,7), 0)

    if frame_count > 120:
        bb = gray_frame[start_coords[1]:end_coords[1], start_coords[0]: end_coords[0]]

        if first_frame is None:
            first_frame = bb
            continue

        frame_delta = cv2.absdiff(first_frame, bb)
        thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        try:
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
            (x, y, w, h) = cv2.boundingRect(max_cnt[0])

            area_occupied = (w * h) / (bb.shape[0] * bb.shape[1])
            if area_occupied < 0.1:
                raise Exception

            if args["mode"] == "debug":
                combined = np.hstack((bb, thresh))
                cv2.imshow("bb v/s thresh", combined)

            img = cv2.resize(thresh, (120, 120))
            img = np.expand_dims(img, axis=2)
            img = np.expand_dims(img, axis=0)
            img = img / 255.

            predicted_proba = model.predict(img)
            predicted_label = np.argmax(predicted_proba[0])
            bb_text = label2text[predicted_label]
        except Exception as e:
            bb_text = "no hand"
        
        utils.draw_text_with_backgroud(frame, bb_text, x=start_coords[0], y=start_coords[1], font_scale=0.4)
        tt += time.time() - tik
        fps = round(frame_count / tt, 2)
        main_text = "Go On..." + f"   fps: {fps}"
        utils.draw_text_with_backgroud(frame, main_text, x=15, y=25, font_scale=0.35, thickness=1)
        utils.draw_text_with_backgroud(frame, "Instructions for better results :-", x=15, y=55, font_scale=0.32, thickness=1)
        utils.draw_text_with_backgroud(frame, "- Place your hand completely inside the window", x=15, y=75, font_scale=0.32, thickness=1)
        utils.draw_text_with_backgroud(frame, "- Place your hand close to window", x=15, y=95, font_scale=0.32, thickness=1)
    else:
        main_text = "Within 5 seconds ensure that the background behind the window doesn't change"
        utils.draw_text_with_backgroud(frame, main_text, x=15, y=25, font_scale=0.35, thickness=1)

    cv2.imshow("out", frame)
    if cv2.waitKey(10) == ord("q"):
        break

vidcap.release()
cv2.destroyAllWindows()