import argparse
import imutils
import time
import cv2
import os

from imutils.video import VideoStream
from pathlib import Path


def parser():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--cascade", required=False,
        default=str(Path(__file__).parent / "cascade" / "haarcascade_frontalface_default.xml"),
        help = "path to where the face cascade resides")
    ap.add_argument("-s", "--source", required=False,
        default=0,
        help = "create dataset using provided camera id")
    ap.add_argument("-o", "--output", required=True,
        help="path to output directory")
    args = vars(ap.parse_args())
    return args


def create_dataset():
    args = parser()
    output = args["output"]
    cascade = args["cascade"]
    camera = args["source"]
    detector = cv2.CascadeClassifier(cascade)
    print("[INFO] starting video stream...")
    os.makedirs(output, exist_ok=True)
    vs = VideoStream(src=camera).start()
    # wait for a camera to start
    time.sleep(2.0)
    total = 0

    while True:
        frame = vs.read()
        face_frame = frame.copy()
        frame = imutils.resize(frame, width=400)
        rects = detector.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_frame = frame[y:y+h, x:x+w]

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        # if the `k` key was pressed, write the *original* frame to disk
        # so we can later process it and use it for face recognition
        if key == ord("k"):
            p = os.path.sep.join([output, "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, face_frame)
            total += 1
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break

        print("[INFO] {} face images stored".format(total))

    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    create_dataset()
