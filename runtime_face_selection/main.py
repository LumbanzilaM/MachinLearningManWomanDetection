import cv2 as cv
from face_detection import FaceDetection


def main():
    video_stream = cv.VideoCapture(0)
    detector = FaceDetection()

    while True:
        ret, frame = video_stream.read()
        rects, labels = detector.process_frame(frame)
        for x, y, w, h in rects:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv.imshow("EyeTracker", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video_stream.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()