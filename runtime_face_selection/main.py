import cv2 as cv
from face_detection import FaceDetection


colourDict = {
    "man": (255, 0, 0),
    "woman": (0, 0, 255)
}

def main():
    video_stream = cv.VideoCapture(0)
    detector = FaceDetection()

    while True:
        ret, frame = video_stream.read()
        rects, labels = detector.process_frame(frame)
        for (x, y, w, h), label in zip(rects, labels):
            cv.rectangle(frame, (x, y), (x + w, y + h), colourDict[label], 2)
            cv.putText(frame, label, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        cv.imshow("EyeTracker", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video_stream.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()