import cv2 as cv
from imutils import face_utils
import dlib
import numpy as np
from PIL import Image
from tensorflow import keras


MODEL_PATH = "model90-94.h5"
#MODEL_PATH = "model2.h5"

class FaceDetection(object):
    def __init__(self, rect_size=200):
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.eye_position = (.35, .35)
        self.rect_size = rect_size
        self.model = keras.models.load_model(MODEL_PATH)

    def process_frame(self, frame):
        return self.isolate_faces(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), frame)

    def isolate_faces(self, greyscaled_frame, colorized_frame):
        face_rects = self.face_detector(greyscaled_frame, 0)
        bounding_boxes = []
        labels = []
        for rect in face_rects:
            isolated_face = self.align_faces(colorized_frame, rect)
            labels.append(self.labelize_face(isolated_face))
            bounding_boxes.append(face_utils.rect_to_bb(rect))
        return bounding_boxes, labels

    def align_faces(self, frame, rect):
        shape = face_utils.shape_to_np(self.face_predictor(frame, rect))
        lStart, lEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        rStart, rEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        desiredRightEyeX = 1. - self.eye_position[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.eye_position[0]) * self.rect_size
        scale = desiredDist / dist
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        M = cv.getRotationMatrix2D(eyesCenter, angle, scale)
        tX = self.rect_size * .5
        tY = self.rect_size * self.eye_position[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        w, h = self.rect_size, self.rect_size
        output = cv.warpAffine(frame, M, (w, h), flags=cv.INTER_CUBIC)

        return output

    def resize_ratio(self, img, size):
        old_size = img.size  # old_size[0] is in (width, height) format
        ratio = float(size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img.thumbnail(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (size, size), color='white')
        new_im.paste(img, ((size - new_size[0]) // 2,
                           (size - new_size[1]) // 2))
        return new_im

    def labelize_face(self, face):
        feed = []
        #cv.imshow("face", face)
        newimg = Image.fromarray(face)
        newimg = self.resize_ratio(newimg, 96)
        newimg = np.array(newimg)
        feed.append(newimg)
        feed = np.array(feed)
        predictions = self.model.predict(feed)
        if np.argmax(predictions[0]) == 0:
            return "man"
        return "woman"