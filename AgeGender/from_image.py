from Retinaface.Retinaface import FaceDetector
from AgeGender.Detector import AgeGender
import cv2
from time import time

face_detector = FaceDetector(name='mobilenet', weight_path='../Retinaface/weights/mobilenet.pth', device='cuda')
age_gender_detector = AgeGender(name='full', weight_path='./weights/ShufflenetFull.pth', device='cuda')


frame = cv2.imread('/home/minds/PycharmProjects/FaceRecognizer-motive/OutputF1/763118H01_1_18-TJ-06-036__face1.jpg')
faces, boxes, scores, landmarks = face_detector.detect_align(frame)
if len(faces.shape) > 1:
    tic = time()
    genders, ages = age_gender_detector.detect(faces)
    print(time()-tic)
    for i, b in enumerate(boxes):
        cv2.putText(frame, f'{genders[i]},{ages[i]}', (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, [0, 200, 0], 3)
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 3)

    for p in landmarks:
        for i in range(5):
            cv2.circle(frame, (p[i][0], p[i][1]), 3, (0, 255, 0), -1)

cv2.imshow('frame', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()