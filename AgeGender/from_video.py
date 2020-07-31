from Retinaface.Retinaface import FaceDetector
from AgeGender.Detector import AgeGender
import cv2
import os

face_detector = FaceDetector(name='mobilenet', weight_path='../Retinaface/weights/mobilenet.pth', device='cuda')
age_gender_detector = AgeGender(name='full', weight_path='weights/ShufflenetFull.pth', device='cuda')

path = '/home/minds/PycharmProjects/FaceRecognizer-motive/OutputF6/'
pathO = '/home/minds/PycharmProjects/facelib/AgeGender/videosample6/'
img_files = os.listdir(path)

for img_file in img_files:
    frame = cv2.imread(path + str(img_file))
    faces, boxes, scores, landmarks = face_detector.detect_align(frame)
    if len(faces.shape) > 1:
        genders, ages = age_gender_detector.detect(faces)
        for i, b in enumerate(boxes):
            cv2.putText(frame, f'{genders[i]},{ages[i]}', (int(b[0]), int(b[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        [0, 200, 0], 3)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 3)

    cv2.imwrite(os.path.join(pathO, str(img_file)), frame)