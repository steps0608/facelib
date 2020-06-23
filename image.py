import cv2
import argparse
from Retinaface.Retinaface import FaceDetector
from AgeGender.Detector import AgeGender
from FacialExpression.FaceExpression import EmotionDetector
from InsightFace.data.config import get_config
from InsightFace.models.Learner import face_learner
from InsightFace.utils import update_facebank, load_facebank, special_draw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.45, type=float)
    parser.add_argument("-u", "--update", default=True, help="whether perform update the facebank")
    parser.add_argument("-tta", "--tta", default=True, help="whether test time augmentation")
    parser.add_argument('-m', '--mobilenet', default=False, help="use mobilenet for backbone")
    args = parser.parse_args()

    conf = get_config(training=False)
    conf.use_mobilfacenet = args.mobilenet
    face_rec = face_learner(conf, inference=True)
    face_rec.threshold = args.threshold
    face_rec.model.eval()

    face_detector = FaceDetector(name='mobilenet', weight_path='Retinaface/weights/mobilenet.pth', device='cuda', face_size=(224, 224))
    face_detector2 = FaceDetector(name='mobilenet', weight_path='Retinaface/weights/mobilenet.pth', device=conf.device)
    age_gender_detector = AgeGender(name='full', weight_path='AgeGender/weights/ShufflenetFull.pth', device='cuda')
    emotion_detector = EmotionDetector(name='densnet121', weight_path='FacialExpression/weights/densnet121.pth', device='cuda')

    if args.update:
        targets, names = update_facebank(conf, face_rec.model, face_detector2, tta=args.tta)
    else:
        targets, names = load_facebank(conf)


    frame = cv2.imread('Retinaface/demo_img/multiface.jpg')
    faces, boxes, scores, landmarks = face_detector.detect_align(frame)
    faces, boxes, scores, landmarks = face_detector2.detect_align(frame)

    if len(faces.shape) > 1:
        results, score = face_rec.infer(conf, faces, targets, args.tta)
        genders, ages = age_gender_detector.detect(faces)
        emotions, emo_probs = emotion_detector.detect_emotion(faces)

        for i, b in enumerate(boxes):
            cv2.putText(frame, f'{genders[i]},{ages[i]}', (int(b[0]), int(b[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, [0, 200, 0], 2)
            cv2.putText(frame, emotions[i], (int(b[0]), int(b[1]) + 140), cv2.FONT_HERSHEY_SIMPLEX, 1.1, [0, 200, 0], 2)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 3)
            special_draw(frame, b, landmarks[i], names[results[i] + 1], score[i])

        for p in landmarks:
            for i in range(5):
                cv2.circle(frame, (p[i][0], p[i][1]), 3, (0, 255, 0), -1)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()