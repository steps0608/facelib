#
import cv2
import argparse
from InsightFace.data.config import get_config
from InsightFace.models.Learner import face_learner
from InsightFace.utils import update_facebank, load_facebank, special_draw
from Retinaface.Retinaface import FaceDetector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.45, type=float)
    parser.add_argument("-u", "--update", default=True, help="whether perform update the facebank")
    parser.add_argument("-tta", "--tta", default=True, help="whether test time augmentation")
    parser.add_argument('-m', '--mobilenet', default=False, help="use mobilenet for backbone")
    args = parser.parse_args()

    conf = get_config(training=False)
    detector = FaceDetector(name='mobilenet', weight_path='../weights/mobilenet.pth', device=conf.device)
    conf.use_mobilfacenet = args.mobilenet
    face_rec = face_learner(conf, inference=True)
    face_rec.threshold = args.threshold
    face_rec.model.eval()

    if args.update:
        targets, names = update_facebank(conf, face_rec.model, detector, tta=args.tta)
    else:
        targets, names = load_facebank(conf)

    frame = cv2.imread('../Retinaface/demo_img/multiface.jpg')
    faces, boxes, scores, landmarks = detector.detect_align(frame)
    if len(faces.shape) > 1:
        results, score = face_rec.infer(conf, faces, targets, args.tta)
        for idx, bbox in enumerate(boxes):
            special_draw(frame, bbox, landmarks[idx], names[results[idx] + 1], score[idx])

    cv2.imshow('face Capture', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
