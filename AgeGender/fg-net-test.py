import os
import re
import cv2
import glob
import shutil

from Retinaface.Retinaface import FaceDetector
from AgeGender.Detector import AgeGender
from AgeGender.models import model
import cv2


def test_single(img):
    if type(img) == str:
        img = cv2.imread(img)

    face_detector = FaceDetector(name='mobilenet', weight_path='../Retinaface/weights/mobilenet.pth', device='cuda')
    age_gender_detector = AgeGender(name='full', weight_path='./weights/ShufflenetFull.pth', device='cuda')

    faces, boxes, scores, landmarks = face_detector.detect_align(img)

    genders = None
    ages = None
    if len(faces.shape) > 1:
        genders, ages = age_gender_detector.detect(faces)
        for i, b in enumerate(boxes):
            cv2.putText(img, f'{genders[i]},{ages[i]}', (int(b[0]), int(b[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        [0, 200, 0], 3)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 3)


    return img, genders, ages

def test_batch(path, name_contain_label = False, result_path = None):
    age_loss_total = 0
    age_acc_count = 0
    if path[-1] != '/':
        print("[WARNING] PARAM: path NOT ENDS WITH '/'!")
        path += '/'

        # check param: result_path
    if result_path is None:
        result_path = path + "all_results/"
    false_results = result_path + "false_results/"
    all_results = result_path + "all_results/"

    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if os.path.exists(result_path):
        shutil.rmtree(result_path)
        os.mkdir(result_path)

    if not os.path.exists(all_results):
        os.mkdir(all_results)

    if name_contain_label:
        if not os.path.exists(false_results):
            os.mkdir(false_results)
        if os.path.exists(false_results):
            shutil.rmtree(false_results)
            os.mkdir(false_results)

    file_list = []

    for img_path in glob.glob(path + "*"):
        img_name = img_path[len(path):]
        formatt = re.findall("[^.]*.([^.]*)", img_name)[0]
        if not formatt: continue
        formatt = formatt.lower()
        if not formatt in ['png', 'jpg', 'jpeg']: continue

        print("[evaluate] evaluating {}".format(img_name))
        img = cv2.imread(img_path)
        img, gen_pred, age_pred = test_single(img)

        cv2.imwrite(all_results + img_name, img)

        if name_contain_label:
            try:
                (age, gender) = re.findall(r'([^_]*)_([^_]*)_*', img_name)[0]
                age, gender = int(age), int(gender)
                if abs(age - age_pred) >= 5:
                    cv2.imwrite(false_results + img_name, img)
            except:
                print("Error while extracting labels from {}".format(img_name))

        try:
            age_pred2 = age_pred[0]
            img_name_a = int(img_name[:2])
            age_loss = abs(age_pred2 - img_name_a)
            age_loss_total += abs(age_loss)
            if age_loss < 5:
                age_acc_count += 1
        except:
            file_list.append(img_name)
            continue

    file_total = len(glob.glob(path + '*.jpg'))
    print('No face detected: ' + str(file_list))
    age_mae = age_loss_total / file_total
    age_acc = age_acc_count / file_total * 100
    print("[evaluate] Done!")
    print("FG-NET Age MAE: {:.2f}".format(age_mae))
    print("FG-NET Age ACC: {:.2f}%".format(age_acc))


if __name__ == "__main__":
    path = 'fg-net/'
    test_batch(path, name_contain_label=False)

    pass