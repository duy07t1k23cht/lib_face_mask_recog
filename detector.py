import sys

from numpy.lib.type_check import imag
sys.path.append(".")

import cv2

from detection_model.retinaface import RetinaNetDetector
import utils
import imgaug.augmenters as iaa
import random
import os
import glob
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
import mediapipe
import random
import numpy as np

from components.landmark_detection import detect_landmarks
from components.convex_hull import find_convex_hull
from components.delaunay_triangulation import find_delauney_triangulation
from components.affine_transformation import apply_affine_transformation
from components.clone_mask import merge_mask_with_image


mpDraw = mediapipe.solutions.drawing_utils
mpFaceMesh = mediapipe.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()


def test_detector(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets, landmarks = detector.predict(image)
    if len(landmarks)!=0:
        bound = utils.get_all_faces(dets, 0.9)
    else:
        bound = []  

    if len(bound) == 0:
        print("No face detected")
        return

    for bbox in bound:
            left, top, right, bottom = bbox[:4]
            face_width, face_height = right-left, bottom- top 
            padding = [0 - face_width//3, 0 - face_height//2, face_width//3, face_height//4]
            padding = [0, 0, 0, 0]
            faces = utils.crop(image, bbox, padding)
            # cv2.rectangle(image, (left, top), (right, bottom), color=(0, 255, 0), thickness=3)
            # cv2.putText(image, "id: 01", (left, top), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)

    cv2.imshow("Org image", image)
    cv2.imwrite("Final_resulg.jpg", faces)
    cv2.imshow("Face detected", faces)

    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_face(image_path):
    image = cv2.imread(image_path)
    if image.shape[0] > 3000 or image.shape[1] > 3000:
        image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))
    dets, landmarks = detector.predict(image)
    if len(landmarks) !=0 :
        bound = utils.get_all_faces(dets, 0.9)
    else:
        bound = []  

    if len(bound) == 0:
        print("No face detected, image {}".format(image_path))
        return None

    for bbox in bound:
            pad = random.randint(0, 10)
            pad = 0
            padding = [0 - pad, 0 - pad, pad, pad]
            # padding = [0, 0, 0, 0]
            faces = utils.crop(image, bbox, padding)

    return faces


def augment(image):
    if image is None:
        return None

    seq_valid = iaa.Sequential(
        [
            iaa.Sometimes(0.3, iaa.OneOf([
                iaa.GaussianBlur((0.5, 1.5)), 
                iaa.AverageBlur(k=(1, 3)), 
                iaa.MedianBlur(k=(1, 5)),
            ]),),
            iaa.Sometimes(0.3, iaa.ChangeColorTemperature((5000, 10000))),
            iaa.Sometimes(0.3, iaa.AddToBrightness((-20, 30))),
            iaa.Sometimes(0.3, iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-20, 20)))),
            iaa.Sometimes(0.3, iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-20, 20)))),
            iaa.Sometimes(0.3, iaa.AveragePooling(2)),
            iaa.Sometimes(0.3, iaa.SigmoidContrast(gain=random.choice([3, 4]), cutoff=(0.5, 0.5))),
            iaa.Sometimes(0.3, iaa.JpegCompression(compression=(10, 30))),
            iaa.Sometimes(0.3, iaa.AverageBlur(k=random.randint(1, 3)))
        ]
    )
    aug_image = seq_valid.augment(image=image)

    return aug_image


def save_augment(image_path, output_path, prefix_name, number=1):
    image_name = os.path.basename(image_path)
    for i in range(number):
        save_name = "{}_{}_{}".format(prefix_name, i, image_name)
        
        raw_image = cv2.imread(image_path)
        aug_image = augment(raw_image)
        if aug_image is None:
            print("Fail to augment image {}".format(image_path))
            continue

        cv2.imwrite(
            os.path.join(output_path, save_name),
            augment(aug_image)
        )


def create_data(index):
    image_path = all_data[index]
    folder_name = os.path.basename(Path(image_path).parent)
    if int(folder_name) > 100:
        return
    if os.path.basename(image_path).find("mask") != -1:
        mode = "mask"
    else:
        mode = "no_mask"
    output_folder = os.path.join(output_path, mode, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    save_augment(image_path, output_folder, folder_name, 50)


def detect_landmarks_468(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    ih, iw, _ = img.shape
    output = []
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS)
            for lm in faceLms.landmark:
                x, y = int(lm.x * iw), int(lm.y * ih)
                output.append([x, y])

    return output


def crop_eye():
    all_no_mask_img = glob.glob("/home/duy/Documents/DATN/diem_danh/data_test/no_mask/*/*")
    for image_path in tqdm(all_no_mask_img):
        image_name = os.path.basename(image_path)
        image_dir = os.path.basename(os.path.dirname(image_path))
        image = cv2.imread(image_path)
        # image = detect_face(image_path)
        landmarks = detect_landmarks_468(image)
        cut_y = landmarks[51][1] + random.randint(-5, 5)
        cropped = image[random.randint(0, 10):cut_y, :]
        # print("all lm: ", len(landmarks))
        # for idx, lm in enumerate(landmarks):
        #     # cv2.circle(image, (lm[0], lm[1]), radius=1, color=(0, 255, 0), thickness=1)
        #     if idx > 50 and idx < 100:
        #         cv2.putText(image, str(idx), (lm[0], lm[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.25, color=(0, 255, 0))

        # cv2.imshow("Org", image)
        # cv2.imshow("Cropped", cropped)
        # print(image_dir)
        os.makedirs(os.path.join(output_path, "mask_crop", image_dir), exist_ok=True)
        cv2.imwrite(os.path.join(output_path, "mask_crop", image_dir, "crop_{}".format(image_name)), cropped)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            return
        else:
            continue


def crop_face(dataset_path):
    for image_path in tqdm(glob.glob(os.path.join(dataset_path, "*", "*"))):
        if not (image_path.endswith(".jpg") or image_path.endswith(".png")):
            continue 
        face = detect_face(image_path)
        if face is not None:
            cv2.imwrite(image_path, face)


def main():
    # global data_path
    # data_path = "/home/duy/Documents/DATN/VN-celeb"

    # global all_data
    # all_data = glob.glob(os.path.join(data_path, "*", "*.png"))
    # # print(type(all_data))

    global detector
    detector = RetinaNetDetector("/media/duy/Duy/Techainer2021/liveness_face/face_ekyc/weights/detector/mobilenet0.25_Final.pth")

    # crop_face("/media/duy/Personal/DATN/dataset/data_sky/Sky_2706")
    # image_path = "/media/duy/Personal/DATN/BaoCao/images/the_mask.jpg"
    # face = detect_face(image_path)
    # cv2.imwrite("/media/duy/Personal/DATN/BaoCao/images/the_mask_detect.jpg", face)
    # test_detector("/media/duy/Personal/DATN/BaoCao/images/special_mask.jpg")
    face = detect_face("mask3.jpg")
    cv2.imwrite("face_mask3.jpg", face)
    # face = detect_face("mask1.jpg")
    # face = detect_face("mask1.jpg")

    # global output_path
    # output_path = "/home/duy/Documents/DATN/VN-celeb-train"

    # total = len(all_data)
    # # print(total)

    # os.makedirs(output_path, exist_ok=True)

    # crop_eye()

    # Gen data
    # for i in tqdm(range(len(all_data))):
    #     try:
    #         create_data(i)
    #     except Exception as e:
    #         print(e)
    #         print(all_data[i])

    # mp.set_start_method('spawn')
    # pool = mp.Pool(8)
    # _ = list(tqdm(
    #     pool.imap(create_data, range(total)), 
    #     total=total, 
    #     desc="Generating data"
    # ))
    # pool.terminate() 

    # test_detector("/home/duy/Documents/DATN/xxx/data-20210512T025645Z-001/data/2/face_mask.jpg")
    # detected = detect_face("/home/duy/Documents/DATN/xxx/data-20210512T025645Z-001/data/2/face_mask.jpg")
    # augment(detected)

    # create_data(0)



if __name__ == "__main__":
    main()
    