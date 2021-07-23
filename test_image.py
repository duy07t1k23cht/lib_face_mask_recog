import os
import shutil
import cv2
from PIL import Image
import glob
from tqdm import tqdm

import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import random

from models.inception_resnet_v1 import InceptionResnetV1
import models.custom_model as custom_model


# FACENET_STATEDICT_PATH = "weights/29062021_33ep_0.0016loss.pth"
FACENET_STATEDICT_PATH = "weights/facenet_vnceleb_nomask.pth"
DEEPLAB_STATEDICT_PATH = "weights/checkpoint_2806_ep140_loss0.0139_acc0.9858.pth"

FEATURES_PATH = "./embeding_features/SkyMap/final"
# FEATURES_PATH = "./embeding_features/VN_Celeb"
train_features = np.load(os.path.join(FEATURES_PATH, "train_features.npy"))
train_labels = np.load(os.path.join(FEATURES_PATH, "train_labels.npy"))

print(train_features.shape)
print(train_labels.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DeepLab transform
transforms_image =  T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# Init facenet model
facenet_state_dict = torch.load(FACENET_STATEDICT_PATH, map_location=device)
facenet_model = InceptionResnetV1()
if torch.cuda.is_available():
    facenet_model.cuda()
facenet_model.load_state_dict(facenet_state_dict)
facenet_model.eval()

# Init deeplab model
dl_state_dict = torch.load(DEEPLAB_STATEDICT_PATH, map_location=device)
model, input_size = custom_model.initialize_model(2, keep_feature_extract=True, use_pretrained=False)
model = model.to(device)
model.load_state_dict(dl_state_dict)
model.eval()


def resize_image(image, size=224):
    new_height = image.shape[0]
    new_width = image.shape[1]

    if image.shape[0] != size:
        new_height = size
        new_width = int(image.shape[1] * (new_height / image.shape[0]))

    if image.shape[1] != size:
        new_width = size
        new_height = int(image.shape[0] * (new_width / image.shape[1]))

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def get_mask(image, debug=False):
    # image_np = np.asarray(image)
    # width = int(image_np.shape[1] * 0.3)
    # height = int(image_np.shape[0] * 0.3)
    # dim = (width, height)
    # image_np = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)
    image_np = resize_image(image)

    image = Image.fromarray(image_np)
    image = transforms_image(image)
    image = image.unsqueeze(0)

    image = image.to(device)

    outputs = model(image)["out"]

    _, preds = torch.max(outputs, 1)

    preds = preds.to("cpu")

    preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

    preds_np = cv2.cvtColor(preds_np, cv2.COLOR_GRAY2BGR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if debug:
        cv2.imshow("preds_np_color", preds_np)
        cv2.imshow("image_np", image_np)

        cv2.waitKey()
        cv2.destroyAllWindows()
    
    return preds_np


def normalize_mask(image, debug=False):
    mask_image = get_mask(image)
    # print(np.sum(mask_image[:,:,0]) / (mask_image.shape[0] * mask_image.shape[1]))
    if np.sum(mask_image[:,:,0]) < int(0.25 * mask_image.shape[0] * mask_image.shape[1]):
        return image, None 

    mask_image[np.where(mask_image > 0)] = 255
    mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]))
    
    mask = np.where(mask_image > 0)
    raw_image = image.copy()
    image[mask] = 255
    mask_image[mask] = 255
    top_height = int(image.shape[0] * 0.3)
    image[:top_height,:,:] = raw_image[:top_height,:,:]
    mask_image[:top_height,:,:] = 0

    if debug:
        cv2.imshow("image", image)
        cv2.imshow("mask_image", mask_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return image, mask_image


def get_features(image):
    # You may need to convert the color.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image_size = 160
    transform = T.Compose([
        T.Resize([image_size, image_size]), 
        T.ToTensor(),
        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        image = transform(image)
        image = image.unsqueeze(0).to(device)
        output = facenet_model(image)
        
    return output.cpu().numpy()[0]


def get_id(image):
    # image = normalize_mask(image)[0]
    features = get_features(image)

    min_distance = None
    min_idx = None
    for idx, train_feature in enumerate(train_features):
        distance = np.linalg.norm(train_feature - features)
        if min_distance is None:
            min_distance = distance
            min_idx = idx
        else:
            if distance < min_distance:
                min_distance = distance
                min_idx = idx

    # print(min_distance)
    return train_labels[min_idx], min_distance


def calc_accuracy(dataset_path):
    count_total = 0
    count_correct = 0

    all_folders = glob.glob(os.path.join(dataset_path, "*"))
    for folder in tqdm(all_folders, desc="Testing {}".format(os.path.basename(dataset_path))):
        all_images = glob.glob(os.path.join(folder, "*"))
        # if len(all_images) > 10:
            # all_images = random.sample(all_images, 10)
        true_label = int(os.path.basename(folder))
        for image_path in all_images:
            image = cv2.imread(image_path)
            predicted = get_id(image)[0]
            if predicted == true_label:
                count_correct += 1
            count_total += 1
    
    return count_correct, count_total, count_correct / count_total


def visualize(image_path_list):
    all_image = None
    masked_image = None
    mask = None
    blank_image = np.ones((2, 224, 3), np.uint8)
    blank_image[:,:,:] = 255
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        image = resize_image(image, 224)

        if all_image is None:
            all_image = image.copy()
        else:
            all_image = cv2.vconcat([all_image, image])

        norm_mask, _mask = normalize_mask(image)

        if masked_image is None:
            masked_image = norm_mask.copy()
        else:
            masked_image = cv2.vconcat([masked_image, norm_mask])

        if mask is None:
            mask = _mask.copy()
        else:
            mask = cv2.vconcat([mask, _mask])

        print(all_image.shape)
        print(blank_image.shape)
        all_image = cv2.vconcat([all_image, blank_image])
        mask = cv2.vconcat([mask, blank_image])
        masked_image = cv2.vconcat([masked_image, blank_image])

    # all_image = image_list[0]
    # for image in image_list[1:]:
    #     all_image = cv2.vconcat(all_image, image)
    # cv2.imshow("", all_image)
    # cv2.waitKey()
    cv2.imshow("All image", all_image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Masked", masked_image)
    cv2.imwrite("test_mask_1.jpg", all_image)
    cv2.imwrite("test_mask_2.jpg", mask)
    cv2.imwrite("test_mask_3.jpg", masked_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    # for image_path in glob.glob("/media/duy/Personal/DATN/dataset/data_sky/Sky_test_2906/22/*"):
    #     print("=" * 50)
    #     image_path = "/media/duy/Personal/DATN/BaoCao/images/the_mask_detect.jpg"
    #     image = cv2.imread(image_path)
    #     raw_image = image.copy()
    #     cv2.imshow("Image", image)
    #     image_mask = normalize_mask(image)[0]
    #     cv2.imshow("Mask", image_mask)
    #     cv2.imwrite("/media/duy/Personal/DATN/BaoCao/images/the_mask_detect_norm.jpg", image_mask)
    #     true_label = image_path.split("/")[-2]
    #     predicted = get_id(raw_image)
    #     print(true_label)
    #     print(predicted)
    #     key = cv2.waitKey()
    #     if key == 27:
    #         break
    # image = cv2.imread("test_images/3_72_face_mask (copy).jpg")
    # print(get_id(image))
    # print(calc_accuracy("/media/duy/Personal/DATN/dataset/VN_Celeb_test_2906"))
    print(calc_accuracy("/media/duy/Personal/DATN/dataset/data_sky/Sky_test_2906 (copy)"))

    # image_path = "mask1.jpg"
    # image = cv2.imread(image_path)
    # image_mask = normalize_mask(image)[0]
    # cv2.imshow("Mask", image_mask)
    # cv2.waitKey()


if __name__ == "__main__":
    main()
    # for image_path in ["face_mask1.jpg", "face_mask2.jpg", "face_mask3.jpg", "/media/duy/Personal/DATN/DeepLabV3FineTuning/test_images/mask2 (copy).jpg"]:
    #     image = cv2.imread(image_path)
    #     normed, mask = normalize_mask(image)
        
    #     cv2.imshow("Raw image", image)
    #     cv2.imshow("Normed", normed)
    #     cv2.imshow("Mask", mask)
    #     cv2.waitKey()

    # # cv2.destroyAllWindows()
    # image_path_list = [
    #     # "test_images/mask1.jpg",
    #     # "test_images/mask2.jpg",
    #     # "test_images/mask3.jpg",
    #     # "/media/duy/Mason/DATN/diem_danh/face_mask1.jpg",
    #     # "/media/duy/Mason/DATN/diem_danh/face_mask2.jpg",
    #     # "/media/duy/Mason/DATN/diem_danh/face_mask3.jpg",
    #     "face_mask1.jpg",
    #     "face_mask2.jpg",
    #     "face_mask3.jpg"
    #     # "/media/duy/Personal/DATN/dataset/image_seg_dataset_sky_224/val/Images/solid_81_81_42_face.png"
    #     # "test_images/mask2 (copy).jpg"
    # ]
    # visualize(image_path_list)
    # # pass