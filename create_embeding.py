import os
import shutil
import cv2
from PIL import Image

import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import random
from facenet_pytorch import fixed_image_standardization

from models.inception_resnet_v1 import InceptionResnetV1
import glob
from tqdm import tqdm


def build_transforms(args, train, PIXEL_MEAN=[0.5, 0.5, 0.5], PIXEL_STD=[0.5, 0.5, 0.5]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    if train:
        transform = T.Compose(
            [
                T.Resize([args.image_size, args.image_size]),
                T.ToTensor(),
                # normalize_transform,
            ]
        )
    else:
        transform = T.Compose([T.Resize([args.image_size, args.image_size]), T.ToTensor()])
    return transform


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
        output = model(image)
        
    return output.cpu().numpy()[0]


def get_features_train(train_folder_path):
    train_features = []
    train_labels = []

    for image_path in tqdm(glob.glob(os.path.join(train_folder_path, "*", "*")), desc="Getting train features"):
        label = int(image_path.split("/")[-2])
        image = cv2.imread(image_path)

        features = get_features(image)
        
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels, dtype=np.uint8)

    print(train_features.shape)
    print(train_labels.shape)

    np.save("./embeding_features/SkyMap/final/train_features.npy", train_features)
    np.save("./embeding_features/SkyMap/final/train_labels.npy", train_labels)


def main():
    global device   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Change this
    state_dict_path = "weights/facenet_sky_nonorm.pth"

    global state_dict
    state_dict = torch.load(state_dict_path, map_location=device)

    global model
    model = InceptionResnetV1()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(state_dict)
    model.eval()
    
    # get_features_train("/media/duy/Personal/DATN/dataset/VN_Celeb_100_Aligned_2706",)
    get_features_train("/media/duy/Personal/DATN/dataset/data_sky/Sky_2906",)


if __name__ == "__main__":
    main()
    # image_size = 448
    # transform = T.Compose([
    #     T.Resize([image_size, image_size]), 
    #     T.ToTensor(),
    #     # fixed_image_standardization
    #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    # with torch.no_grad():
    #     image = Image.open("test_images/3_72_face_mask (copy).jpg")
    #     t_image = transform(image)
    #     image.show()
    #     plt.imshow(t_image.permute(1, 2, 0))
    #     plt.show()