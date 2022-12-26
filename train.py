import os
from glob import glob
import itertools

import cv2
import numpy as np

from torchvision.models import efficientnet_b0
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# https://pystyle.info/pytorch-train-classification-problem-using-a-pretrained-model/
def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    else:
        return torch.device("cpu")

def replace_out_features(model, out_features=3):
    """replace output layer

    efficientnet_b0 default settings
    print(model)
    ...
    (classifier): Sequential(
        (0): Dropout(p=0.2, inplace=True)
        (1): Linear(in_features=1280, out_features=1000, bias=True)
    )

    Args:
        model (_type_): _description_
        out_features (int, optional): _description_. Defaults to 3.
    """
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features=1280, out_features=out_features, bias=True),
        )
    return model

class ImageFolder(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = torch.Tensor(y)
        self.transform = transform
        
    def __getitem__(self, idx):
        return self.transform(self.x[idx]), self.y[idx]
    
    def __len__(self):
        return len(self.x)

def main():
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    }


    default_model = efficientnet_b0(pretrained=True)
    print(default_model)
    model = replace_out_features(default_model, out_features=3)
    print(model)

    dataset_dir = "data/Sign-Language-Digits-Dataset/Dataset"
    rock_paths = glob(os.path.join(dataset_dir, "0", "*"))
    scissors_paths = glob(os.path.join(dataset_dir, "2", "*"))
    paper_paths = glob(os.path.join(dataset_dir, "5", "*"))

    rock_imgs = np.array([cv2.imread(path) for path in rock_paths])
    scissors_imgs = np.array([cv2.imread(path) for path in scissors_paths])
    paper_imgs = np.array([cv2.imread(path) for path in paper_paths])

    print(rock_imgs.shape)
    print(scissors_imgs.shape)
    print(paper_imgs.shape)

    rock_indexs = [0 for _ in range(len(rock_imgs))]
    scissors_indexs = [1 for _ in range(len(scissors_imgs))]
    paper_indexs = [2 for _ in range(len(paper_imgs))]
    indexs = list(itertools.chain(rock_indexs, scissors_indexs, paper_indexs))
    print(indexs)
    imgs = np.concatenate([rock_imgs, scissors_imgs, paper_imgs])
    print(imgs.shape)

    indexs = np.eye(3)[indexs]
    print(indexs)

    dataset = ImageFolder(imgs, indexs, data_transforms['train'])
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    device = get_device()
    # モデルを計算するデバイスに転送する。
    model = model.to(device)
    model.train()
    # 損失関数を作成する。
    criterion = nn.CrossEntropyLoss()
    # 最適化手法を選択する。
    optimizer = optim.Adam(model.parameters())
    epochs = 10

    for epoch in range(epochs):
        print("epoch: ",epoch+1)
        total_loss = 0
        for x, y in dataloader:
            # データ及びラベルを計算を実行するデバイスに転送する。
            x, y = x.to(device), y.to(device)
            # 学習時は勾配を計算するため、set_grad_enabled(True) で中間層の出力を記録するように設定する。
            with torch.set_grad_enabled(True):
                # 順伝搬を行う。
                outputs = model(x)
                # 確率の最も高いクラスを予測ラベルとする。
                preds = outputs.argmax(dim=1)
                # 損失関数の値を計算する。
                loss = criterion(outputs, y)
                print(f"{float(loss)}")
                # 逆伝搬を行う。
                optimizer.zero_grad()
                loss.backward()
                # パラメータを更新する。
                optimizer.step()
            total_loss += float(loss)
        print(f"\nloss: {total_loss}")

    # モデルを保存する。
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()