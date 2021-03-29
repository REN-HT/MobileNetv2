import os
from PIL import Image
import numpy as np
from torch.utils import data
from torchvision import transforms as T


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True):
        # 得到每个样本路径
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)

        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        if train:
            self.imgs = imgs[:int(0.7 * imgs_num)]  # 训练集前70%为训练样本
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]  # 训练集后30%为验证样本

        if transforms is None:
            normalize = T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])

            if not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # dog-->1，cat-->0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        img = Image.open(img_path)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

