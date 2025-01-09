import json
from collections import defaultdict
import random
from PIL import Image
import numpy as np
import torchvision
import os

class cifar10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        if isinstance(self.transform, list):
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            return img1, img2, target
        else:
            image = self.transform(image)
            return image, target

class IMBALANCECIFAR10(cifar10):
    cls_num = 10

    def __init__(self, root, imb_factor=None, noise_ratio=None, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

        if train and imb_factor is not None:
            np.random.seed(rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
            self.gen_imbalanced_data(img_num_list)

        self.classnames = self.classes
        self.labels = self.targets
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        if train and noise_ratio is not None:
            self.get_noisy_data(self.cls_num, f"./output/cifar10_ir{1/imb_factor:.0f}_nr{noise_ratio*100:.0f}_noise_file.json", noise_ratio)
            self.cls_num_list = self.get_cls_num_list()

    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_noisy_data(self, cls_num, noise_file, noise_ratio):
        train_label = self.labels

        if os.path.exists(noise_file):
            noise_label = json.load(open(noise_file, "r"))
        else:  # inject noise
            noise_label = []
            num_train = len(self.targets)
            idx = list(range(num_train))
            random.shuffle(idx)
            cls_num_list = self.cls_num_list

            num_noise = int(noise_ratio * num_train)
            noise_idx = idx[:num_noise]

            p = np.array([cls_num_list for _ in range(cls_num)])
            for i in range(cls_num):
                p[i][i] = 0
            p = p / p.sum(axis=1, keepdims=True)
            for i in range(num_train):
                if i in noise_idx:
                    newlabel = np.random.choice(cls_num, p=p[train_label[i]])
                    assert newlabel != train_label[i]
                    noise_label.append(newlabel)
                else:
                    noise_label.append(train_label[i])

            noise_label = np.array(noise_label, dtype=np.int8).tolist()
            # label_dict['noisy_labels'] = noise_label
            print("save noisy labels to %s ..." % noise_file)
            json.dump(noise_label, open(noise_file, "w"))
        self.clean_targets = self.targets[:]
        self.targets = noise_label
        self.labels = self.targets

    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list


class CIFAR10(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=None, train=train, transform=transform)


class CIFAR10_IR10(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, train=train, transform=transform)

class CIFAR10_IR100_NR60(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.6, train=train, transform=transform)

class CIFAR10_IR100_NR50(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.5, train=train, transform=transform)

class CIFAR10_IR100_NR40(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.4, train=train, transform=transform)

class CIFAR10_IR100_NR30(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.3, train=train, transform=transform)

class CIFAR10_IR100_NR20(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.2, train=train, transform=transform)

class CIFAR10_IR100_NR10(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, noise_ratio=0.1, train=train, transform=transform)



class CIFAR10_IR10_NR60(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.6, train=train, transform=transform)

class CIFAR10_IR10_NR50(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.5, train=train, transform=transform)

class CIFAR10_IR10_NR40(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.4, train=train, transform=transform)

class CIFAR10_IR10_NR30(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.3, train=train, transform=transform)

class CIFAR10_IR10_NR20(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.2, train=train, transform=transform)

class CIFAR10_IR10_NR10(IMBALANCECIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, noise_ratio=0.1, train=train, transform=transform)