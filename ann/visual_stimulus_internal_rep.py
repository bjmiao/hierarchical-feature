# This needs python 3.10. From AlexNet-Pytorch project

import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
# sys.path.insert(0, "/home/bjmiao/Documents/alexnet/AlexNet-PyTorch/")
# sys.path.insert(0, "/home/bjmiao/Documents/alexnet/AlexNet-PyTorch/")
# sys.path.insert(0, "utils")

import utils
from utils import config
from utils.dataset import CUDAPrefetcher, ImageDataset
from utils.utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter

from models.AlexNet import *
from models.ResNet import *
from models.vgg import *


def load_dataset() -> CUDAPrefetcher:
    test_dataset = ImageDataset("/home/bjmiao/Documents/alexnet/data/block1_1/", config.image_size, "Test")
    with open("../internal_feature/stimulus.txt", "w") as f:
        print(str(test_dataset.image_file_paths), file=f)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=100,
                                #  batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)
    return test_prefetcher

# def visualize_for_vgg -> None

def main_for_vgg() -> None:
    os.makedirs("../internal_feature/vgg/", exist_ok=True)
    model = vgg11(
        pretrained = True,
        model_root = "/home/bjmiao/Documents/hierarchical-feature/pretrained/online"
    )
    model = model.to(config.device)
    test_prefecther = load_dataset()
    print(model.features)
    num_layers = len(model.features)
    features_raw_images_all = []
    features_result_each_layer = [[] for _ in range(num_layers)]

    test_prefecther.reset()
    batch_data = test_prefecther.next()
    while batch_data is not None:
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)
        print(images.shape)
        x = images
        features_raw_images_all.append(images.to("cpu").detach().numpy())
        for layer_idx in range(num_layers):
            x = model.features[layer_idx](x)
            print(layer_idx, x.shape)
            features_result_each_layer[layer_idx].append(x.to("cpu").detach().numpy())
        # x = x.view(x.size(0), -1)
        # x = model.classifier(x)
        features_final = x.to("cpu")
        features_ref = model.features(images).to("cpu")
        assert torch.equal(features_ref, features_final)
        batch_data = test_prefecther.next()

    for layer_idx in range(num_layers):
        print(layer_idx)
        for f in features_result_each_layer[layer_idx]:
            print("\t", f.shape)

    features_total = {
        f"features_result_layer{i}":np.concatenate(features_result_each_layer[i], axis=0) for i in range(len(features_result_each_layer))
        # f"features_result_layer{i}":np.concatenate(features_result_each_layer[i], axis=0) for i in range(1)
    }
    features_total["features_raw_images"] = np.concatenate(features_raw_images_all, axis=0)
    print(features_total.keys())

    base_path = "../internal_feature/VGG11/"
    os.makedirs(base_path, exist_ok=True)
    for name, f_arr in features_total.items():
        print(f_arr.shape)
        np.save(os.path.join(base_path, f"{name}.npy"), f_arr)

def main_for_resnet() -> None:
    os.makedirs("../internal_feature/resnet/", exist_ok=True)

    model = resnet18(
        pretrained = True,
        model_root = "/home/bjmiao/Documents/hierarchical-feature/pretrained/online"
    )
    model = model.to(config.device)
    test_prefecther = load_dataset()
    features_images_all = []
    features_result_layer0_all = []
    features_result_layer1_all = []
    features_result_layer2_all = []
    features_result_layer3_all = []
    features_result_layer4_all = []

    test_prefecther.reset()
    batch_data = test_prefecther.next()
    while batch_data is not None:
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)
        print(images.shape)
        
        x = images
        x = model.group1(x)
        features_result_layer0 = x.to("cpu")
        x = model.layer1(x)
        features_result_layer1 = x.to("cpu")
        x = model.layer2(x)
        features_result_layer2 = x.to("cpu")
        x = model.layer3(x)
        features_result_layer3 = x.to("cpu")
        x = model.layer4(x)
        features_result_layer4 = x.to("cpu")
        x = model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = model.group2(x)
        features_final = x.to("cpu")

        features_ref = model(images).to("cpu")
        assert torch.equal(features_ref, features_final)
        print(f"{features_result_layer0.shape=}")
        print(f"{features_result_layer1.shape=}")
        print(f"{features_result_layer2.shape=}")
        print(f"{features_result_layer3.shape=}")
        print(f"{features_result_layer4.shape=}")
        features_images_all.append(images.to("cpu").detach().numpy())
        features_result_layer0_all.append(features_result_layer0.detach().numpy())
        features_result_layer1_all.append(features_result_layer1.detach().numpy())
        features_result_layer2_all.append(features_result_layer2.detach().numpy())
        features_result_layer3_all.append(features_result_layer3.detach().numpy())
        features_result_layer4_all.append(features_result_layer4.detach().numpy())
        batch_data = test_prefecther.next()

    features_total = {
        "features_raw_images":np.concatenate(features_images_all, axis=0),
        "features_result_layer0":np.concatenate(features_result_layer0_all, axis=0),
        "features_result_layer1":np.concatenate(features_result_layer1_all, axis=0),
        "features_result_layer2":np.concatenate(features_result_layer2_all, axis=0),
        "features_result_layer3":np.concatenate(features_result_layer3_all, axis=0),
        "features_result_layer4":np.concatenate(features_result_layer4_all, axis=0),
    }

    for name, f_arr in features_total.items():
        print(f_arr.shape)
        np.save(f"../internal_feature/resnet/block1_1/{name}.npy", f_arr)



def build_model() -> nn.Module:
    # alexnet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)
    alexnet_model = alexnet(num_classes=config.model_num_classes)
    alexnet_model = alexnet_model.to(device=config.device, memory_format=torch.channels_last)
    return alexnet_model

def main_for_alexnet() -> None:
    # Initialize the model
    model = build_model()
    test_prefecther = load_dataset()
    batch_data = test_prefecther.next()
    images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    target = batch_data["target"].to(device=config.device, non_blocking=True)


    print(model)
    features_result_layer1 = model.features[:3](images)
    features_result_layer2 = model.features[3:6](features_result_layer1)
    features_result_layer3 = model.features[6:8](features_result_layer2)
    features_result_layer4 = model.features[8:10](features_result_layer3)
    features_final = model.features[10:](features_result_layer4)

    features_ref = model.features(images)
    assert torch.equal(features_final, features_ref)
    features_total = {
        "feature_raw_images":images,
        "features_result_layer1":features_result_layer1,
        "features_result_layer2":features_result_layer2,
        "features_result_layer3":features_result_layer3,
        "features_result_layer4":features_result_layer4,
        "features_result_layer5":features_final,
    }

    for name, f in features_total.items():
        print(f.shape)
        f_cpu = f.to('cpu')
        f_arr = f_cpu.detach().numpy()
        os.makedirs("../internal_feature/alexnet/block1_1", exist_ok=True)
        np.save(f"../internal_feature/alexnet/block1_1/{name}.npy", f_arr)

if __name__ == "__main__":
    # main_for_alexnet()
    # main_for_vgg()
    main_for_resnet()
