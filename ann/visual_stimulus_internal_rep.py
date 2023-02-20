# This needs python 3.10. From AlexNet-Pytorch project

import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

import tqdm
import glob
from PIL import Image
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
import models.cornet as cornet
import models.vonenet as vonenet

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


def main_for_alexnet() -> None:
    def build_model() -> nn.Module:
        # alexnet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)
        alexnet_model = alexnet(num_classes=config.model_num_classes)
        alexnet_model = alexnet_model.to(device=config.device, memory_format=torch.channels_last)
        return alexnet_model

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

class FeatureConcatHelper(object):
    def __init__(self):
        self.all_features = {}
    def store(self, feat, name, on_gpu=True):
        if not name in self.all_features.keys():
            self.all_features[name] = []
        if on_gpu:
            self.all_features[name].append(feat.to("cpu").detach().numpy())
        else:
            self.all_features[name].append(feat)
    
    def dump(self, path):
        os.makedirs(path, exist_ok=True)

        for key, feature_list in self.all_features.items():
            feature_filename = f"features_{key}.npy"
            feat_all = np.concatenate(feature_list, axis=0)
            print(key, feat_all.shape)
            np.save(os.path.join(path, feature_filename), feat_all)

def main_for_vonenet() -> None:
    model = vonenet.VOneNet(model_arch="resnet50")
    print(model)

    model = model.to(config.device)

    test_prefecther = load_dataset()
    test_prefecther.reset()
    batch_data = test_prefecther.next()

    fch = FeatureConcatHelper()
    while batch_data is not None:
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)
        print("image shape", images.shape)
        x = images
        x = model.vone_block(x)
        fch.store(x, "vone_block")
        x = model.bottleneck(x)
        fch.store(x, "bottleneck")
        x = model.model(x)
        fch.store(x, "model")

        batch_data = test_prefecther.next()

    fch.dump("../internal_feature/vonenet_resnet50/block1_1/")


def main_for_CORNet(ngpus = 0, model_type = None, times = 5) -> None:
    """
    From CORNet/run.py
    - ngpus :
    - model_type: (r, rt, s, z, None)
    - time: number of time steps to run the model (only R model)
    """
    def get_cornet(model_type, times = 0, pretrained=False):
        map_location = None if ngpus > 0 else 'cpu'
        model = getattr(cornet, f'cornet_{model_type.lower()}')
        if model_type.lower() == 'r':
            model = model(pretrained=pretrained, map_location=map_location, times=times)
        else:
            model = model(pretrained=pretrained, map_location=map_location)

        if ngpus == 0:
            model = model.module  # remove DataParallel
        if ngpus > 0:
            model = model.cuda()
        return model

    model = get_cornet(model_type = model_type, times = times, pretrained=True)

    model.eval()

    try:
        m = model.module
    except:
        m = model

    model = model.to(device=config.device)

    test_prefecther = load_dataset()
    test_prefecther.reset()
    batch_data = test_prefecther.next()

    fch = FeatureConcatHelper()
    while batch_data is not None:
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)

        x = images
        
        if 'r' in model_type:
            x, featall = model.forward_and_extract(x)
            for key, feat in featall.items():
                # print(key, feat.shape)
                fch.store(feat, key, on_gpu=False)
        else:
            x, v1_internal = model.V1.forward_and_extract(x)
            # print(v1_internal['conv'])
            for key, feat in v1_internal.items():
                fch.store(feat, f"v1_{key}", on_gpu=False)
            del v1_internal
            print("V1", x.shape)

            x, v2_internal = model.V2.forward_and_extract(x)            
            for key, feat in v2_internal.items():
                fch.store(feat, f"v2_{key}", on_gpu=False)
            del v2_internal
            print("V2", x.shape)

            x, v4_internal = model.V4.forward_and_extract(x)            
            for key, feat in v4_internal.items():
                fch.store(feat, f"v4_{key}", on_gpu=False)
            del v4_internal
            print("V4", x.shape)

            x, it_internal = model.IT.forward_and_extract(x)
            for key, feat in it_internal.items():
                fch.store(feat, f"it_{key}", on_gpu=False)
            del it_internal
            print("IT", x.shape)

            output = model.decoder(x)

            # output_ref = model(images)
            # assert(torch.equal(output_ref, output))

        batch_data = test_prefecther.next()

    fch.dump(f"../internal_feature/CORNet_{model_type}/block1_1/")


if __name__ == "__main__":
    # main_for_alexnet()
    # main_for_vgg()
    # main_for_resnet()
    # main_for_CORNet(model_type='s')
    # main_for_CORNet(model_type='z')
    # main_for_CORNet(model_type='r')
    main_for_CORNet(model_type='rt')
    # main_for_vonenet()

