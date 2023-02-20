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
import models.ResNet as ResNet
import models.vgg as vgg
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

    fch = FeatureConcatHelper()

    while batch_data is not None:
        images = batch_data["image"].to(device=config.device, non_blocking=True)
        # images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)

        x, features = model.forward_and_extract(images)
        for key, feat in features.items():
            fch.store(feat, key, on_gpu = False)

        batch_data = test_prefecther.next()

    fch.dump(f"../internal_feature/AlexNet/block1_1")

def main_for_vgg(depth=11) -> None:
    '''size: 11/13/16/19'''
    model_name = f"vgg{depth}"
    model = vgg.__dict__[model_name](
        pretrained = True,
        model_root = "/home/bjmiao/Documents/hierarchical-feature/pretrained/online"
    )
    model = model.to(config.device)
    test_prefecther = load_dataset()
    test_prefecther.reset()
    batch_data = test_prefecther.next()

    fch = FeatureConcatHelper()

    while batch_data is not None:
        images = batch_data["image"].to(device=config.device, non_blocking=True)
        # images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)

        x, features = model.forward_and_extract(images)
        for key, feat in features.items():
            fch.store(feat, key, on_gpu = False)
        # features_final = x.to("cpu")

        # features_ref1 = model(images).to("cpu")
        # features_ref2 = model(images).to("cpu")
        # assert torch.equal(features_ref1, features_ref2) # assert will fail because classifier includes randomness in dropout
        # assert torch.equal(features_ref1, features_final)

        batch_data = test_prefecther.next()
    fch.dump(f"../internal_feature/{model_name}/")

def main_for_resnet(depth=18) -> None:
    os.makedirs("../internal_feature/resnet/", exist_ok=True)
    model_name = f"resnet{depth}"
    print(model_name)
    model = ResNet.__dict__[model_name](
        pretrained = True,
        model_root = "/home/bjmiao/Documents/hierarchical-feature/pretrained/online"
    )
    print(model)

    model = model.to(config.device)
    test_prefecther = load_dataset()
    test_prefecther.reset()
    batch_data = test_prefecther.next()

    fch = FeatureConcatHelper()

    while batch_data is not None:
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)
        print("New batch", images.shape)
        
        x, features = model.forward_and_extract(images)
        for key, feat in features.items():
            # print(key, feat.shape)
            fch.store(feat, key, on_gpu = False)
        
        x_ref = model(images)
        assert(torch.equal(x_ref, x))
        # break
        batch_data = test_prefecther.next()

    fch.dump(f"../internal_feature/{model_name}/block1_1/")

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
    main_for_resnet(depth=18)
    # main_for_vgg(depth=11)
    # main_for_vonenet()
    # main_for_CORNet(model_type='s')
    # main_for_CORNet(model_type='z')
    # main_for_CORNet(model_type='r')
    # main_for_CORNet(model_type='rt')

