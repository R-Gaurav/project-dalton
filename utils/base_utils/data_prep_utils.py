import _init_paths

import copy
import torch
import torchvision
import tonic
import tonic.transforms as tontf
import yaml

from consts.exp_consts import EXC
from consts.dir_consts import DRC
from utils.base_utils import log

class DataPrepUtils(object):
  def __init__(self, dataset):
    """
    Args:
      dataset <str>: Which dataset to load.
    """
    self._dset = dataset
    self._dcfg = self.load_data_yaml_file()

  def load_data_yaml_file(self):
    with open(DRC.DATA_YAML_DIR + "%s.yaml" % self._dset) as f:
      dcfg = yaml.full_load(f)

    return dcfg

  def _get_image_transforms(self, use_transforms):
    if use_transforms:
      log.INFO("Using transforms to augment the training data...")
      transforms = torchvision.transforms.Compose(
          [
          torchvision.transforms.RandomCrop( # All the archs have same inp cfg.
              size=self._dcfg["spk_cnn_arch_1"]["inp_lyr"]["inp_dmx"],
              padding=4),
          torchvision.transforms.RandomHorizontalFlip(p=0.5),
          torchvision.transforms.RandomRotation(30),
          torchvision.transforms.ToTensor(),
          ]
      )
    else:
      transforms = torchvision.transforms.ToTensor()

    return transforms

  def _load_mnist(self, use_transforms):
    """
    Note: While iterating, the pixel values are scaled between 0 and 1
    (by default).
    """
    transforms = self._get_image_transforms(use_transforms)

    train_ds = torchvision.datasets.MNIST(
        DRC.DATA_DIR, train=True, download=True, transform=transforms)
    test_ds = torchvision.datasets.MNIST(
        DRC.DATA_DIR, train=False, download=True,
        transform=torchvision.transforms.ToTensor())

    return train_ds, test_ds

  def _load_cifar10(self, use_transforms):
    """
    Note: While iterating, the pixel values are scaled between 0 and 1
    (by default).
    """
    transforms = self._get_image_transforms(use_transforms)

    train_ds = torchvision.datasets.CIFAR10(
        DRC.DATA_DIR, train=True, download=True, transform=transforms)
    test_ds = torchvision.datasets.CIFAR10(
        DRC.DATA_DIR, train=False, download=True,
        transform=torchvision.transforms.ToTensor())

    return train_ds, test_ds

  def _load_fmnist(self, use_transforms):
    """
    Note: While iterating, the pixel values are scaled between 0 and 1
    (by default).
    """
    transforms = self._get_image_transforms(use_transforms)

    train_ds = torchvision.datasets.FashionMNIST(
        DRC.DATA_DIR, train=True, download=True, transform=transforms)
    test_ds = torchvision.datasets.FashionMNIST(
        DRC.DATA_DIR, train=False, download=True,
        transform=torchvision.transforms.ToTensor())

    return train_ds, test_ds

  def _load_dvs_mnist(self):
    log.INFO("Loading N-MNIST Dataset...")
    sensor_size = tonic.datasets.NMNIST.sensor_size
    ds_factor = self._dcfg["spatial_ds_factor"]
    transform = tontf.Compose([
        tontf.Denoise(filter_time=100000),
        tontf.ToFrame(
            sensor_size=sensor_size, time_window=self._dcfg["time_window"])
      ])
    train_ds = tonic.datasets.NMNIST(
        save_to=DRC.DATA_DIR, train=True, transform=transform)
    test_ds = tonic.datasets.NMNIST(
        save_to=DRC.DATA_DIR, train=False, transform=transform)

    return train_ds, test_ds

  def _load_dvs_gesture(self):
    log.INFO("Loading DVS-Gesture Dataset...")
    sensor_size = tonic.datasets.DVSGesture.sensor_size # (dm_x, dm_y, n_ch)
    ds_factor = self._dcfg["spatial_ds_factor"]
    transform = tontf.Compose([
        tontf.Denoise(filter_time=100000),
        tontf.Downsample(spatial_factor=ds_factor),
        tontf.ToFrame(
            sensor_size=(int(sensor_size[0]*ds_factor),
                         int(sensor_size[1]*ds_factor),
                         sensor_size[2]),
            time_window=self._dcfg["time_window"]),
        ])
    train_ds = tonic.datasets.DVSGesture(
        save_to=DRC.DATA_DIR, train=True, transform=transform)
    test_ds = tonic.datasets.DVSGesture(
        save_to=DRC.DATA_DIR, train=False, transform=transform)

    return train_ds, test_ds

  def _separate_train_test_dvs_cifar10(self, data, targets):
    train_size = self._dcfg["train_size"]
    test_size = self._dcfg["test_size"]
    train_x, train_y, test_x, test_y = [], [], [], []

    for cls in range(0, 10): # The 10 classes are indexed from 0 to 9.
      num_train = 0 # Reinitialize num_train to 0 for each class.
      # Note that data and targets are aligned by default.
      for idx, (file, trgt) in enumerate(zip(data, targets)):
        if trgt == cls:
          if num_train < train_size: # Training Data.
            train_x.append(file)
            train_y.append(trgt)
            num_train += 1
          else: # Test Data.
            test_x.append(file)
            test_y.append(trgt)

    return train_x, train_y, test_x, test_y

  def _load_dvs_cifar10(self):
    log.INFO("Loading DVS-CIFAR10 Dataset...")
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size # (dm_x, dm_y, n_ch)
    ds_factor = self._dcfg["spatial_ds_factor"]
    transform = tontf.Compose([
        tontf.Denoise(filter_time=100000),
        tontf.Downsample(spatial_factor=ds_factor),
        tontf.ToFrame(
            sensor_size=(int(sensor_size[0]*ds_factor),
                         int(sensor_size[1]*ds_factor),
                         sensor_size[2]),
            time_window=self._dcfg["time_window"]),
        ])
    dataset = tonic.datasets.cifar10dvs.CIFAR10DVS(
        save_to=DRC.DATA_DIR, transform=transform)
    return dataset, copy.deepcopy(dataset)

  def load_dataset(self, batch_size, use_transforms):
    if self._dset == EXC.MNIST:
      train_ds, test_ds = self._load_mnist(use_transforms)
    elif self._dset == EXC.CIFAR10:
      train_ds, test_ds = self._load_cifar10(use_transforms)
    elif self._dset == EXC.FMNIST:
      train_ds, test_ds =  self._load_fmnist(use_transforms)
    elif self._dset == EXC.DVS_GESTURE:
      train_ds, test_ds = self._load_dvs_gesture()
    elif self._dset == EXC.DVS_MNIST:
      train_ds, test_ds = self._load_dvs_mnist()
    elif self._dset == EXC.DVS_CIFAR10:
      train_ds, test_ds = self._load_dvs_cifar10()
      # Both `train_dataset` and `test_dataset` above are same, we separate them
      # below. Since there is no default separation of training and test data,
      # and there are 1000 samples for each class, we simply choose first 900
      # samples for training and rest 100 samples for test -- for every class.
      train_x, train_y, test_x, test_y = self._separate_train_test_dvs_cifar10(
          train_ds.data, train_ds.targets)
      train_ds.data, train_ds.targets = train_x, train_y
      test_ds.data, test_ds.targets = test_x, test_y
      assert len(train_ds.data) == self._dcfg["train_size"]*10 # 10 classes.
      assert len(train_ds.targets) == self._dcfg["train_size"]*10 # 10 classes.
      assert len(test_ds.data) == self._dcfg["test_size"]*10 # 10 classes.
      assert len(test_ds.targets) == self._dcfg["test_size"]*10 # 10 classes.

    else:
      sys.exit("Enter name of one of the datasets as mentioned in exp_consts")

    if not self._dcfg["event_type"]:
      train_loader = torch.utils.data.DataLoader(
          train_ds, batch_size=batch_size, num_workers=8, pin_memory=True,
          shuffle=True)
      test_loader = torch.utils.data.DataLoader(
          test_ds, batch_size=batch_size, num_workers=8, pin_memory=True,
          shuffle=True)
    else:
      train_loader = torch.utils.data.DataLoader(
          train_ds, batch_size=batch_size, num_workers=8, pin_memory=True,
          collate_fn=tonic.collation.PadTensors(batch_first=True),
          drop_last=True, shuffle=True)
      test_loader = torch.utils.data.DataLoader(
          test_ds, batch_size=batch_size, num_workers=8, pin_memory=True,
          collate_fn=tonic.collation.PadTensors(batch_first=True),
          drop_last=True, shuffle=True)

    return train_loader, test_loader
