# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import cv2
import numpy as np

from ..dataset_base import DatasetBase
from .sunrgbd import SUNRBDBase


class SUNRGBD(SUNRBDBase, DatasetBase):
    def __init__(
        self, data_dir=None, split="train", depth_mode="refined", with_input_orig=False, rec_data_dir=None, draw=False
    ):
        super(SUNRGBD, self).__init__()

        self._n_classes = self.N_CLASSES
        self._cameras = ["realsense", "kv2", "kv1", "xtion"]
        assert split in self.SPLITS, f"parameter split must be one of {self.SPLITS}, got {split}"
        self._split = split
        assert depth_mode in ["refined", "raw"]
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self.draw = draw

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            self._data_dir = data_dir
            self._rec_data_dir = None
            self.img_dir, self.depth_dir, self.label_dir = self.load_file_lists()
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        self._class_names = self.CLASS_NAMES_ENGLISH
        self._class_colors = np.array(self.CLASS_COLORS, dtype="uint8")

        # note that mean and std differ depending on the selected depth_mode
        # however, the impact is marginal, therefore, we decided to use the
        # stats for refined depth for both cases
        # stats for raw: mean: 18320.348967710495, std: 8898.658819551309
        self._depth_mean = 19025.14930492213
        self._depth_std = 9880.916071806689

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def load_image(self, idx):
        if self.camera is None:
            img_dir = self.img_dir[self._split]["list"]
        else:
            img_dir = self.img_dir[self._split]["dict"][self.camera]
        fp = os.path.join(self._data_dir, img_dir[idx])
        image = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_depth(self, idx):
        if self.camera is None:
            depth_dir = self.depth_dir[self._split]["list"]
        else:
            depth_dir = self.depth_dir[self._split]["dict"][self.camera]

        if self._depth_mode == "raw":
            depth_file = depth_dir[idx].replace("depth_bfx", "depth")
        else:
            depth_file = depth_dir[idx]

        fp = os.path.join(self._data_dir, depth_dir[idx])
        depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        return depth

    def load_label(self, idx):  # label不需要重建
        if self.camera is None:
            label_dir = self.label_dir[self._split]["list"]
        else:
            label_dir = self.label_dir[self._split]["dict"][self.camera]
        label = np.load(os.path.join(self._data_dir, label_dir[idx])).astype(np.uint8)

        return label

    def load_file_lists(self):
        def _get_filepath(filename):
            return os.path.join(self._data_dir, filename)

        img_dir_train_file = _get_filepath("train_rgb.txt")
        depth_dir_train_file = _get_filepath("train_depth.txt")
        label_dir_train_file = _get_filepath("train_label.txt")

        img_dir_test_file = _get_filepath("test_rgb.txt")
        depth_dir_test_file = _get_filepath("test_depth.txt")
        label_dir_test_file = _get_filepath("test_label.txt")

        img_dir = dict()
        depth_dir = dict()
        label_dir = dict()

        for phase in ["train", "test"]:
            img_dir[phase] = dict()
            depth_dir[phase] = dict()
            label_dir[phase] = dict()

        img_dir["train"]["list"], img_dir["train"]["dict"] = self.list_and_dict_from_file(img_dir_train_file)
        depth_dir["train"]["list"], depth_dir["train"]["dict"] = self.list_and_dict_from_file(depth_dir_train_file)
        label_dir["train"]["list"], label_dir["train"]["dict"] = self.list_and_dict_from_file(label_dir_train_file)

        img_dir["test"]["list"], img_dir["test"]["dict"] = self.list_and_dict_from_file(img_dir_test_file)
        depth_dir["test"]["list"], depth_dir["test"]["dict"] = self.list_and_dict_from_file(depth_dir_test_file)
        label_dir["test"]["list"], label_dir["test"]["dict"] = self.list_and_dict_from_file(label_dir_test_file)

        if self.draw:
            img_dir = [x for x in img_dir if x.find("10027") != -1]
            depth_dir = [x for x in depth_dir if x.find("10027") != -1]
            label_dir = [x for x in label_dir if x.find("10027") != -1]

        return img_dir, depth_dir, label_dir

    def list_and_dict_from_file(self, filepath):
        with open(filepath, "r") as f:
            file_list = f.read().splitlines()
        dictionary = dict()
        for cam in self.cameras:
            dictionary[cam] = [i for i in file_list if cam in i]

        return file_list, dictionary

    def __len__(self):
        if self.camera is None:
            return len(self.img_dir[self._split]["list"])
        return len(self.img_dir[self._split]["dict"][self.camera])


class MySUNRGBD(SUNRBDBase, DatasetBase):
    def __init__(
        self, data_dir=None, split="train", depth_mode="refined", with_input_orig=False, rec_data_dir=None, draw=False
    ):
        super(MySUNRGBD, self).__init__()

        self._n_classes = self.N_CLASSES
        self._cameras = ["test"]
        assert split in self.SPLITS, f"parameter split must be one of {self.SPLITS}, got {split}"
        self._split = split
        assert depth_mode in ["refined", "raw"]
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self.draw = draw
        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            self._data_dir = data_dir
            self._rec_data_dir = rec_data_dir if rec_data_dir.lower().find("sun") != -1 else None
            self.img_dir, self.depth_dir, self.label_dir = self.load_file_lists()
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        self._class_names = self.CLASS_NAMES_ENGLISH
        self._class_colors = np.array(self.CLASS_COLORS, dtype="uint8")

        # note that mean and std differ depending on the selected depth_mode
        # however, the impact is marginal, therefore, we decided to use the
        # stats for refined depth for both cases
        # stats for raw: mean: 18320.348967710495, std: 8898.658819551309
        self._depth_mean = 19025.14930492213
        self._depth_std = 9880.916071806689

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def load_image(self, idx):
        if self.camera is None:
            img_dir = self.img_dir[self._split]["list"]
        else:
            img_dir = self.img_dir[self._split]["dict"][self.camera]

        if self._rec_data_dir is None or self._rec_data_dir.find("sun") == -1:
            fp = os.path.join(self._data_dir, img_dir[idx])
        else:
            fp = os.path.join(self._rec_data_dir, "rgb_rec", img_dir[idx])

        print(fp)
        image = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_depth(self, idx):
        if self.camera is None:
            depth_dir = self.depth_dir[self._split]["list"]
        else:
            depth_dir = self.depth_dir[self._split]["dict"][self.camera]

        if self._depth_mode == "raw":
            depth_file = depth_dir[idx].replace("depth_bfx", "depth")
        else:
            depth_file = depth_dir[idx]

        if self._rec_data_dir is None or self._rec_data_dir.find("sun") == -1:
            fp = os.path.join(self._data_dir, depth_file)
        else:
            fp = os.path.join(self._rec_data_dir, "depth_rec", depth_file)
        print(fp)

        depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)  # 应该读取gt,而不是8bit
        return depth

    def load_label(self, idx):
        if self.camera is None:
            label_dir = self.label_dir[self._split]["list"]
        else:
            label_dir = self.label_dir[self._split]["dict"][self.camera]
        label = np.load(os.path.join(self._data_dir, label_dir[idx])).astype(np.uint8)

        return label

    def load_file_lists(self):
        def _get_filepath(filename):
            if self._rec_data_dir is not None and filename.find("label") == -1:
                return os.path.join(self._rec_data_dir, filename)  # 需要记录txt里面
            return os.path.join(self._data_dir, filename)

        # img_dir_train_file = _get_filepath('train_rgb.txt')
        # depth_dir_train_file = _get_filepath('train_depth.txt')
        # label_dir_train_file = _get_filepath('train_label.txt')

        img_dir_test_file = _get_filepath("my_test_rgb.txt")
        depth_dir_test_file = _get_filepath("my_test_depth.txt")
        label_dir_test_file = _get_filepath("my_test_label.txt")

        img_dir = dict()
        depth_dir = dict()
        label_dir = dict()

        for phase in ["train", "test"]:
            img_dir[phase] = dict()
            depth_dir[phase] = dict()
            label_dir[phase] = dict()

        img_dir["test"]["list"], img_dir["test"]["dict"] = self.list_and_dict_from_file(img_dir_test_file)
        depth_dir["test"]["list"], depth_dir["test"]["dict"] = self.list_and_dict_from_file(depth_dir_test_file)
        label_dir["test"]["list"], label_dir["test"]["dict"] = self.list_and_dict_from_file(label_dir_test_file)

        img_dir["train"]["list"], img_dir["train"]["dict"] = img_dir["test"]["list"], img_dir["test"]["dict"]
        depth_dir["train"]["list"], depth_dir["train"]["dict"] = depth_dir["test"]["list"], depth_dir["test"]["dict"]
        label_dir["train"]["list"], label_dir["train"]["dict"] = label_dir["test"]["list"], label_dir["test"]["dict"]

        if self.draw:
            img_dir = [x for x in img_dir if x.find("10027") != -1]
            depth_dir = [x for x in depth_dir if x.find("10027") != -1]
            label_dir = [x for x in label_dir if x.find("10027") != -1]
        return img_dir, depth_dir, label_dir

    def list_and_dict_from_file(self, filepath):
        if filepath.find("label") != -1 or self._rec_data_dir is None or self._rec_data_dir.find("sun") == -1:
            with open(filepath, "r") as f:
                file_list = f.read().splitlines()
        else:
            pardir = os.path.dirname(filepath)
            if filepath.find("depth") != -1 or filepath.find("gt") != -1:  # bug sunrgbd中含有rgb
                # if filepath.find('rgb') !=-1 or filepath.find('color')!=-1: # bug sunrgbd中含有rgb
                file_list = os.listdir(os.path.join(pardir, "depth_rec"))
                file_list = [f for f in file_list if f.find("16bit") != -1]
            else:
                file_list = os.listdir(os.path.join(pardir, "rgb_rec"))
                file_list = [f for f in file_list if f.find("rec.png") != -1]  # 这个判断是否VTM
            file_list.sort()

        dictionary = dict()
        for cam in self.cameras:
            # dictionary[cam] = [i for i in file_list if cam in i]
            dictionary[cam] = [i for i in file_list]

        return file_list, dictionary

    def __len__(self):
        if self.camera is None:
            return len(self.img_dir[self._split]["list"])
        return len(self.img_dir[self._split]["dict"][self.camera])
