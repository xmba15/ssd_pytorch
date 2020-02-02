#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from .base_dataset import BaseDataset, BaseDatasetConfig


class UecFoodDatasetConfig(BaseDatasetConfig):
    def __init__(self):
        super(UecFoodDatasetConfig, self).__init__()

        self.CLASSES = [
            "rice",
            "eels on rice",
            "pilaf",
            "chicken-'n'-egg on rice",
            "pork cutlet on rice",
            "beef curry",
            "sushi",
            "chicken rice",
            "fried rice",
            "tempura bowl",
            "bibimbap",
            "toast",
            "croissant",
            "roll bread",
            "raisin bread",
            "chip butty",
            "hamburger",
            "pizza",
            "sandwiches",
            "udon noodle",
            "tempura udon",
            "soba noodle",
            "ramen noodle",
            "beef noodle",
            "tensin noodle",
            "fried noodle",
            "spaghetti",
            "Japanese-style pancake",
            "takoyaki",
            "gratin",
            "sauteed vegetables",
            "croquette",
            "grilled eggplant",
            "sauteed spinach",
            "vegetable tempura",
            "miso soup",
            "potage",
            "sausage",
            "oden",
            "omelet",
            "ganmodoki",
            "jiaozi",
            "stew",
            "teriyaki grilled fish",
            "fried fish",
            "grilled salmon",
            "salmon meuniere",
            "sashimi",
            "grilled pacific saury",
            "sukiyaki",
            "sweet and sour pork",
            "lightly roasted fish",
            "steamed egg hotchpotch",
            "tempura",
            "fried chicken",
            "sirloin cutlet",
            "nanbanzuke",
            "boiled fish",
            "seasoned beef with potatoes",
            "hambarg steak",
            "steak",
            "dried fish",
            "ginger pork saute",
            "spicy chili-flavored tofu",
            "yakitori",
            "cabbage roll",
            "omelet",
            "egg sunny-side up",
            "natto",
            "cold tofu",
            "egg roll",
            "chilled noodle",
            "stir-fried beef and peppers",
            "simmered pork",
            "boiled chicken and vegetables",
            "sashimi bowl",
            "sushi bowl",
            "fish-shaped pancake with bean jam",
            "shrimp with chill source",
            "roast chicken",
            "steamed meat dumpling",
            "omelet with fried rice",
            "cutlet curry",
            "spaghetti meat sauce",
            "fried shrimp",
            "potato salad",
            "green salad",
            "macaroni salad",
            "Japanese tofu and vegetable chowder",
            "pork miso soup",
            "chinese soup",
            "beef bowl",
            "kinpira-style sauteed burdock",
            "rice ball",
            "pizza toast",
            "dipping noodles",
            "hot dog",
            "french fries",
            "mixed rice",
            "goya chanpuru",
            "green curry",
            "okinawa soba",
            "mango pudding",
            "almond jelly",
            "jjigae",
            "dak galbi",
            "dry curry",
            "kamameshi",
            "rice vermicelli",
            "paella",
            "tanmen",
            "kushikatu",
            "yellow curry",
            "pancake",
            "champon",
            "crape",
            "tiramisu",
            "waffle",
            "rare cheese cake",
            "shortcake",
            "chop suey",
            "twice cooked pork",
            "mushroom risotto",
            "samul",
            "zoni",
            "french toast",
            "fine white noodles",
            "minestrone",
            "pot au feu",
            "chicken nugget",
            "namero",
            "french bread",
            "rice gruel",
            "broiled eel bowl",
            "clear soup",
            "yudofu",
            "mozuku",
            "inarizushi",
            "pork loin cutlet",
            "pork fillet cutlet",
            "chicken cutlet",
            "ham cutlet",
            "minced meat cutlet",
            "thinly sliced raw horsemeat",
            "bagel",
            "scone",
            "tortilla",
            "tacos",
            "nachos",
            "meat loaf",
            "scrambled egg",
            "rice gratin",
            "lasagna",
            "Caesar salad",
            "oatmeal",
            "fried pork dumplings served in soup",
            "oshiruko",
            "muffin",
            "popcorn",
            "cream puff",
            "doughnut",
            "apple pie",
            "parfait",
            "fried pork in scoop",
            "lamb kebabs",
            "dish consisting of stir-fried potato, eggplant and green pepper",
            "roast duck",
            "hot pot",
            "pork belly",
            "xiao long bao",
            "moon cake",
            "custard tart",
            "beef noodle soup",
            "pork cutlet",
            "minced pork rice",
            "fish ball soup",
            "oyster omelette",
            "glutinous oil rice",
            "trunip pudding",
            "stinky tofu",
            "lemon fig jelly",
            "khao soi",
            "Sour prawn soup",
            "Thai papaya salad",
            "boned, sliced Hainan-style chicken with marinated rice",
            "hot and sour, fish and vegetable ragout",
            "stir-fried mixed vegetables",
            "beef in oyster sauce",
            "pork satay",
            "spicy chicken salad",
            "noodles with fish curry",
            "Pork Sticky Noodles",
            "Pork with lemon",
            "stewed pork leg",
            "charcoal-boiled pork neck",
            "fried mussel pancakes",
            "Deep Fried Chicken Wing",
            "Barbecued red pork in sauce with rice",
            "Rice with roast duck",
            "Rice crispy pork",
            "Wonton soup",
            "Chicken Rice Curry With Coconut",
            "Crispy Noodles",
            "Egg Noodle In Chicken Yellow Curry",
            "coconut milk soup",
            "pho",
            "Hue beef rice vermicelli soup",
            "Vermicelli noodles with snails",
            "Fried spring rolls",
            "Steamed rice roll",
            "Shrimp patties",
            "ball shaped bun with pork",
            "Coconut milk-flavored crepes with shrimp and beef",
            "Small steamed savory rice pancake",
            "Glutinous Rice Balls",
            "loco moco",
            "haupia",
            "malasada",
            "laulau",
            "spam musubi",
            "oxtail soup",
            "adobo",
            "lumpia",
            "brownie",
            "churro",
            "jambalaya",
            "nasi goreng",
            "ayam goreng",
            "ayam bakar",
            "bubur ayam",
            "gulai",
            "laksa",
            "mie ayam",
            "mie goreng",
            "nasi campur",
            "nasi padang",
            "nasi uduk",
            "babi guling",
            "kaya toast",
            "bak kut teh",
            "curry puff",
            "chow mein",
            "zha jiang mian",
            "kung pao chicken",
            "crullers",
            "eggplant with garlic sauce",
            "three cup chicken",
            "bean curd family style",
            "salt & pepper fried shrimp with shell",
            "baked salmon",
            "braised pork meat ball with napa cabbage",
            "winter melon soup",
            "steamed spareribs",
            "chinese pumpkin pie",
            "eight treasure rice",
            "hot & sour soup",
        ]

        self.COLORS = BaseDatasetConfig.generate_color_chart(self.num_classes)


_uec_config = UecFoodDatasetConfig()


class UecFoodDataset(BaseDataset):
    __name__ = "uec_food"

    def __init__(
        self,
        data_path,
        classes=_uec_config.CLASSES,
        colors=_uec_config.COLORS,
        phase="train",
        transform=None,
        shuffle=True,
        input_size=None,
        random_seed=2000,
        normalize_bbox=False,
        normalize_image=False,
        bbox_transformer=None,
    ):
        super(UecFoodDataset, self).__init__(
            data_path,
            classes=_uec_config.CLASSES,
            colors=_uec_config.COLORS,
            phase=phase,
            transform=transform,
            shuffle=shuffle,
            normalize_bbox=normalize_bbox,
            bbox_transformer=bbox_transformer,
        )

        self._input_size = input_size
        self._normalize_image = normalize_image

        assert phase in ("train", "val")
        self._image_path_file = os.path.join(
            self._data_path, "dataset256/uec_food_{}.txt".format(self._phase)
        )

        lines = [
            line.rstrip("\n").split(" ") for line in open(self._image_path_file)
        ]
        image_dict = {}
        for image_name, x_min, y_min, x_max, y_max, label in lines:
            if image_name not in image_dict.keys():
                image_dict[image_name] = [[], []]

            image_dict[image_name][0].append(
                [int(x_min), int(y_min), int(x_max), int(y_max)]
            )
            image_dict[image_name][1].append(int(label))

        self._image_paths = list(image_dict.keys())
        self._targets = list(image_dict.values())

        np.random.seed(random_seed)

    def _data_generation(self, idx):
        abs_image_path = os.path.join(
            self._data_path, "dataset256/{}".format(self._image_paths[idx])
        )
        img = cv2.imread(abs_image_path)
        o_height, o_width, _ = img.shape

        bboxes, category_ids = self._targets[idx]
        bboxes = [
            BaseDataset.authentize_bbox(o_height, o_width, bbox)
            for bbox in bboxes
        ]

        if self._transform:
            img, bboxes, category_ids = self._transform(
                img, bboxes, category_ids, phase=self._phase
            )

        # assert number of bboxes after transformation is greater than 0
        assert len(bboxes) > 0

        # use the height, width after transformation for normalization
        height, width, _ = img.shape
        if self._normalize_bbox:
            bboxes = [
                [
                    float(bbox[0]) / width,
                    float(bbox[1]) / height,
                    float(bbox[2]) / width,
                    float(bbox[3]) / height,
                ]
                for bbox in bboxes
            ]

        bboxes = np.array(bboxes)
        category_ids = np.array(category_ids).reshape(-1, 1)
        targets = np.concatenate((bboxes, category_ids), axis=-1)

        return img, targets
