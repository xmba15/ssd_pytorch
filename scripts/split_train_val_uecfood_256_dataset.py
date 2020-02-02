#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import tqdm


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def shuffle_two_lists(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)

    return a, b


def main():
    uec_dataset_path = os.path.abspath(
        os.path.join(_CURRENT_DIR, "../data/dataset256/UECFOOD256")
    )
    assert os.path.isdir(uec_dataset_path)

    multiple_food_file = os.path.abspath(
        os.path.join(_CURRENT_DIR, "../data/dataset256/multiple_food.txt")
    )

    lines = [line.rstrip("\n") for line in open(multiple_food_file, "r")]
    lines = lines[1:]
    lines = [line.split(" ") for line in lines]
    multiple_label_images = {}
    for line in lines:
        multiple_label_images[int(line[0])] = [
            (int(elem) - 1) for elem in line[1:-1]
        ]
    procesed_images = []

    train_data = []
    train_bboxes = []
    val_data = []
    val_bboxes = []
    train_val_ratio = 0.9

    for i in tqdm.tqdm(range(256)):
        current_set_path = os.path.join(uec_dataset_path, str(i + 1))
        bbox_info_file = os.path.join(current_set_path, "bb_info.txt")

        lines = [line.rstrip("\n") for line in open(bbox_info_file, "r")]
        lines = lines[1:]
        current_data = {}

        for line in lines:
            img_num, x_min, y_min, x_max, y_max = [
                int(e) for e in line.split(" ")
            ]
            if img_num in procesed_images:
                continue
            current_data[img_num] = []
            current_data[img_num].append([x_min, y_min, x_max, y_max, i])
            if img_num in multiple_label_images.keys():
                categories = multiple_label_images[img_num]
                for category in categories:
                    if category == i:
                        continue
                    extra_check_file = os.path.join(
                        uec_dataset_path, str(category + 1), "bb_info.txt"
                    )

                    new_lines = [
                        new_line.rstrip("\n")
                        for new_line in open(extra_check_file, "r")
                    ][1:]
                    for new_line in new_lines:
                        if int(new_line.split(" ")[0]) == img_num:
                            new_splitted = [
                                int(e) for e in new_line.split(" ")[1:]
                            ]
                            new_splitted.append(category)
                            current_data[img_num].append(new_splitted)
                            break

                procesed_images.append(img_num)

        train_len = int(train_val_ratio * len(current_data))

        all_keys = list(current_data.keys())
        all_keys = [
            "UECFOOD256/{}/{}.jpg".format(i + 1, key) for key in all_keys
        ]
        all_values = list(current_data.values())

        train_data += all_keys[:train_len]
        train_bboxes += all_values[:train_len]
        val_data += all_keys[train_len:]
        val_bboxes += all_values[train_len:]

    random.seed(2000)

    train_data, train_bboxes = shuffle_two_lists(train_data, train_bboxes)
    val_data, val_bboxes = shuffle_two_lists(val_data, val_bboxes)

    train_txt = os.path.abspath(
        os.path.join(_CURRENT_DIR, "../data/dataset256", "uec_food_train.txt")
    )
    val_txt = os.path.abspath(
        os.path.join(_CURRENT_DIR, "../data/dataset256", "uec_food_val.txt")
    )

    with open(train_txt, "w") as f:
        for i in tqdm.tqdm(range(len(train_data))):
            line_to_write = ""
            for bbox in train_bboxes[i]:
                x_min, y_min, x_max, y_max, label = bbox
                line_to_write += "{} {} {} {} {} {}\n".format(
                    train_data[i], x_min, y_min, x_max, y_max, label
                )
            f.write(line_to_write)

    with open(val_txt, "w") as f:
        for i in tqdm.tqdm(range(len(val_data))):
            line_to_write = ""
            for bbox in val_bboxes[i]:

                x_min, y_min, x_max, y_max, label = bbox

                line_to_write += "{} {} {} {} {} {}\n".format(
                    val_data[i], x_min, y_min, x_max, y_max, label
                )
            f.write(line_to_write)


if __name__ == "__main__":
    main()
