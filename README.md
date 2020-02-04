## Training Data  ##
***

### [Bosch Traffic Light Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) ###
***

- After getting the url provided from [HERE](https://hci.iwr.uni-heidelberg.de/node/6132), you can download the dataset manually or using the cookies of the url, to download the dataset with my script:

```bash
    bash ./scripts/download_bosch_traffic_light_dataset.sh [/path/to/cookies file]
```

- The dataset should be placed into ./data/bosch_traffic_light
```text
---bosch_traffic_light---
                        |---dataset_train_rgb
                        |---dataset_test_rgb
```

### [Japanese Food Dataset](http://foodcam.mobi/dataset256.html)
***

- Download and unzip with the following script:
```bash
    bash ./scripts/download_uecfood_256_dataset.sh
```

### [VOC(2007) Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
***

- Download and unzip with the following script:
```bash
    bash ./scripts/download_voc_dataset.sh
```

## Training ##
***

```bash
    python [dataset_name]_train.py --batch_size {batch_size} --lr_rate {lr_rate} --num_epoch {num_epoch} --snapshot {snapshot} --save_period {save_period}
```

```text
    snapshot: for resuming training from a checkpoint
    save_period: frequency to save checkpoints
```

## Testing ##
***

```bash
    python [dataset_name]_test_single_image.py --snapshot {snapshot} --image_path {image_path} --conf_thresh {conf_thresh}
```
