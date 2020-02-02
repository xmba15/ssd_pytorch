## Training Data  ##
***

### [Bosch Traffic Light Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) ###
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
- Download and unzip with the following script:
```bash
    bash ./scripts/download_uecfood_256_dataset.sh
```
