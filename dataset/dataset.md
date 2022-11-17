# Customizing Dataset

To customize your own dataset like `CelebDF` or `DeeperForensics`, `DatasetWrapper` is what you need.

Here's an example:

```python

def get_celebdf_dataset():
    # dataset config
    label_path_dict = {}
        0.:["/path/to/celebdf/real/"], 
        1.:["/path/to/celebdf/fake/"]
    }

    # build dataset
    dataset = DatasetWrapper(label_path_dict, size=img_size, isTrain=False)

    return dataset
```

`DatasetWrapper` builds dataset by taking a `label_path_dict`, of which the key refers to the output label and the value is a list containing the path to images. If oversampling is needed, just repeat the path in list several times.

For example, if we want to build a `FaceForensics++` dataset, with real images oversampling and fake images with multi-class labels, the `label_path_dict` should be:

```python
lpd_of_FF = {
    '0' = ['/path/to/FF/train/real', '/path/to/FF/train/real/', ...],
    '1' = ['/path/to/FF/train/fake/Deepfakes'],
    '2' = ['/path/to/FF/train/fake/Face2Face'],
    ...
}
```