import albumentations as A

def make_transform(mode):
  if mode == "train":
    return A.Compose(
      [
        A.LongestMaxSize(500),
        A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
      ],
      bbox_params = A.BboxParams(format='pascal_voc', label_fields=['category'])
    )
  else:
    return A.Compose(
      [
        A.LongestMaxSize(500),
        A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
      ],
      bbox_params = A.BboxParams(format='pascal_voc', label_fields=['category'])
    )