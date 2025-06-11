import torch
import random 
import torchvision.transforms.functional as F
import torchvision.transforms as T

class Resize(object):
  def __init__(self, size=(800, 800)):
    self.size = size

  def __call__(self, image, target):
    rescaled_img = F.resize(image, self.size)
    ratios = tuple(float(s) / float(s_org) for s, s_org in zip(rescaled_img.size, image.size))
    ratio_width, ratio_height = ratios

    annotations = target["annotations"]
    scaled_annotations = torch.as_tensor(annotations, dtype=torch.float32) * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    target["annotations"] = scaled_annotations

    return rescaled_img, target

class Normalize(object):
  def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    self.mean = mean
    self.std = std
  
  def convert_xyxy_to_cxcywh(self, bbox):
    xmin, ymin, xmax, ymax = bbox
    b = [torch.tensor((xmin + xmax) / 2, dtype=torch.float32),
        torch.tensor((ymin + ymax) / 2, dtype=torch.float32),
        torch.tensor((xmax - xmin), dtype=torch.float32),
        torch.tensor((ymax - ymin), dtype=torch.float32)]
    return torch.stack(b, dim=-1)

  def __call__(self, image, target=None):
    image = F.normalize(image, mean=self.mean, std=self.std)
    if target is None:
      return image, None

    target = target.copy()
    h, w = image.shape[-2:]

    annotations = target["annotations"]
    for index in range(len(annotations)):
      annotations[index] = self.convert_xyxy_to_cxcywh(annotations[index])
      annotations[index] = annotations[index] / torch.tensor([w, h, w, h], dtype=torch.float32)
    target["annotations"] = annotations

    return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
  def __call__(self, image, target):
    return T.functional.to_tensor(image), target

def make_transform():
   return Compose([
      Resize(),
      Compose([
        ToTensor(),
        Normalize()      
      ])
  ])