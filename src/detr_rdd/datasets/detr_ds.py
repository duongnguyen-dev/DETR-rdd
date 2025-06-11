import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from detr_rdd.configs import CLASS_MAPPING

class DETRDataset(Dataset):
  def __init__(self, images_paths, annotations_paths, transform=None):
    self.ds = self.prepare_dataset(images_paths, annotations_paths)
    self.images = [x['image'] for x in self.ds]
    self.target = [x['target'] for x in self.ds]
    self.transform = transform
  
  def prepare_dataset(self, images_paths, annotations_paths):
    ds = []

    for index in range(0, len(annotations_paths)):
      tree = ET.parse(annotations_paths[index])
      annotated_objs = []
      labels = []

      img = Image.open(images_paths[index])
      for obj in tree.iter('object'):
        bndbox = obj.find('bndbox')
        xmax = int(bndbox.findtext('xmax'))
        xmin = int(bndbox.findtext('xmin'))
        ymax = int(bndbox.findtext('ymax'))
        ymin = int(bndbox.findtext('ymin'))
        label = obj.findtext('name')
        if label not in CLASS_MAPPING:
          label = 4
        else:
          label = CLASS_MAPPING.index(label)

        labels.append(label)
        annotated_objs.append([xmin, ymin, xmax, ymax])

      ds.append({"image": img, "target": {"annotations": annotated_objs, "labels": labels}})

    return ds

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    image = self.images[index]
    target = self.target[index]

    if self.transform:
      image, target = self.transform(image, target)

    return image, target
