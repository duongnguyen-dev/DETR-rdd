import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from random import randint
from datasets import Dataset
from transformers import AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split

from detr_rdd.datasets.transforms import make_transform
from detr_rdd.configs import CLASS_MAPPING

def get_filepath(dataset_dir):
  images_paths = []
  annotations_paths = []

  annotations_path = os.path.join(dataset_dir, "train/annotations/xmls")

  for f in os.listdir(annotations_path):
    images_paths.append(os.path.join(dataset_dir, f"train/images/{f.split('.')[0]}.jpg"))
    annotations_paths.append(os.path.join(annotations_path, f))

  X_train, X_test, y_train, y_test = train_test_split(images_paths, annotations_paths, test_size=0.1, random_state=42)

  return X_train, X_test, y_train, y_test

def visualize_random_example(images_paths, annotations_paths):
  index = randint(0, len(images_paths))
  tree = ET.parse(annotations_paths[index])

  annotated_objs = []

  for obj in tree.iter('object'):
    bndbox = obj.find('bndbox')
    xmax = int(bndbox.findtext('xmax'))
    xmin = int(bndbox.findtext('xmin'))
    ymax = int(bndbox.findtext('ymax'))
    ymin = int(bndbox.findtext('ymin'))
    label = obj.findtext('name')

    annotated_objs.append([label, xmax, xmin, ymax, ymin])
  
  img = Image.open(images_paths[index])
  print(images_paths[index])
  drawed_img = ImageDraw.Draw(img)
  font = ImageFont.load_default(size=12)

  for obj in annotated_objs:
    outline = CLASS_MAPPING[obj[0]]["color"] if obj[0] in CLASS_MAPPING.keys() else "orange"
    text = CLASS_MAPPING[obj[0]]["name"] if obj[0] in CLASS_MAPPING.keys() else "other_corruption"
    drawed_img.rectangle((obj[2], obj[4], obj[1], obj[3]), outline=outline, width=2)
    drawed_img.text((obj[2], obj[4] - 20), text=text, font=font, fill=outline)
  
  plt.imshow(img)
  plt.axis('off') # Hide axes
  plt.show()

def label_mapping(label):
  if label == "D00":
    return 0
  elif label == "D10":
    return 1
  elif label == "D20":
    return 2
  elif label == "D40":
    return 3
  else:
    return 4
  
def extract_id(filename: str) -> int | None:
  match = re.search(r'\d+', filename)
  return int(match.group(0)) if match else None

def convert_voc_to_coco(bbox):
  xmin, ymin, xmax, ymax = bbox
  width = xmax - xmin
  height = ymax - ymin
  return [xmin, ymin, width, height]
  
def load_ds(image_paths, annotation_paths, transform):
  img_ids, images, bboxes, categories, areas = [], [], [], [], []

  for index in range(len(image_paths)):
    img = np.array(Image.open(image_paths[index]).convert("RGB"))[:, :, ::-1]

    img_id = extract_id(image_paths[index].split("/")[-1])
    img_ids.append(img_id)

    tree = ET.parse(annotation_paths[index])
    obj_bboxes, obj_categories, obj_areas = [], [], []
    for obj in tree.iter('object'):
      bndbox = obj.find('bndbox')
      xmax = int(bndbox.findtext('xmax'))
      xmin = int(bndbox.findtext('xmin'))
      ymax = int(bndbox.findtext('ymax'))
      ymin = int(bndbox.findtext('ymin'))
      label = label_mapping(obj.findtext('name'))

      bbox = [xmin, ymin, xmax, ymax]

      if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
        obj_bboxes.append([xmin, ymin, xmax, ymax])
        obj_categories.append(label)
        obj_areas.append((xmax-xmin) * (ymax - ymin))
      else:
        print(
          f"Image with invalid bbox: {img_id} Invalid bbox detected and discarded: {bbox} - category: {label}"
        )
    out = transform(image=img, bboxes=obj_bboxes, category=obj_categories)

    images.append(out["image"])
    bboxes.append([convert_voc_to_coco(bbox) for bbox in out['bboxes']])
    categories.append(out['category'])
    areas.append(obj_areas)

  return img_ids, images, bboxes, categories, areas

def create_detr_resnet50_image_processor():
  image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
  return image_processor

def formatted_anns(img_id, categories, areas, bboxes):
  annotations = []
  for i in range(0, len(categories)):
    # Ensure the bounding box is in the format [xmin, ymin, width, height]
    # and has non-negative width and height before creating the annotation.
    if bboxes[i][2] >= 0 and bboxes[i][3] >= 0:
      new_ann = {
        "image_id": img_id,
        "isCrowd": 0,
        "area": areas[i],
        "category_id": categories[i],
        "bbox": list(bboxes[i])
      }
      annotations.append(new_ann)
  return annotations

def transform_aug_ann(examples, transform):
  image_paths = examples["image_paths"]
  annotation_paths = examples["annotation_paths"]
  img_ids, images, bboxes, categories, areas = load_ds(image_paths, annotation_paths, transform)
  image_processor = create_detr_resnet50_image_processor()

  targets = [
      {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, bbox_)}
      for id_, cat_, ar_, bbox_ in zip(img_ids, categories, areas, bboxes)
  ]

  return image_processor(images=images, annotations=targets, return_tensors="pt")

def transform_train(examples):
  return transform_aug_ann(examples, transform=make_transform("train"), )

def transform_val(examples):
  return transform_aug_ann(examples, transform=make_transform("val"))

def make_transformed_dataset(X_train, y_train, X_test, y_test):
  train_ds = Dataset.from_dict({"image_paths": X_train, "annotation_paths": y_train})
  val_ds = Dataset.from_dict({"image_paths": X_test, "annotation_paths": y_test})

  transformed_train_ds = train_ds.with_transform(transform_train)
  transformed_val_ds = val_ds.with_transform(transform_val)

  return transformed_train_ds, transformed_val_ds

def create_dataset(dataset_dir):
  X_train, y_train, X_test, y_test = get_filepath(dataset_dir)
  transformed_train_ds, transformed_val_ds = make_transformed_dataset(X_train, y_train, X_test, y_test)

  return transformed_train_ds, transformed_val_ds