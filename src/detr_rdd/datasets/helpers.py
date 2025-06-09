import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from random import randint
from detr_rdd.configs import CLASS_MAPPING

def get_filepath(path):
  images_paths = []
  annotations_paths = []

  annotations_path = os.path.join(path, "train/annotations/xmls")

  for f in os.listdir(annotations_path):
    images_paths.append(os.path.join(path, f"train/images/{f.split('.')[0]}.jpg"))
    annotations_paths.append(os.path.join(annotations_path, f))

  return images_paths, annotations_paths

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