from detr_rdd.datasets.helpers import create_detr_resnet50_image_processor

id2label = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "aligator_crack",
    3: "pothole",
    4: "other_corruptions"
}

label2id = {v: k for k, v in id2label.items()}

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    image_processor = create_detr_resnet50_image_processor()
    encoding = image_processor.pad(pixel_values, return_tensors="pt")

    labels = [item["labels"] for item in batch]

    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch
