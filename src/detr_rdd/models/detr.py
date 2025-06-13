from transformers import AutoModelForObjectDetection

from detr_rdd.models.helpers import *
from detr_rdd.configs import *

def create_model():
    return AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
