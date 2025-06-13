from transformers import TrainingArguments, Trainer

from detr_rdd.configs import *
from detr_rdd.models.detr import *
from detr_rdd.datasets.helpers import create_detr_resnet50_image_processor

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    fp16=True,
    learning_rate=1e-5,
    weight_decay=1e-4,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    batch_eval_metrics=True,
)

def train_model(model, train_ds, eval_ds, compute_metrics):
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=create_detr_resnet50_image_processor(),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub()