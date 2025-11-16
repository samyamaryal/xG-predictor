import torchvision
import torch
from matplotlib import pyplot as plt
from PIL import Image
from transformers import VivitImageProcessor, AutoModel, VivitForVideoClassification  # ViViTModel via AutoModel
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    VivitImageProcessor,
    VivitForVideoClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import glob

MODEL_ID = "google/vivit-b-16x2-kinetics400"
processor = VivitImageProcessor.from_pretrained(MODEL_ID)
#model = AutoModel.from_pretrained(MODEL_ID).eval()
model = VivitForVideoClassification.from_pretrained(MODEL_ID).eval()

shot_batch_files = sorted(glob.glob("processed_data/shot_batch_*.npz"))
non_shot_batch_files = sorted(glob.glob("processed_data/non_shot_batch_*.npz"))

p_data = []
p_labels = []
n_data = []
n_labels = []

for f in shot_batch_files:
    batch = np.load(f)
    p_data.append(batch["data"])
    p_labels.append(batch["labels"])

for f in non_shot_batch_files:
    batch = np.load(f)
    n_data.append(batch["data"])
    n_labels.append(batch["labels"])

# shot_data = np.load("processed_data/data.npz")
# p_data = shot_data["data"]
# p_labels = shot_data['labels']

# non_shot_data = np.load("processed_data/non_shots.npz")
# n_data = non_shot_data["data"]
# n_labels = non_shot_data['labels']

X = np.concatenate((p_data, n_data), axis=1).squeeze()
y = np.concatenate((p_labels, n_labels), axis=1).squeeze()

id2label = {0: "no shot", 1: "shot"}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

model = VivitForVideoClassification.from_pretrained(
    MODEL_ID,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True, 
)

class VideoArrayDataset(Dataset):
    def __init__(self, videos, labels):
        self.videos = videos              # list of np arrays, each (T,H,W,3) uint8
        self.labels = labels              # list/np array of ints
        assert len(self.videos) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        v = self.videos[idx]
        y = self.labels[idx]
        # Ensure shape & dtype
        v = np.asarray(v)
        assert v.ndim == 4 and v.shape[-1] == 3
        if v.dtype != np.uint8:
            v = v.astype(np.uint8)
        return {"video": v, "label": y}

indices = np.arange(len(y))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
tr_idx, va_idx = indices[:split], indices[split:]

train_videos = []
train_labels = []
val_videos = []
val_labels = []

for i in tr_idx:
    train_videos.append(X[i])
    train_labels.append(y[i])

for i in va_idx:
    val_videos.append(X[i])
    val_labels.append(y[i])

# train_videos = [X[i] for i in tr_idx]
# train_labels = [y[i] for i in tr_idx]
# val_videos   = [X[i] for i in va_idx]
# val_labels   = [y[i] for i in va_idx]

train_ds = VideoArrayDataset(train_videos, train_labels)
val_ds   = VideoArrayDataset(val_videos,   val_labels)

NUM_FRAMES = model.config.num_frames  # typically 32 for this checkpoint

def collate_fn(batch):
    # batch is a list of dicts {"video": np.ndarray, "label": int}
    videos = [ex["video"] for ex in batch]     # list of (T,H,W,3) uint8
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)

    enc = processor(
            images=videos,
            return_tensors="pt",
            num_frames=NUM_FRAMES,
            do_resize=True,
            do_normalize=True,
            do_rescale=True,
        )

    enc["labels"] = labels
    print("ENC", enc)
    return enc

metric = evaluate.load("accuracy")  # or "f1" with average="macro"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)


NUM_FRAMES = model.config.num_frames  # typically 32 for this checkpoint

def collate_fn(batch):
    # batch is a list of dicts {"video": np.ndarray, "label": int}
    videos = [ex["video"] for ex in batch]     # list of (T,H,W,3) uint8
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)

    batch_list = []
    for vid in videos:
        frames_list = []
        for j in range(vid.shape[0]): # From 32 frames
            processed_frame = processor.preprocess(videos = vid[j], return_tensors="pt", num_frames=NUM_FRAMES)
            frames_list.append(processed_frame['pixel_values']) # all frames corresponding to the same shot
        batch_list.append(torch.cat(frames_list, dim=1)) # All frames corresponding to the same shot will be a single element on the list
        # batch_list.append(frames_list)

    processed_data = torch.cat(batch_list, dim=0)

    # processed_data["labels"] = labels
    return {
        "pixel_values": processed_data,
        "labels": labels 
    }



train_bs = 2
eval_bs = 2

args = TrainingArguments(
    output_dir="./vivit-exp",
    per_device_train_batch_size=train_bs,
    per_device_eval_batch_size=eval_bs,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=20,
    logging_steps=1,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    logging_dir="./runs/vivit-exp",   # directory for TB logs
    report_to=["tensorboard"],  
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()
