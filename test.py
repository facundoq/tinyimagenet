from tinyimagenet import TinyImageNet
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

split ="val"
dataset = TinyImageNet(Path("~/.torchvision/tinyimagenet/"),split=split)
n = len(dataset)
print(f"TinyImageNet, split {split}, has  {n} samples.")
n_samples = 5
print(f"Showing info of {n_samples} samples...")

for i in range(0,n,n//n_samples):
    image,klass = dataset[i]
    id = dataset.idx_to_class[klass]

    imagenet_klass = dataset.class_to_imagenet_idx[id]
    print(f"Sample of class {klass:3d} (imagenet {imagenet_klass}), image {image}, words {dataset.idx_to_words[klass]}")
    



