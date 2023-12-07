# torchvision-tinyimagenet
Dataset class for PyTorch and the TinyImageNet dataset.

# Installation

``` pip install tinyimagenet ```

# How to use
````
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
    print(f"Sample of class {klass:3d}, image {image}, words {dataset.idx_to_words[klass]}")
````

You can also check the [quickstart notebook](https://colab.research.google.com/drive/1FCDsDJg86mCjyeAWOxDW9iF49goWCx4j?usp=sharing) to peruse the dataset.

Finally, we also provide some example notebooks that use TinyImageNet with PyTorch models:

* [Evaluate a pretrained EfficientNet model](https://colab.research.google.com/github/facundoq/tinyimagenet/blob/main/Eval%20EfficientNet%20with%20TinyImageNet.ipynb#scrollTo=41aVk-yvEV-o)
* [Train a simple CNN on the dataset](
https://colab.research.google.com/github/facundoq/tinyimagenet/blob/main/Train%20basic%20CNN%20with%20TinyImageNet.ipynb#scrollTo=4CiA6z8reXYP)
* [Finetune an EfficientNet model pretrained on the full ImageNet to classify only the 200 classes of TinyImageNet](https://colab.research.google.com/github/facundoq/tinyimagenet/blob/main/Finetune%20EfficientNet%20with%20TinyImageNet.ipynb)


