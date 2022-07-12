# torchvision-tinyimagenet
Dataset class for PyTorch and the TinyImageNet dataset.


# How to use
````
    split ="train" # can also be "val" or "test". Note that "test" does not have labels
    dataset = TinyImageNet(Path("~/.torchvision/tinyimagenet/"),split=split)
    n = len(dataset)
    print(f"TinyImageNet, split {split}, has  {n} samples.")
    print("Showing some samples")
    for i in range(0,n,n//5):
        image,klass = dataset[i]
        print(f"Sample of class {klass:3d}, image {image}, words {dataset.idx_to_words[klass]}")
````