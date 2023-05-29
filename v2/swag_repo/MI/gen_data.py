from .util import *
import argparse
import os
from torchvision import transforms, datasets
from MI.util import dataset_name

args_form = argparse.ArgumentParser(allow_abbrev=False)

args_form.add_argument("--seed", type=int, default=0)
args_form.add_argument("--data", type=str, default="CIFAR10")
args_form.add_argument("--data_root", type=str, default="/network/scratch/x/xu.ji/datasets/")
args_form.add_argument("--num_datasets", type=int, default=5)
args_form.add_argument("--subsize", type=float, default=0.3)

args = args_form.parse_args()

set_seed(args.seed)
data_root = os.path.join(args.data_root, args.data)

if args.data == "CIFAR10":
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_data = datasets.CIFAR10(root=data_root, train=False, download=False,
                                 transform=transform_test)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_data = datasets.CIFAR10(root=data_root, train=True, download=False,
                                  transform=transform_train)
else:
    raise NotImplementedError

n = len(train_data)
for i in range(args.num_datasets):
    train_data_i = torch.utils.data.Subset(train_data,
                                           np.random.choice(n, size=int(args.subsize * n),
                                                            replace=False))
    torch.save({"train_dataset": train_data_i, "test_dataset": test_data},
               os.path.join(data_root, "%s.pt" % dataset_name(args.data, args.seed, i)))
