
from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data import TensorDataset

import os
from v3.util.general import set_seed
import pandas as pd
import seaborn as sns
import sys

def basic_data(args, batch_sz):
    raise NotImplementedError
    args.C = 5
    args.in_dim = 2

    num_train_samples = args.C * 10 # 10
    num_test_samples = args.C * 50

    #train_noise = 0.05 #0.05
    #xs, ys = sk_datasets.make_moons(n_samples=(num_train_samples + num_test_samples), shuffle=True,
    #                                          noise=train_noise)


    xs, ys = sk_datasets.make_blobs(n_samples=(num_train_samples + num_test_samples),
                             centers=args.C, n_features=args.in_dim, cluster_std=0.7, shuffle=True)

    x_train = xs[:num_train_samples]
    y_train = ys[:num_train_samples]

    x_test = xs[num_train_samples:]
    y_test = ys[num_train_samples:]

    # float64 in [-2.5, 2.5]
    f, ax = plt.subplots(2, figsize=(4, 2*4))
    assert len(x_train.shape) == 2 and x_train.shape[1] == args.in_dim
    for c in range(args.C):
        ax[0].scatter(x_train[:, 0][y_train == c], x_train[:, 1][y_train == c])

        ax[1].scatter(x_test[:, 0][y_test == c], x_test[:, 1][y_test == c])

    plt.tight_layout()
    f.savefig(os.path.join(args.out_dir, "data.png"), bbox_inches="tight")
    plt.close("all")

    x_train, y_train = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train)
    #x_train = normalize_2D(x_train)
    train_data_orig = TensorDataset(x_train, y_train)

    #assert (y_train == 1).logical_or(y_train == 0).all()

    #x_test, y_test = sk_datasets.make_moons(n_samples=num_test_samples, shuffle=True,
    #                                        noise=test_noise)
    x_test, y_test = torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test)
    #x_test = normalize_2D(x_test)
    test_data = TensorDataset(x_test, y_test)

    #num_val = int(len(train_data_orig) * args.val_pc)
    #val_data, train_data1 = torch.utils.data.random_split(train_data_orig, [num_val, len(train_data_orig) - num_val])

    train_dl = torch.utils.data.DataLoader(train_data_orig, batch_size=batch_sz, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_sz, shuffle=False)

    return train_dl, test_dl



def basic_data_instance(args, batch_sz, data_instance, size=1):
    from sklearn import datasets as sk_datasets
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sns.set_style("dark")
    #set_seed(0)

    args.C = 5
    args.in_dim = 2

    num_train_samples = args.C * 10 # 10
    num_test_samples = args.C * 50

    num_train_samples *= size
    num_test_samples *= size

    xs, ys = sk_datasets.make_blobs(n_samples=(num_train_samples + num_test_samples),
                             centers=args.C, n_features=args.in_dim, cluster_std=0.7, shuffle=True,
                                    random_state=data_instance)

    #print("Stats:")
    x_train = xs[:num_train_samples]
    y_train = ys[:num_train_samples]
    #print(np.unique(y_train, return_counts=True))
    #print((x_train.shape, y_train.shape))

    x_test = xs[num_train_samples:]
    y_test = ys[num_train_samples:]
    #print(np.unique(y_test, return_counts=True))
    #print((x_test.shape, y_test.shape))

    # float64 in [-2.5, 2.5]
    f, ax = plt.subplots(1, 2, figsize=(2 * 5, 4))
    assert len(x_train.shape) == 2 and x_train.shape[1] == args.in_dim
    table_train = []
    table_test = []
    #table = []
    for c in range(args.C):
        x0 = x_train[:, 0][y_train == c]
        x1 = x_train[:, 1][y_train == c]
        for ii in range(x0.shape[0]):
            table_train.append((x0[ii], x1[ii], c, "True"))

        x0_test = x_test[:, 0][y_test == c]
        x1_test = x_test[:, 1][y_test == c]

        for ii in range(x0_test.shape[0]):
            table_test.append((x0_test[ii], x1_test[ii], c, "False"))

    # scatter in marker 1, hue determined by class
    df_train = pd.DataFrame(table_train, columns=[r"$x_0$", r"$x_1$", "Class", "Train"])
    df_test = pd.DataFrame(table_test, columns=[r"$x_0$", r"$x_1$", "Class", "Train"])

    sns.scatterplot(data=df_train, x=r"$x_0$", y=r"$x_1$", hue="Class", palette="colorblind", s=40, ax=ax[0])
    ax[0].set_title("Training data")

    sns.scatterplot(data=df_test, x=r"$x_0$", y=r"$x_1$", hue="Class", palette="colorblind", s=40, ax=ax[1])
    ax[1].set_title("Test data")

    #ax[0].set_xlim(-4, 12)
    #ax[1].set_xlim(-4, 12)

    #ax[0].set_ylim(-5, 10)
    #ax[1].set_ylim(-5, 10)


    #plt.tight_layout()
    """
    stepsize = 2
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, stepsize))

    ax.set_ylim(-6, 10)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, stepsize))
    """

    #box = ax[1].get_position()
    #ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[1].legend(title="Class", loc="upper right")

    ax[0].legend().set_visible(False)

    f.savefig(os.path.join(args.out_dir, "data_%d.pdf" % data_instance), bbox_inches="tight")
    plt.close("all")

    x_train, y_train = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train)
    #x_train = normalize_2D(x_train)
    train_data_orig = TensorDataset(x_train, y_train)

    #assert (y_train == 1).logical_or(y_train == 0).all()

    #x_test, y_test = sk_datasets.make_moons(n_samples=num_test_samples, shuffle=True,
    #                                        noise=test_noise)
    x_test, y_test = torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test)
    #x_test = normalize_2D(x_test)
    test_data = TensorDataset(x_test, y_test)

    #num_val = int(len(train_data_orig) * args.val_pc)
    #val_data, train_data1 = torch.utils.data.random_split(train_data_orig, [num_val, len(train_data_orig) - num_val])

    train_dl = torch.utils.data.DataLoader(train_data_orig, batch_size=batch_sz, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_sz, shuffle=False)

    return train_dl, test_dl


def basic_data_instance_MNIST(args, batch_sz, data_instance, size):
    sns.set_style("dark")

    args.C = 10
    set_seed(data_instance)  # seed set again in outer loop

    args.in_dim = (28, 28)

    if args.data == "digits":

        train_data_orig = datasets.MNIST("/network/scratch/x/xu.ji/datasets/MNIST", train=True,
                                         transform=transforms.ToTensor(),
                                         target_transform=None, download=False)

        test_data = datasets.MNIST("/network/scratch/x/xu.ji/datasets/MNIST", train=False,
                                   transform=transforms.ToTensor(),
                                   target_transform=None, download=False)

    elif args.data == "fashion":

        train_data_orig = datasets.FashionMNIST("/network/scratch/x/xu.ji/datasets/FashionMNIST", train=True,
                                         transform=transforms.ToTensor(),
                                         target_transform=None, download=False)

        test_data = datasets.FashionMNIST("/network/scratch/x/xu.ji/datasets/FashionMNIST", train=False,
                                   transform=transforms.ToTensor(),
                                   target_transform=None, download=False)

    train_data = torch.utils.data.Subset(train_data_orig,
                                         np.random.choice(len(train_data_orig), size=size, replace=True))

    print("Num train %d" % len(train_data))
    print("Num test %d" % len(test_data))

    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_sz, shuffle=False)

    return train_dl, test_dl


def subsample_data_instance(args, batch_sz, data_instance, subsample, max_data_instance=10, size=1, plot=True):
    from sklearn import datasets as sk_datasets
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sns.set_style("dark")
    #set_seed(0)

    args.C = 5
    args.in_dim = 2

    subsample_factor = 2
    num_train_samples = args.C * 10 * subsample_factor
    num_test_samples = args.C * 50

    num_train_samples *= size
    num_test_samples *= size

    x_train, y_train, centers = sk_datasets.make_blobs(n_samples=(num_train_samples),
                             centers=args.C, n_features=args.in_dim, cluster_std=0.7, shuffle=True,
                                    random_state=data_instance, return_centers=True)

    x_test, y_test = sk_datasets.make_blobs(n_samples=(num_test_samples),
                             n_features=args.in_dim, cluster_std=0.7, shuffle=True,
                                    random_state=(max_data_instance + data_instance), centers=centers)

    print("Stats:")
    #x_train = xs[:num_train_samples]
    #y_train = ys[:num_train_samples]
    classes, train_counts = np.unique(y_train, return_counts=True)
    print((classes, train_counts))
    print((x_train.shape, y_train.shape))

    #x_test = xs[num_train_samples:]
    #y_test = ys[num_train_samples:]
    print(np.unique(y_test, return_counts=True))
    print((x_test.shape, y_test.shape))

    # for each data_instance superset, get subsample
    rs = np.random.RandomState(subsample)
    """
    subsample_is = rs.randint(low=0, high=2, size=num_train_samples).nonzero()[0]
    print("Num subsampled %s" % subsample_is.shape)
    x_train = x_train[subsample_is]
    y_train = y_train[subsample_is]
    """
    assert (train_counts[0].item() == train_counts).all()
    per_class = train_counts[0].item()
    new_counts = int(per_class / float(subsample_factor))
    new_xs = []
    new_ys = []
    for y in range(args.C):
        subsample_y = rs.choice(per_class, new_counts, replace=False)
        y_inds = (y_train == y).nonzero()[0][subsample_y]
        new_ys_y = y_train[y_inds]
        assert (new_ys_y == y).all()
        new_xs_y = x_train[y_inds]
        new_xs += [new_xs_y]
        new_ys += [new_ys_y]

    new_xs = np.concatenate(new_xs, axis=0)
    new_ys = np.concatenate(new_ys)
    assert new_ys.shape[0] == new_xs.shape[0] and len(new_ys.shape) == 1 and new_ys.shape[0] == args.C * new_counts
    print("Pre shuffle shapes")
    print((new_xs.shape, new_ys.shape))
    new_order = np.arange(new_ys.shape[0])
    np.random.shuffle(new_order)

    y_train = new_ys[new_order]
    x_train = new_xs[new_order]

    print("New counts")
    print(np.unique(y_train, return_counts=True))
    print((x_train.shape, y_train.shape))
    print(y_train[:10])

    sys.stdout.flush()

    # float64 in [-2.5, 2.5]
    f, ax = plt.subplots(1, 2, figsize=(2 * 5, 4))
    assert len(x_train.shape) == 2 and x_train.shape[1] == args.in_dim
    table_train = []
    table_test = []
    #table = []
    for c in range(args.C):
        x0 = x_train[:, 0][y_train == c]
        x1 = x_train[:, 1][y_train == c]
        for ii in range(x0.shape[0]):
            table_train.append((x0[ii], x1[ii], c, "True"))

        x0_test = x_test[:, 0][y_test == c]
        x1_test = x_test[:, 1][y_test == c]

        for ii in range(x0_test.shape[0]):
            table_test.append((x0_test[ii], x1_test[ii], c, "False"))

    print(len(table_train))
    print(len(table_test))

    print("table_train")
    print(table_train)

    # scatter in marker 1, hue determined by class
    df_train = pd.DataFrame(table_train, columns=[r"$x_0$", r"$x_1$", "Class", "Train"])
    df_test = pd.DataFrame(table_test, columns=[r"$x_0$", r"$x_1$", "Class", "Train"])

    sns.scatterplot(data=df_train, x=r"$x_0$", y=r"$x_1$", hue="Class", palette="colorblind", s=40, ax=ax[0])
    ax[0].set_title("Training data")

    sns.scatterplot(data=df_test, x=r"$x_0$", y=r"$x_1$", hue="Class", palette="colorblind", s=40, ax=ax[1])
    ax[1].set_title("Test data")

    #ax[0].set_xlim(-4, 12)
    #ax[1].set_xlim(-4, 12)

    #ax[0].set_ylim(-5, 10)
    #ax[1].set_ylim(-5, 10)


    #plt.tight_layout()
    """
    stepsize = 2
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, stepsize))

    ax.set_ylim(-6, 10)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, stepsize))
    """

    #box = ax[1].get_position()
    #ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[1].legend(title="Class", loc="upper right")

    ax[0].legend().set_visible(False)

    if plot:
        f.savefig(os.path.join(args.out_dir, "data_%d_subsample_%d.pdf" % (data_instance, subsample)), bbox_inches="tight")
    plt.close("all")

    x_train, y_train = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train)
    #x_train = normalize_2D(x_train)
    train_data_orig = TensorDataset(x_train, y_train)

    #assert (y_train == 1).logical_or(y_train == 0).all()

    #x_test, y_test = sk_datasets.make_moons(n_samples=num_test_samples, shuffle=True,
    #                                        noise=test_noise)
    x_test, y_test = torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test)
    #x_test = normalize_2D(x_test)
    test_data = TensorDataset(x_test, y_test)

    #num_val = int(len(train_data_orig) * args.val_pc)
    #val_data, train_data1 = torch.utils.data.random_split(train_data_orig, [num_val, len(train_data_orig) - num_val])

    train_dl = torch.utils.data.DataLoader(train_data_orig, batch_size=batch_sz, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=batch_sz, shuffle=False)

    return train_dl, test_dl