import torch
from torch.utils.data import DataLoader, TensorDataset


def get_loaders(config):
    dataset = torch.load(f"data/{config['problem']}-dataset-{config['image_size']}.pt")
    print(f"Train set shape: {dataset['train'].shape}")
    print(f"Validation set shape: {dataset['val'].shape}")
    print(f"Test set shape: {dataset['test'].shape}")

    train_min = dataset["train"].min()
    train_max = dataset["train"].max()
    # val_min = dataset['val'].min()
    # val_max = dataset['val'].max()

    dataset_train = dataset["train"]
    dataset_val = dataset["val"]

    dataset_train = 2.0 * (dataset_train - train_min) / (train_max - train_min) - 1.0
    dataset_val = 2.0 * (dataset_val - train_min) / (train_max - train_min) - 1.0

    train = TensorDataset(dataset_train.detach().clone())
    test = TensorDataset(dataset_val.detach().clone())

    bs = config["batch_size"]
    j = config["num_workers"]

    train_loader = DataLoader(
        train,
        batch_size=bs,
        shuffle=True,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=bs,
        shuffle=False,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, test_loader


def get_loaders_interpolant(config):
    n_sub = config["n_sub"]
    n_full = config["n_full"]
    problem = config["problem"]
    image_size = config["image_size"]
    dataset = torch.load(
        f"data/{problem}-sparse-and-full-{n_sub}-{n_full}-{image_size}.pt"
    )

    train_data = dataset["train"]
    val_data = dataset["val"]

    train_x = train_data["sub_meas"]
    train_y = train_data["full_meas"]

    val_x = val_data["sub_meas"]
    val_y = val_data["full_meas"]

    # if channels not already added, add channels
    if len(train_x.shape) == 3:
        train_x = train_x.unsqueeze(1)
    if len(train_y.shape) == 3:
        train_y = train_y.unsqueeze(1)
    if len(val_x.shape) == 3:
        val_x = val_x.unsqueeze(1)
    if len(val_y.shape) == 3:
        val_y = val_y.unsqueeze(1)

    train = TensorDataset(train_x.detach().clone(), train_y.detach().clone())
    test = TensorDataset(val_x.detach().clone(), val_y.detach().clone())

    bs = config["batch_size"]
    j = config["num_workers"]

    train_loader = DataLoader(
        train,
        batch_size=bs,
        shuffle=True,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=bs,
        shuffle=False,
        num_workers=j,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, test_loader
