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


def get_loaders_multiflow_v2(config):
    n_sub = config["n_sub"]
    n_full = config["n_full"]
    problem = config["problem"]
    image_size = config["image_size"]
    modal = config["modal"]
    if modal:
        volume_path = config["volume_path"]
        dataset = torch.load(
            f"{volume_path}/data/{problem}-multiflow-{n_sub}-{n_full}-{image_size}.pt"
        )
    else:
        dataset = torch.load(
            f"data/{problem}-multiflow-{n_sub}-{n_full}-{image_size}.pt"
        )

    train_data = dataset["train"]
    val_data = dataset["val"]

    train_sub = train_data["sub_meas"]
    train_sub_max, train_sub_min = train_sub.max(), train_sub.min()
    train_sub = 2.0 * (train_sub - train_sub_min) / (train_sub_max - train_sub_min) - 1.0
    train_full = train_data["full_meas"]
    train_full_max, train_full_min = train_full.max(), train_full.min()
    train_full = 2.0 * (train_full - train_full_min) / (train_full_max - train_full_min) - 1.0
    train_media = train_data["media"]
    train_media_max, train_media_min = train_media.max(), train_media.min()
    train_media = 2.0 * (train_media - train_media_min) / (train_media_max - train_media_min) - 1.0

    val_sub = val_data["sub_meas"]
    val_full = val_data["full_meas"]
    val_media = val_data["media"]

    # if channels not already added, add channels
    if len(train_sub.shape) == 3:
        train_sub = train_sub.unsqueeze(1)
    if len(train_full.shape) == 3:
        train_full = train_full.unsqueeze(1)
    if len(train_media.shape) == 3:
        train_media = train_media.unsqueeze(1)

    if len(val_sub.shape) == 3:
        val_sub = val_sub.unsqueeze(1)
    if len(val_full.shape) == 3:
        val_full = val_full.unsqueeze(1)
    if len(val_media.shape) == 3:
        val_media = val_media.unsqueeze(1)

    train = TensorDataset(
        train_sub.detach().clone(),
        train_full.detach().clone(),
        train_media.detach().clone(),
    )
    test = TensorDataset(
        val_sub.detach().clone(), val_full.detach().clone(), val_media.detach().clone()
    )

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