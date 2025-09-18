import torch
from torch.utils.data import DataLoader, TensorDataset


def make_checkpoint_multiflow_v1(
    path, step, epoch, model, linear_map=None, optim=None, scaler=None
):
    checkpoint = {
        "epoch": int(epoch),
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "linear_map": linear_map,
    }

    if optim is not None:
        checkpoint["optim_state_dict"] = optim.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint_multiflow_v1(path, model, optim=None, scaler=None):
    # Load checkpoint to CPU first to avoid device mismatches
    checkpoint = torch.load(path, map_location="cpu")

    state_dict = checkpoint["model_state_dict"]

    # Create a new state dict to handle the 'module.' prefix
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove `module.`
        name = (
            name[10:] if name.startswith("_orig_mod.") else name
        )  # remove `orig_mod.`
        new_state_dict[name] = v

    # Load the cleaned state dict
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict, strict=False)

    step = int(checkpoint["step"])
    epoch = int(checkpoint["epoch"])

    if optim and "optim_state_dict" in checkpoint:
        # Create a new state dict here too
        new_optim_state_dict = OrderedDict()
        for k, v in optim.state_dict().items():
            name = k[7:] if k.startswith("module.") else k
            name = (
                name[10:] if name.startswith("_orig_mod.") else name
            )  # remove `orig_mod.`
            new_optim_state_dict[name] = v
        optim.load_state_dict(new_optim_state_dict)

    if scaler and "scaler_state_dict" in checkpoint:
        new_scaler_state_dict = OrderedDict()
        for k, v in scaler.state_dict().items():
            name = k[7:] if k.startswith("module.") else k
            name = (
                name[10:] if name.startswith("_orig_mod.") else name
            )  # remove `orig_mod.`
            new_scaler_state_dict[name] = v
        scaler.load_state_dict(new_scaler_state_dict)

    return step, epoch, model, optim, scaler, checkpoint["linear_map"]


def get_loaders_multiflow_v1(config):
    n_sub = config["n_sub"]
    n_full = config["n_full"]
    problem = config["problem"]
    image_size = config["image_size"]
    modal = config["modal"]
    if modal:
        volume_path = config["volume_path"]
        dataset = torch.load(
            f"{volume_path}/data/{problem}-multiflow-v1-{n_sub}-{n_full}-{image_size}.pt"
        )
    else:
        dataset = torch.load(
            f"data/{problem}-multiflow-v1-{n_sub}-{n_full}-{image_size}.pt"
        )

    train_data = dataset["train"]
    val_data = dataset["val"]

    train_sub = train_data["sub_meas"]
    train_full = train_data["full_meas"]
    train_media = train_data["media"]

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
