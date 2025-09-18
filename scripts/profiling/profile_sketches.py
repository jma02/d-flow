import time
import torch
import math
from torch.func import vmap, jvp, vjp
from typing import Callable, Dict, List, Optional
import polars as pl
from torch import autocast
import torch.nn as nn
import numpy as np
from flow import sample_images
from transforms import radon
from utils import load_checkpoint
from torchdiffeq import odeint
from unet import Unet
from tqdm import tqdm
import os
import argparse
from torch.profiler import schedule

torch.manual_seed(159753)
np.random.seed(159753)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

im_size = 32

# initialize argparse
parser = argparse.ArgumentParser()
parser.add_argument("--problem", type=str, default="circles")
# either change it here or change it in cli
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--run_name", type=str, required=True)
args = parser.parse_args()
os.makedirs(f"profiling/profiling-logs-{args.run_name}", exist_ok=True)

problem = args.problem
checkpoint_path = args.checkpoint_path
if not checkpoint_path:
    checkpoint_path = f"problems/{problem}/checkpoints/{im_size}x{im_size}/ckp_3906.tar"
run_name = args.run_name

method = "euler"

# this channel parameter needs to match whatever model you are actually loading
model = Unet(ch=32).to(device)

step, epoch, model, _, _, _ = load_checkpoint(checkpoint_path, model)

# ema_model = torch.optim.swa_utils.AveragedModel(
#     model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
# )
model.eval()


def inverse_loss_fn(x, y):
    """
    Loss function for the inverse problem.
    """
    device = x.device
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, device=device, dtype=torch.float32)

    x = x.float()
    y = y.float()

    loss = torch.mean((x - y) ** 2)

    return loss


def ode_integrate(
    ode_func: nn.Module,
    init_x: torch.Tensor,
    ode_opts: Dict = {},
    t_eps: float = 0,
    init_t: float = 0.0,
    final_t: float = 1.0,
    t_arr: Optional[List[float]] = None,
    num_steps: int = 5,
    intermediate_points: bool = False,
    method: str = "euler",
) -> torch.Tensor:
    if t_arr is None:
        t = torch.linspace(init_t - t_eps, final_t, num_steps).to(init_x.device)
    else:
        t = torch.tensor(t_arr, dtype=torch.float32, device=init_x.device)

    ode_opts = {"atol": 1e-5, "rtol": 1e-5, "method": method, **ode_opts}

    z = odeint(
        func=lambda t, x: ode_func(x, t.expand(x.shape[0])), y0=init_x, t=t, **ode_opts
    )

    if not intermediate_points:
        return z[-1]
    return z


def profiler_to_polars(prof, top_n=10):
    """
    Extract Polars DataFrames from a torch.profiler.profile object.
    """
    key_avg = prof.key_averages()

    # # Check what attributes are available on the first event
    # if len(key_avg) > 0:
    #     first_evt = key_avg[0]
    #     print(
    #         f"Available attributes: {[attr for attr in dir(first_evt) if not attr.startswith('_')]}"
    #     )

    data = {
        "Name": [evt.key for evt in key_avg],
        "Occurrences": [evt.count for evt in key_avg],
        "Self_GPU_time_us": [
            evt.self_device_time_total for evt in key_avg
        ],  # Most important
        "Total_GPU_time_us": [evt.device_time_total for evt in key_avg],
        "Self_CPU_time_us": [evt.self_cpu_time_total for evt in key_avg],
        "Self_GPU_memory_bytes": [evt.self_device_memory_usage for evt in key_avg],
        "Total_GPU_memory_bytes": [evt.device_memory_usage for evt in key_avg],
    }

    df = pl.DataFrame(data, strict=False)
    return df


# load data and select a random ground truth from test set
print("Loading data")
test_data = torch.load(f"data/{problem}-dataset-{im_size}.pt")["test"]
idx = np.random.randint(0, len(test_data))
gt = test_data[idx].to(device).unsqueeze(0)

# apply radon transform, and apply multiplicative noise
img_radon = radon.radon_transform(gt.squeeze().cpu().numpy(), N=5)
img_radon = torch.tensor(img_radon, device=device, dtype=torch.float32).unsqueeze(0)
img_radon = img_radon.unsqueeze(0)  # add batch dimension

# mult noise
img_radon = (1 + 0.1 * torch.randn_like(img_radon)) * img_radon

# getting normalization constants
train_min, train_max = (
    torch.min(torch.load(f"data/{problem}-dataset-{im_size}.pt")["train"]),
    torch.max(torch.load(f"data/{problem}-dataset-{im_size}.pt")["train"]),
)

os.chdir(f"profiling/profiling-logs-{run_name}")
# test model out quickly
print("Warmup profile")
time_start = time.time()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling-logs", worker_name="local", use_gzip=False),
    profile_memory=True,
    with_stack=True,
) as prof:
    with torch.profiler.record_function("sample_images"):
        with autocast(device_type="cuda", dtype=torch.float16):
            # sample images from the flow model
            img = sample_images(
                model, (1, 1, im_size, im_size), num_steps=5, device=device
            )
        img = img[-1].squeeze().cpu().numpy()
df = profiler_to_polars(prof, top_n=10)
# save polars df
df.write_parquet("sample_images.parquet")
print(f"Warmup profile took {time.time() - time_start:.2f} seconds")


def sketch_jacobian(
    oversampling_param: int,
    sketch_dim: int,
    jac_dim: int,
    model: nn.Module,
    ode_integrate: Callable,
    x0: torch.Tensor,
    num_steps: int = 5,
    intermediate_points: bool = False,
    chunk_size: int = 8,
    method: str = "euler",
) -> torch.Tensor:
    # rangefinder step
    with torch.profiler.record_function("sketch_jacobian"):
        ell = sketch_dim + oversampling_param
        x0_detached = x0.detach().clone().requires_grad_(False)

        # we sample images of shape im_size x im_size and we compute jacobian vector products via jvp, then convert the outputs to vectors
        # those vectors are the columns of Y
        # we use jvp to form Y but we can easily also use finite difference
        def f(x_):
            with torch.profiler.record_function("ode_integrate"):
                x1 = ode_integrate(
                    model,
                    x_,
                    num_steps=num_steps,
                    intermediate_points=intermediate_points,
                    method=method,
                )
                # unnormalize
                x1 = 0.5 * (x1 + 1.0) * (train_max - train_min) + train_min
                return x1

        # we can do some vectorizing here using vmap
        Y = torch.zeros(jac_dim, ell).to(x0_detached.device)

        def single_jvp(omega):
            with torch.profiler.record_function("single_jvp"):
                return jvp(f, (x0_detached,), (omega.unsqueeze(0),))[1].view(
                    im_size * im_size
                )

        with torch.no_grad():
            omegas = torch.randn(ell, 1, im_size, im_size, device=x0.device)
            Y = vmap(single_jvp, chunk_size=chunk_size)(
                omegas
            ).T  # shape: (jac_dim, ell)
            # compute QR
            with torch.profiler.record_function("compute_qr"):
                Q_range, _ = torch.linalg.qr(Y, mode="reduced")
                del Y, omegas

        with torch.no_grad():
            _, vjp_fn = vjp(f, x0_detached)
            Q_out = Q_range.reshape(im_size * im_size, ell)
            # transpose and reshape to images to pass into f
            Q_out_imgs = Q_out.T.reshape(ell, *f(x0_detached).shape)

            def apply_vjp(v_cotangent):
                # this should reshape back to a vector
                with torch.profiler.record_function("apply_vjp"):
                    return vjp_fn(v_cotangent)[0].reshape(-1)

            # todo: probably need to make this dynamic
            B = vmap(apply_vjp, chunk_size=chunk_size)(Q_out_imgs)

            with torch.profiler.record_function("svd_decomposition"):
                U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
            del B, Q_out_imgs
            U = Q_range @ U_tilde
            torch.cuda.empty_cache() if x0.device.type == "cuda" else None
        return U, S, Vh


x0 = torch.randn((1, 1, im_size, im_size), device=device, dtype=torch.float32)
num_steps = 5
intermediate_points = False


def f(x_):
    with torch.profiler.record_function("ode_integrate"):
        x1 = ode_integrate(
            model,
            x_,
            num_steps=5,
            intermediate_points=intermediate_points,
            method=method,
        )
        # unnormalize
        x1 = 0.5 * (x1 + 1.0) * (train_max - train_min) + train_min
    return x1


print("Profiling full Jacobian sketch")
time_start = time.time()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling-logs", worker_name="local", use_gzip=False),
    profile_memory=True,
    with_stack=True,
) as prof:
    full_jacobian = torch.autograd.functional.jacobian(
        f,
        x0,
        create_graph=False,
        vectorize=True,
    ).view(im_size * im_size, im_size * im_size)
df = profiler_to_polars(prof, top_n=10)
df.write_parquet("full_jacobian.parquet")
print(f"Full Jacobian profile took {time.time() - time_start:.2f} seconds")
# sketch .1 columns of jacobian
oversampling_param = 0
sketch_dim = math.floor(im_size * im_size * 0.1)
jac_dim = im_size * im_size

print("Profiling Jacobian sketch")
time_start = time.time()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling-logs", worker_name="local", use_gzip=False),
    profile_memory=True,
    with_stack=True,
) as prof:
    sketch_jacobian_res = sketch_jacobian(
        oversampling_param=oversampling_param,
        sketch_dim=sketch_dim,
        jac_dim=jac_dim,
        model=model,
        ode_integrate=ode_integrate,
        x0=x0,
        num_steps=num_steps,
        intermediate_points=intermediate_points,
        method=method,
        chunk_size=None,
    )
df = profiler_to_polars(prof, top_n=10)
df.write_parquet("sketch_jacobian.parquet")
print(f"Jacobian sketch profile took {time.time() - time_start:.2f} seconds")
# here is some code to just try running steepest descent using sketching
optim_steps = 10000
x0 = torch.randn((1, 1, im_size, im_size), device=device, dtype=torch.float32)
sketch_dim = 100
oversampling_param = 10
x1_traj = []
y = img_radon.clone().to(device)
eta = 0.25
losses = []
target_loss = 0.02
# Initialize progress bar
pbar = tqdm(range(optim_steps), desc="Steepest Descent")
print("Profiling steepest descent with sketching")
time_start = time.time()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling-logs", worker_name="local", use_gzip=False),
    schedule=schedule(skip_first=10, wait=5, warmup=1, active=3, repeat=2),
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in pbar:
        # compute sketch of jacobian at x1
        jac_dim = im_size * im_size
        # the jacobian is already profiled here
        U, S, Vh = sketch_jacobian(
            oversampling_param=oversampling_param,
            sketch_dim=sketch_dim,
            jac_dim=jac_dim,
            model=model,
            ode_integrate=ode_integrate,
            x0=x0,
            num_steps=5,
            intermediate_points=False,
            chunk_size=None,
            method=method,
        )
        # integrate x0
        with torch.profiler.record_function("ode_integrate_during_update"):
            x1 = ode_integrate(
                model, x0, num_steps=5, intermediate_points=False, method=method
            )
        x1 = 0.5 * (x1 + 1.0) * (train_max - train_min) + train_min
        # apply forward operator, << I'm not adding noise here
        x1_radon = radon.radon_transform(x1, N=5)
        loss = inverse_loss_fn(x1, gt)
        # use autodiff to compute grad_{x_1} L(x_1)
        # we can investigate adjoint method here as well
        with torch.profiler.record_function("compute_grad_x1"):
            grad_x1 = torch.autograd.grad(
                inverse_loss_fn(x1_radon, y), x1, create_graph=False
            )[0]
        grad_x1 = grad_x1.view(im_size * im_size)
        # compute grad_{x_0} L(x_1) using chain rule
        # use element wise multiplication for the dot product <S, U^T @ grad_x1>
        with torch.profiler.record_function("matmul"):
            loss_grad = Vh.T @ (S * (U.T @ grad_x1))

        # update x0 using the computed gradient
        x0 = x0 - eta * loss_grad.view_as(x0)
        x1_traj.append(x1.detach().clone().cpu().numpy())

        # step the profiler!!!
        prof.step()
        # Update progress bar with loss
        pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        if step % 100 == 0:
            losses.append(loss)
        if loss < target_loss:
            print(f"Target loss reached at step {step}, Loss: {loss.item():.6f}")
            break
df = profiler_to_polars(prof, top_n=10)
df.write_parquet("steepest_descent.parquet")
print(f"Steepest descent profile took {time.time() - time_start:.2f} seconds")

# final experiment
# we want to compare performance of torch autograd against the full gradient, versus our sketching idea

x0 = torch.randn((1, 1, im_size, im_size), device=device, dtype=torch.float32)
x0 = x0.requires_grad_(True)
print("Profiling autograd vs sketch")
time_start = time.time()
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiling-logs", worker_name="local", use_gzip=False),
    profile_memory=True,
    with_stack=True,
) as prof:
    x1 = ode_integrate(model, x0, num_steps=5, intermediate_points=False, method=method)
    x1 = 0.5 * (x1 + 1.0) * (train_max - train_min) + train_min
    x1 = x1.requires_grad_(True)
    y = radon.radon_transform(x1, N=5)
    loss = inverse_loss_fn(y, img_radon)
    with torch.profiler.record_function("compute_grad"):
        grad = torch.autograd.grad(loss, x0, create_graph=True)[0]
    with torch.profiler.record_function("compute_grad_via_sketch"):
        U, S, Vh = sketch_jacobian(
            oversampling_param=oversampling_param,
            sketch_dim=sketch_dim,
            jac_dim=jac_dim,
            model=model,
            ode_integrate=ode_integrate,
            x0=x0,
            num_steps=5,
            intermediate_points=False,
            chunk_size=None,
            method=method,
        )
        grad_x1 = torch.autograd.grad(
            inverse_loss_fn(y, img_radon), x1, create_graph=False
        )[0]
        grad_x1 = grad_x1.view(im_size * im_size)

        loss_grad = Vh.T @ (S * (U.T @ grad_x1))

df = profiler_to_polars(prof, top_n=10)
df.write_parquet("autograd_vs_sketch.parquet")
print(f"Final experiment took {time.time() - time_start:.2f} seconds")
