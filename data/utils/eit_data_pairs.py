# Assuming access to a dataset of media, we will generate data pairs of sparse CT measurements and "full" CT measurements
import argparse
import torch
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as tmp

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from forward_solvers.dtn_from_sigma import dtn_from_sigma
from forward_solvers.fem import Mesh, V_h
from scipy import io as sio
from tqdm import tqdm


def process_image_worker(worker_args):
    """Worker function for processing a single image"""
    img_data, img_idx, mesh_data, img_size, device_id = worker_args
    
    # Set device for this worker
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device)
    
    # Recreate mesh on this device
    p, t, vol_idx, bdy_idx = mesh_data
    p = p.to(device)
    t = t.to(device) 
    vol_idx = vol_idx.to(device)
    bdy_idx = bdy_idx.to(device)
    
    mesh = Mesh(p, t, bdy_idx, vol_idx)
    v_h = V_h(mesh)
    
    # Process single image
    img = img_data.to(device)  # shape (H, W)
    dtn_map = dtn_from_sigma(sigma_vec=img, v_h=v_h, mesh=mesh, img_size=img_size, device=device)
    
    return img_idx, dtn_map.cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate data pairs of conductivities and dtn maps.")
    parser.add_argument('--problem', type=str, default='shepp-logan', help='Dataset to use')
    parser.add_argument('--workers_per_gpu', type=int, default=4, help='Number of workers per GPU')

    args = parser.parse_args()

    # Set multiprocessing start method
    tmp.set_start_method('spawn', force=True)

    dataset_source = torch.load(f"data/eit-{args.problem}-dataset-128.pt")

    dataset = {}

    # load mesh
    data_root = 'forward_solvers/mesh-data'
    mesh_file = 'mesh_128_h05.mat'
    img_size = 128
    original_size = 128
    pad_size = 0
    device = "cuda"

    mat_fname  = os.path.join(data_root, mesh_file)
    mat_contents = sio.loadmat(mat_fname)

    p = torch.tensor(mat_contents['p'], dtype=torch.float64)
    t = torch.tensor(mat_contents['t']-1, dtype=torch.long)
    vol_idx = torch.tensor(mat_contents['vol_idx'].reshape((-1,))-1, dtype=torch.long)
    bdy_idx = torch.tensor(mat_contents['bdy_idx'].reshape((-1,))-1, dtype=torch.long)

    # Prepare mesh data for workers (keep on CPU for transfer)
    mesh_data = (p, t, vol_idx, bdy_idx)

    for split in ['train', 'val', 'test']:
        images = dataset_source[split]  # shape (N, 1, H, W)
        N, C, H, W = images.shape
        images = images.squeeze(1)  # shape (N, H, W)

        # Prepare workers - multiple workers per GPU
        num_gpus = torch.cuda.device_count()
        total_workers = args.workers_per_gpu * num_gpus
        
        print(f"Processing {split} with {total_workers} workers ({args.workers_per_gpu} per GPU) across {num_gpus} GPUs")
        
        # Create worker arguments - distribute images across all workers
        worker_args = []
        for i in range(N):
            # Round-robin assignment to workers, then map workers to GPUs
            worker_id = i % total_workers
            device_id = worker_id % num_gpus
            img_data = images[i]  # shape (H, W)
            worker_args.append((img_data, i, mesh_data, img_size, device_id))

        # Process with multiprocessing
        dtn_maps = [None] * N
        
        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            # Submit all jobs
            futures = [executor.submit(process_image_worker, args) for args in worker_args]
            
            # Collect results with progress bar
            for future in tqdm(futures, desc=f"Processing {split} images"):
                img_idx, result = future.result()
                dtn_maps[img_idx] = result

        # Stack results
        dtn_maps = torch.stack(dtn_maps, dim=0)  # shape (N, 128, 128)
        
        dataset[split] = {
            'dtn_map': dtn_maps,
            'media': images  # images already squeezed above
        }

    save_name = f"data/eit-{args.problem}-multiflow-128.pt"
    torch.save(dataset, save_name)
    print(f"Saved dataset to {save_name}")