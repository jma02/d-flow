
```zsh
python generate_circle_estimates.py --img-size 128 --num-samples 1 --noise 0 --num-iters 200 --data-root mesh-data --mesh-file mesh_128_h05.mat
```

```zsh
python generate_sl_estimates.py --img-size 64 --num-samples 1 --noise 0 --num-iters 200 --data-root mesh-data --mesh-file mesh_128_h05.mat --original-size 256 --pad-size 4
```

```zsh
nohup python train_interpolant.py --ckpt-path checkpoints --samples-path samples --image-size 128 --problem shepp-logan --num-epochs 150 --n-sub 2 --n-full 24 > output-shepp-128.log 2>&1 &
```