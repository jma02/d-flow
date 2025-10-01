#!/bin/sh
source ./venv/bin/activate
python data/utils/make_circles.py --im_size 128 --problem ct
python data/utils/make_shepp_logan.py --im_size 128 --problem msl
python data/utils/make_circles.py --im_size 128 --problem eit
python data/utils/make_shepp_logan.py --im_size 128 --problem eit-hc

python data/utils/sparse_and_full_ct.py --im_size 128 --problem circles --n_sub 1 --n_full 24
python data/utils/sparse_and_full_ct.py --im_size 128 --problem circles --n_sub 2 --n_full 24
python data/utils/sparse_and_full_ct.py --im_size 128 --problem circles --n_sub 3 --n_full 24
python data/utils/sparse_and_full_ct.py --im_size 128 --problem circles --n_sub 4 --n_full 24
python data/utils/sparse_and_full_ct.py --im_size 128 --problem shepp-logan --n_sub 1 --n_full 24
python data/utils/sparse_and_full_ct.py --im_size 128 --problem shepp-logan --n_sub 2 --n_full 24
python data/utils/sparse_and_full_ct.py --im_size 128 --problem shepp-logan --n_sub 3 --n_full 24
python data/utils/sparse_and_full_ct.py --im_size 128 --problem shepp-logan --n_sub 4 --n_full 24

# generating the EIT data takes a while since I haven't implemented batch support for our finite element solvers
# if you need the EIT data, run this script, then run eit_data_pairs.py separately
# or check my hugging face

