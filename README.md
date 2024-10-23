# Code for BCR

Code for "Leveraging Bilateral Correlations for Multi-Label Few-Shot Learning" in IEEE Transactions on Neural Networks and Learning Systems.

If you use the code in this repo for your work, please cite the following bib entries:

```
@ARTICLE{TNNLS.2024.3388094,
  author={An, Yuexuan and Xue, Hui and Zhao, Xingyu and Xu, Ning and Fang, Pengfei and Geng Xin},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Leveraging Bilateral Correlations for Multi-Label Few-Shot Learning}, 
  year={2024},
  doi={10.1109/TNNLS.2024.3388094}
}
```

## Requirements

- Python >= 3.6
- PyTorch (GPU version) >= 1.5
- NumPy >= 1.13.3
- Scikit-learn >= 0.20

## Getting started

Download the MS-COCO, CUB-200-2011, NUS-WIDE, and Visual Genome datasets. 

You can change the path of the datasets in *data/dataset.py*.

## Running the scripts

To train and test the BCR model in the terminal, use:

```bash
$ python run_bcr.py --dataset_name VG --algorithm bcr --model_name Conv4 --n_way 10 --n_shot 1 --max_epoch 500 --hidden_dim 100 --eta 0.5 --gamma 0.5 --num_workers 8 --device cuda:0 --seed 0
```

## Acknowledgment

Our project references the code in the following repo.

[CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot)

