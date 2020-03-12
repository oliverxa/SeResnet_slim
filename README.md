# SeResnet_slim

## train origin model 
```
  python train.py 
```
## train sparse model
```
  python train_sparse.py 
```
## pre prune model
```
  python pre_pruned.py --pretrain weights/model_best.pth --percent 0.5 
```
## prune seresnet model
```
  python seresnet_prune.py --pretrain weights/model_best.pth --percent 0.5
```
## finetune model
```
  python finetune.py --epochs 100 --resume pruned_weights/new_model.pth --cfg cfg.txt --lr 0.001
```
