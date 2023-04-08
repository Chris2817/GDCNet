# GDCNet

GDCNet: Graph Enrichment Learning via Graph Dropping Convolutional Networks

  |  ![](.\images\GDCLayer.png)

## Overview

Here we provide the implementation of a Graph Dropping Convolutional Networks (GDCNet) in Pytorch. `dropprop.py` can be used to execute a full training run on Cora.

```python
python dropprop.py --lam 1.4 --tem 0.5 --num_layers 2 --sample 5 --dataset cora --input_droprate 0.6 --hidden_droprate 0.8 --hidden 32 --lr 0.01 --patience 100 --seed 26 --drop1 0.4 --drop2 0.3 --cuda_device 0 --group 5
```

## Dependencies

- `numpy==1.21.2`
- `scipy==1.6.2`
- `networkx==2.7.1`
- `torch-geometric==2.0.4`
- `torch==1.8.1`
- `ogb==1.3.5`
- `python==3.8.13`
