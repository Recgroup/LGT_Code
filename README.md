**Towards Deeper GCNs: Alleviating Over-smoothing via Iterative Training and Fine-tuning**

**Abstract**
Graph Convolutional Networks (GCNs) suffer from severe performance degradation in deep architectures due to over-smoothing. While existing studies primarily attribute the over-smoothing to repeated applications of graph Laplacian operators, our empirical analysis reveals a critical yet overlooked factor: trainable linear transformations in GCNs significantly exacerbate feature collapse, 
even at moderate depths (e.g., 8 layers). In contrast, Simplified Graph Convolution (SGC), which removes these transformations, maintains stable feature diversity up to 32 layers, highlighting linear transformations' dual role in facilitating expressive power and inducing over-smoothing. However, completely removing linear transformations substantially weakens the model’s expressive 
capacity.
To address this trade-off, we propose **Layer-wise Gradual Training (LGT)**, a novel training strategy that progressively builds deep GCNs while preserving their expressiveness. LGT integrates three complementary components: (1) _layer-wise training_ to stabilize optimization from shallow to deep layers, (2) _low-rank adaptation_ to fine-tune shallow layers and accelerate training, and 
(3) _identity initialization_ to ensure smooth integration of new layers and accelerate convergence. Extensive experiments on benchmark datasets demonstrate that LGT achieves state-of-the-art performance on vanilla GCN, significantly improving accuracy even in 32-layer settings. Moreover, as a training method, LGT can be seamlessly combined with existing methods such as PairNorm and
ContraNorm, further enhancing their performance in deeper networks. LGT offers a general, architecture-agnostic training framework for scalable deep GCNs. The code is available at [[https://anonymous.4open.science/r/LGT_Code-384E](https://anonymous.4open.science/r/LGT_Code-384E])].

**Dependencies**

* python 3.10
* dgl 1.1.1
* pytorch 2.1.2
* torch-geometric 2.0.4
* torch-scatter 2.1.2
* torch-sparse 0.6.18
* numpy 1.24.4

**Running examples**

```Cora
python main.py --data cora --model LGT_GCN --hid 256 --lr 0.1 --epochs 100 --wightDecay 0.0005 --nlayer 2 --seed 30 --rank 2 --dropout 0.5
```

```Citeseer
python main.py --data citeseer --model LGT_GCN --hid 256 --lr 0.1 --epochs 100 --wightDecay 0.0005 --nlayer 4 --seed 30 --rank 2 --dropout 0.5
```

```Pubmed
python main.py --data pubmed --model LGT_GCN --hid 256 --lr 0.001 --epochs 100 --wightDecay 0.0005 --nlayer 8 --seed 30 --rank 2 --dropout 0.5
```

```AmazonPhoto
python main.py --data AmazonPhoto --model LGT_GCN --hid 256 --lr 0.001 --epochs 100 --wightDecay 0.0005 --nlayer 16 --seed 30 --rank 1 --dropout 0.5
```

```CoauthroCS
python main.py --data CoauthorCS --model LGT_GCN --hid 256 --lr 0.005 --epochs 100 --wightDecay 0.0005 --nlayer 32 --seed 30 --rank 0 --dropout 0.5
```