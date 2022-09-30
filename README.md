# FedSkip-Combatting-Statistical-Heterogeneity-with-Federated-Skip-Aggregation
This is the code for paper(ICDM22 Regular Paper) [FedSkip: Combatting Statistical Heterogeneity with Federated Skip Aggregation].

## Dependencies
* PyTorch >= 1.0.0
* torchvision >= 0.2.1
* scikit-learn >= 0.23.1

## Data Preparing
Cifar-10 and Cifar100 will be automatically downloaded in your datadir while for femnist, shakespeare and synthetic, you should refer to [LEAF](https://github.com/TalwalkarLab/leaf) or download our split version and unzip in the datadir/. 
Using LEAF to repeat our split, please refer:
1) generate a small-sized dataset of FEMNIST and full-sized datasets of SYNTHETIC and SHAKESPEARE with help of LEAF
2) remove clients with less than 64 training samples(batch size of local training).

## Model Structure
For Cifar10, Cifar100, Femnist, we use the same model structure as [MOON](https://github.com/QinbinLi/MOON).

For SHAKESPEARE, we adopt two-layer LSTM classifier containing 100 hidden units with an 8D embedding layer according to [FedProx](https://github.com/litian96/FedProx) and [LEAF](https://github.com/TalwalkarLab/leaf).

The model of SYNTHETIC is the same as LEAF: a perceptron with sigmoid activations
## Parameters
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `skip` | Number of skip between two aggregations. |
| `model`                     | The model architecture. Options: `simple-cnn`, `resnet50`.|
| `dataset`      | Dataset to use. Options: `cifar10`. `cifar100`, `femnist`,`shakespeare`,`synthetic`|
| `lr` | Learning rate. |
| `batch-size` | Batch size. |
| `epochs` | Number of local epochs. |
| `n_parties` | Number of parties. |
| `sample_fraction` | the fraction of parties to be sampled in each round. |
| `comm_round`    | Number of communication rounds. |
| `beta` | The concentration parameter of the Dirichlet distribution for non-IID partition. Setting 100000 as IID |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `seed` | The initial seed. |  

For Cifar-10 and FEMNIST, you should use simple-cnn while for CIFAR-100, you should use resnet50. You can set beta as large as possible to simulate IID when partition=non-iid. We set lr=0.01, epochs=10 and batch-size=64 by default in the paper.

## Usage
Here is an example to run FedSkip-3 on CIFAR-10 with a simple CNN:
```
python main.py --dataset=cifar10 \
    --skip=3 \
    --lr=0.01 \
    --epochs=10 \
    --model=simple-cnn \
    --comm_round=100 \
    --n_parties=10 \
    --beta=0.5 \
    --sample_fraction=1.0 \
    --logdir='./logs/' \
    --datadir='./data/' \
```
## Acknowledgement
We borrow some codes from [MOON](https://github.com/QinbinLi/MOON), [LEAF](https://github.com/TalwalkarLab/leaf) and [FedProx](https://github.com/litian96/FedProx)
## Attention

## Contact

If you have any problem with this code, please feel free to contact **zqfan_knight@sjtu.edu.cn**.
