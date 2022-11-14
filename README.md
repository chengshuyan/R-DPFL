## R-DPFL

This repository contains the official code for the ICML 2021 paper:

[Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith

## Motivation

Boundary Blackbox Attack requires only decision labels to perform adversarial attacks, where query efficiency directly determines the attack efficiency. Therefore, how we estimate the gradient on the current boundary is a crucial step in this series of work.

In this paper, we theoretically show that there actually exist a trade-off between the projected length of the true gradient on subspace(the brown item) and the dimensionality of the projection subspace (purple item).

![projection](https://github.com/AI-secure/PSBA/blob/master/imgs/projection.png)

Based on this interesting finding, we propose *Progressive-Scale based projective Boundary Attack (PSBA)* via progressively searching for the optimal scale in a self-adaptive way under spatial, frequency, and spectrum scales. The image below just shows how we progressively search the optimal projection subspace on the spatial domain, and then attack the target models with this optimal scale.

![progressive_attack](https://github.com/AI-secure/PSBA/blob/master/imgs/progressive_attack.png)

### Downloading dependencies

```
pip3 install -r requirements.txt  
```
## Run on real federated datasets
(1) Specify a GPU id if needed:

```
export CUDA_VISIBLE_DEVICES=available_gpu_id
```
Otherwise just run to CPUs [might be slow if testing on Neural Network models]:

```
export CUDA_VISIBLE_DEVICES=
```

(2) Run on one dataset. First, modify the `run_fedavg.sh` and `run_fedprox.sh` scripts, specify the corresponding model of that dataset (choose from `flearn/models/$DATASET/$MODEL.py` and use `$MODEL` as the model name), specify a log file name, and configure all other parameters such as learning rate (see all hyper-parameters values in the appendix of the paper).


For example, for the data:

`nfedprox.sh`:

```
python3  -u main.py --dataset=$1 --optimizer='nfedprox'  \
            --learning_rate=0.01 --num_rounds=25 --clients_per_round=$2 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=$3 \
            --model=$9 \
            --drop_percent=0 \
            --mu=$4 --L=$5 --Clip=$6 --epsilon=$7 --delta=$8\
```

Then run:

```
mkdir mnist
bash \run_nfedprox-Copy3.sh mnist 0.5 20 1 5 5 80 0.000002 'mlp'| tee mnist/nfedprox_drop0_class1_10_epsilon80_mlp_final
```
