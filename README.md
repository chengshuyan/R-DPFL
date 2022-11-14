## R-DPFL

This repository contains the official code for the ICML 2021 paper:

["Progressive-Scale Boundary Blackbox Attack via Projective Gradient Estimation".](https://arxiv.org/abs/2106.06056)

Jiawei Zhang\*, Linyi Li\*, Huichen Li, Xiaolu Zhang, Shuang Yang, Bo Li

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


For example, for all the synthetic data:

`fedavg.sh`:

```
python3  -u main.py --dataset=$1 --optimizer='fedavg'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --drop_percent=$2 \
            --model='mclr' 
```

`fedprox.sh`:

```
python3  -u main.py --dataset=$1 --optimizer='fedprox'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --drop_percent=$2 \
            --model='mclr' \
            --mu=$3
```

Then run:

```
mkdir synthetic_1_1
bash run_fedavg.sh synthetic_1_1 0 | tee synthetic_1_1/fedavg_drop0
bash run_fedprox.sh synthetic_1_1 0 0 | tee synthetic_1_1/fedprox_drop0_mu0
bash run_fedprox.sh synthetic_1_1 0 1 | tee synthetic_1_1/fedprox_drop0_mu1

bash run_fedavg.sh synthetic_1_1 0.5 | tee synthetic_1_1/fedavg_drop0.5
bash run_fedprox.sh synthetic_1_1 0.5 0 | tee synthetic_1_1/fedprox_drop0.5_mu0
bash run_fedprox.sh synthetic_1_1 0.5 1 | tee synthetic_1_1/fedprox_drop0.5_mu1

bash run_fedavg.sh synthetic_1_1 0.9 | tee synthetic_1_1/fedavg_drop0.9
bash run_fedprox.sh synthetic_1_1 0.9 0 | tee synthetic_1_1/fedprox_drop0.9_mu0
bash run_fedprox.sh synthetic_1_1 0.9 1 | tee synthetic_1_1/fedprox_drop0.9_mu1
```
