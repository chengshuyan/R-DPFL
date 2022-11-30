## R-DPFL

This repository contains the official code for the ICML 2021 paper:

[Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith

## Motivation

In Differentially private Federated Learning (DPFL), gradient clipping and random noise addition disproportionately affect statistically heterogeneous datas. As a consequence, DPFL has disparate impact: the accuracy of models trained with DPFL tends to decrease more on these datas. If the accuracy of the original model decreases on heterogeneous datas, DPFL will exacerbate this decrease. In this work, we study the utility loss inequality due to differential privacy and compare the convergence of the private and non-private models. We analyze the gradient differences caused by statistically heterogeneous datas and explain how statistical heterogeneity relates to the effect of privacy on model convergence. In addition, we propose an improved DPFL algorithm, called R-DPFL, to achieve differential privacy with the same cost but better utility. R-DPFL adjusts the gradient clipping value and the number of selected users at begining according to the degree of statistical heterogeneity of datas and weakens the direct proportional relationship between the differential privacy and the gradient difference, reducing the impact of differential privacy on the model trained by heterogeneous datas. Our experimental evaluation shows the effectiveness of our elimination algorithm in achieving the same cost of differential privacy with satisfactory utility.
<!---
![progressive_attack](https://github.com/AI-secure/PSBA/blob/master/imgs/progressive_attack.png)
-->
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
