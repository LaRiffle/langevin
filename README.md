# Langevin

This is the open-source implementation of the paper:
> Differential Privacy Guarantees for Stochastic Gradient Langevin Dynamics


## Installation

### From Source
```
git clone URL
cd langevin
pip install -r requirements.txt
```

### With Docker 

```
docker-compose up
```

Connect to the container:
```
docker exec -ti ariann /bin/bash 
```

## Usage

### Reproduce experiments

To reproduce the paper experiments, see the [USAGE.md](./USAGE.md) page.

### Documentation

```
usage: main.py [-h] [--model MODEL] [--dataset DATASET] [--batch_size BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--epochs EPOCHS] [--optim OPTIM] [--lr LR] [--decreasing] [--langevin] [--renyi]
               [--delta DELTA] [--lambd LAMBD] [--beta BETA] [--sigma SIGMA] [--noise_multiplier NOISE_MULTIPLIER] [--max_grad_norm MAX_GRAD_NORM] [--seed SEED] [--silent] [--compute_features_force]
               [--log_interval LOG_INTERVAL] [--parameter_info]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model to use for inference (resnet18, alexnet, resnet18-finetuning). resnet18-finetuning cannot be used with --langevin
  --dataset DATASET     Dataset to use (cifar10, pneumonia).
  --batch_size BATCH_SIZE
                        Size of the batch to use. Default 128
  --test_batch_size TEST_BATCH_SIZE
                        Size of the batch to use for testing. Default: as batch_size
  --epochs EPOCHS       Number of epochs to train on. Default 30
  --optim OPTIM         Optimizer to use (sgd, adam)
  --lr LR               Learning rate of the SGD. Default 1 / beta.
  --decreasing          Use a decreasing learning rate in 1 / (2 beta + lambda k / 2).
  --langevin            Use Langevin DP SGD
  --renyi               Use Renyi DP SGD and Opacus
  --delta DELTA         delta constant in the DP budget. Default 1e-5
  --lambd LAMBD         L2 regularization to make the logistic regression strongly convex. Default 0.01
  --beta BETA           [needs --langevin] Smoothness constant estimation. Default AUTO-COMPUTED.
  --sigma SIGMA         [needs --langevin] Gaussian noise variance defined as std = sqrt(2.σ^2/λ). Default 0.002
  --noise_multiplier NOISE_MULTIPLIER
                        [needs --renyi] Gaussian noise variance defined as std = noise_multiplier * max_grad_norm. Default 1.2
  --max_grad_norm MAX_GRAD_NORM
                        Maximum gradient norm per sample. Default 20
  --seed SEED           Seed. Default 1
  --silent              Hide display information
  --compute_features_force
                        Force computation of the features even if already computed
  --log_interval LOG_INTERVAL
                        Log intermediate metrics every n batches. Default 1000
  --parameter_info      Print extra information about the parameter values.
  ```