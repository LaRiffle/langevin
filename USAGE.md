# Usage

## Reproduce the experiments

Use the following commands (and use multiple seeds using `--seed`)

### Table 1

```
python main.py --model resnet18 --dataset cifar10 --lr 0.018 --epochs 30 --lambd 0.001 --max_grad_norm 15 --sigma 0.0022 --langevin
python main.py --model resnet18 --dataset cifar10 --lr 0.018 --epochs 30 --lambd 0.001 --max_grad_norm 5 --noise_multiplier 1.33 --renyi
python main.py --model resnet18 --dataset cifar10 --lr 0.018 --epochs 30 --lambd 0.001

python main.py --model alexnet --dataset cifar10 --lr 0.0038 --epochs 30 --lambd 0.005 --max_grad_norm 15 --sigma 0.001 --langevin
python main.py --model alexnet --dataset cifar10 --lr 0.0038 --epochs 30 --lambd 0.001 --max_grad_norm 8 --noise_multiplier 1.33 --renyi
python main.py --model alexnet --dataset cifar10 --lr 0.0038  --epochs 30 --lambd 0.001 

python main.py --model resnet18 --dataset pneumonia --lr 0.0014 --epochs 50 --lambd 0.005 --max_grad_norm 18 --sigma 0.0082 --delta 0.0001 --langevin
python main.py --model resnet18 --dataset pneumonia --lr 0.0014 --epochs 50 --lambd 0.005 --max_grad_norm 25 --noise_multiplier 4.03 --delta 0.0001 --renyi
python main.py --model resnet18 --dataset pneumonia --lr 0.0014 --epochs 50 --lambd 0.005
```

### Table 2
```
python main.py --model resnet18 --dataset cifar10 --epochs 30 --lambd 0.001 --max_grad_norm 20 --sigma 0.0021 --langevin --decreasing
python main.py --model resnet18 --dataset cifar10 --epochs 30 --lambd 0.001 --max_grad_norm 7 --noise_multiplier 1.33 --renyi --decreasing
python main.py --model resnet18 --dataset cifar10 --epochs 30 --lambd 0.001 --decreasing 

python main.py --model alexnet --dataset cifar10 --epochs 30 --lambd 0.001 --max_grad_norm 20 --sigma 0.00095 --langevin --decreasing
python main.py --model alexnet --dataset cifar10 --epochs 30 --lambd 0.001 --max_grad_norm 15 --noise_multiplier 1.33 --renyi --decreasing
python main.py --model alexnet --dataset cifar10 --epochs 30 --lambd 0.0005 --decreasing 

python main.py --model resnet18 --dataset pneumonia --lr 0.0014 --epochs 50 --lambd 0.005 --max_grad_norm 18 --sigma 0.0082 --delta 0.0001 --langevin --decreasing
python main.py --model resnet18 --dataset pneumonia --lr 0.0014 --epochs 50 --lambd 0.005 --max_grad_norm 25 --noise_multiplier 4.03 --delta 0.0001 --renyi --decreasing
python main.py --model resnet18 --dataset pneumonia --lr 0.0014 --epochs 50 --lambd 0.005 --decreasing
```

### Table 3
```
python main.py --model resnet18-finetuning --dataset cifar10 --lr 0.018 --epochs 30 --lambd 0.001 --max_grad_norm 5 --noise_multiplier 1.33 --renyi
python main.py --model resnet18-finetuning --dataset cifar10 --lr 0.018 --epochs 30 --lambd 0.001
```

## Research hyperparameters

You can use and tweek the file `hyperparam_search.py` if you want to explore other hyperparameter settings. 
