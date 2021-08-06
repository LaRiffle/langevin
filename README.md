# Langevin

```
usage: main.py [-h] [--model MODEL] [--dataset DATASET] [--full_train] [--batch_size BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--l2 L2] [--epochs EPOCHS] [--optim OPTIM] [--lr LR] [--beta1 BETA1]
               [--beta2 BETA2] [--momentum MOMENTUM] [--scheduler] [--step_size STEP_SIZE] [--gamma GAMMA] [--langevin] [--sigma SIGMA] [--verbose] [--log_interval LOG_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to use for inference (resnet18, alexnet)
  --dataset DATASET     dataset to use (pneumonia, cifar10)
  --full_train          Train *all* the layers
  --batch_size BATCH_SIZE
                        size of the batch to use. Default 128
  --test_batch_size TEST_BATCH_SIZE
                        size of the batch to use for testing. Default: as batch_size
  --l2 L2               [not with --full_train] L2 regularization to make the logistic regression strongly convex. Default 0
  --epochs EPOCHS       [needs --train] number of epochs to train on. Default 30
  --optim OPTIM         optimizer to use (sgd, adam)
  --lr LR               [needs --train] learning rate of the SGD. Default 0.001
  --beta1 BETA1         [needs --optim adam] first beta parameter for Adam optimizer. Default 0.9
  --beta2 BETA2         [needs --optim adam] first beta parameter for Adam optimizer. Default 0.999
  --momentum MOMENTUM   [needs --train] momentum of the SGD. Default 0
  --scheduler           Use a scheduler for the learning rate
  --step_size STEP_SIZE
                        [needs --scheduler] Period of learning rate decay. Default 10
  --gamma GAMMA         [needs --scheduler] Multiplicative factor of learning rate decay. Default 0.5
  --langevin            Activate Langevin DP SGD
  --sigma SIGMA         [needs --langevin] noise for the Langevin DP. Default 0.01
  --verbose             show extra information and metrics
  --log_interval LOG_INTERVAL
                        [needs --test or --train] log intermediate metrics every n batches. Default 10
```