# Artificial Neural Network on MNIST


## Files

```
.
├── ANN.ipynb
├── train_test.py
├── ANN_class.py
└── README.md
```
## Introduction

`# of layers`: 2

`# neurals for each layer`: 64, 64

`Epoch`: 20

`Alpha`: 0.1



## Loss and Accuracy

<img src="https://github.com/trexwithoutt/Artificial_Neural_Network_detecting_MNIST/blob/master/loss.png" width="600">

<img src="https://github.com/trexwithoutt/Artificial_Neural_Network_detecting_MNIST/blob/master/accuracy.png" width="600">



```
Training Error: 0.305111104410624
>epoch=0, lr=0.100, error=0.305, accuracy=0.9159
Training Error: 0.11553265572269808
Training Error: 0.08597945220389482
Training Error: 0.06923299499781688
Training Error: 0.05804369756110391
Training Error: 0.04971688542357365
>epoch=5, lr=0.100, error=0.050, accuracy=0.9727
Training Error: 0.04316807849710413
Training Error: 0.03792529986282915
Training Error: 0.033453521099773405
Training Error: 0.02957115827923241
Training Error: 0.026376685771993025
>epoch=10, lr=0.100, error=0.026, accuracy=0.9846
Training Error: 0.023722349178411532
Training Error: 0.02143442066478345
Training Error: 0.019500904765450598
Training Error: 0.01789154038737929
Training Error: 0.016455202489956695
>epoch=15, lr=0.100, error=0.016, accuracy=0.9854
Training Error: 0.015221036176883038
Training Error: 0.014259407447295421
Training Error: 0.013333599632849448
Training Error: 0.012566579491967657
CPU times: user 6min 13s, sys: 44.2 s, total: 6min 57s
Wall time: 4min 17s

```

## Confusion Matrix

Confusion matrix can be very helpfull to see your model drawbacks.

I plot the confusion matrix of the validation results.

<img src="https://github.com/trexwithoutt/Artificial_Neural_Network_detecting_MNIST/blob/master/cm.png" width="600">


## Bad Predictions

<img src="https://github.com/trexwithoutt/Artificial_Neural_Network_detecting_MNIST/blob/master/wrong_ex.png" width="600">


