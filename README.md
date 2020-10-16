## Deep Linear Discriminant Analysis (DeepLDA)

This repository implements the work proposed by [Matthias Dorfer, Rainer Kelz and Gerhard Widmer](https://arxiv.org/abs/1511.04707). It adds a LDA layer to usual CNNs and is able to train with CNNs in an end-to-end fashion. \

I have tested DeepLDA with Resnet18 and compared it with vanilla Resnet18 regulated by a cross-entropy loss. The result of Resnet18-DeepLDA is really competetive. The table below displays the accuracies of two models (Trained for 20 epochs, refer the code for details). \

|    Accuracy    |    Vanilla Resnet18    |    Resnet18+DeepLDA    |
|:--------------:|:----------------------:|:----------------------:|
|    Overall     |         0.755          |         0.793          |
|    Airplane    |         0.866          |         0.787          |
|    Automobile  |         0.735          |         0.876          |
|    Bird        |         0.874          |         0.711          |
|    Cat         |         0.423          |         0.696          |
|    Deer        |         0.740          |         0.780          |
|    Dog         |         0.541          |         0.772          |
|    Frog        |         0.879          |         0.841          |
|    Horse       |         0.688          |         0.772          |
|    Ship        |         0.924          |         0.832          |
|    Truck       |         0.880          |         0.861          |


### To implement Deep LDA, refer:
https://github.com/CPJKU/deep_lda \
https://github.com/VahidooX/DeepLDA \
https://github.com/bfshi/DeepLDA


### Also helpful:
https://sthalles.github.io/fisher-linear-discriminant/ \