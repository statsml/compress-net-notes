This is a collection of papers aiming at reducing model sizes or the ASIC/FPGA accelerator for Machine Learning, especially deep neural network related applications. (Inspired by [Embedded-Neural-Network](https://github.com/ZhishengWang/Embedded-Neural-Network).)

You can use the following materials as your entrypoint:
* [Efficient Processing of Deep Neural Networks: A Tutorial and Survey](https://arxiv.org/abs/1703.09039)
* the related work of [Quantized Neural Networks](https://arxiv.org/abs/1609.07061)


# Network Compression
Note that the paper without link represents un-read/un-sort.
## Reduce Precision
[Deep neural networks are robust to weight binarization and other non-linear distortions](https://arxiv.org/abs/1606.01981) showed that DNN can be robust to more than just weight binarization.


### Linear Quantization
* Fixed point
    * [1502]. [Deep Learning with Limited Numerical Precision](https://arxiv.org/abs/1502.02551)
    * [1610]. [QSGD: Communication-Optimal Stochastic Gradient Descent, with Applications to Training Neural Networks](https://arxiv.org/abs/1610.02132)
* Dynamic fixed point
    * [1412]. [Training deep neural networks with low precision multiplications](https://arxiv.org/abs/1412.7024)
    * [1604]. [Hardware-oriented approximation of convolutional neural networks](https://arxiv.org/abs/1604.03168)
    * [1608]. [Scalable and modularized RTL compilation of convolutional neural networks onto FPGA](http://ieeexplore.ieee.org/document/7577356/)
* Binary Quantization
    * Theory proof (EBP)
        * [1405]. [Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights](https://papers.nips.cc/paper/5269-expectation-backpropagation-parameter-free-training-of-multilayer-neural-networks-with-continuous-or-discrete-weights.pdf)
        * [1503]. [Training Binary Multilayer Neural Networks for Image Classification using Expectation Backpropagation](https://arxiv.org/abs/1503.03562)
        * [1505]. [Backpropagation for Energy-Efficient Neuromorphic Computing](https://papers.nips.cc/paper/5862-backpropagation-for-energy-efficient-neuromorphic-computing)
    * More practice with 1 bit
        * [1511]. [BinaryConnect: Training Deep Neural Networks with binary weights during propagations](https://arxiv.org/abs/1511.00363)
        * [1510]. [Neural Networks with Few Multiplications](https://arxiv.org/abs/1510.03009)
        * [1601]. [Bitwise Neural Networks](https://arxiv.org/abs/1601.06071)
        * [1602]. [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
        * [1603]. [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
    * XNOR-Net with slightly large bits (1~2 bit)
        * [1606]. [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)
        * [1608]. [Recurrent Neural Networks With Limited Numerical Precision](https://arxiv.org/abs/1608.06902)
        * [1609]. [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations](https://arxiv.org/abs/1609.07061). (Text overlap with Binarized Neural Network.)
        * [1702]. [Deep Learning with Low Precision by Half-wave Gaussian Quantization](https://arxiv.org/abs/1702.00953)
* Ternary Quantization
    * [1410]. [Fixed-point feedforward deep neural network design using weights +1, 0, and -1](http://ieeexplore.ieee.org/document/6986082/)
    * [1605]. [Ternary Weight Networks](https://arxiv.org/abs/1605.04711)
    * [1612]. [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064)
* Other Quantization or others
    * [1412]. [Compressing Deep Convolutional Networks using Vector Quantization](https://arxiv.org/abs/1412.6115)
* 1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs
* Towards the Limit of Network Quantization.
* Loss-aware Binarization of Deep Networks.


### Non-linear Quantization
* Log Domain Quantization
    * [1603]. [Convolutional neural networks using logarithmic data representation](https://arxiv.org/abs/1603.01025)
    * [1609]. [LogNet: Energy-efficient neural networks using logarithmic computation](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7953288)
    * [1702]. [Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights](https://arxiv.org/abs/1702.03044)
* Parameter Sharing
    * Structured Matrices
        * Structured Convolution Matrices for Energy-efficient Deep learning.
        * Structured Transforms for Small-Footprint Deep Learning.
        * An Exploration of Parameter Redundancy in Deep Networks with Circulant Projections.
        * Theoretical Properties for Neural Networks with Weight Matrices of Low Displacement Rank.
    * Hashing
        * [1504]. [Compressing neural networks with the hashing trick](https://arxiv.org/abs/1504.04788)
        * Functional Hashing for Compressing Neural Networks
    * [1510]. [Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding](https://arxiv.org/abs/1510.00149)
    * Learning compact recurrent neural networks.


## Reduce Number of Operations and Model Size
### Exploiting Activation Statistics
* To be updated.


### Network Pruning
Network Prune: a large amount of the weights in a network are redundant and can be removed (i.e., set to zero).

* Remove low saliency
    * [9006]. [Optimal Brain Damage](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf)
    * [1506]. [Learning both weights and connections for efficient neural network](https://arxiv.org/abs/1506.02626)
* Energy-based prune
    * [1611]. [Designing energy-efficient convolutional neural networks using energy-aware pruning](https://arxiv.org/abs/1611.05128)
* Process sparse weights
    * [1402]. [A scalable sparse matrix-vector multiplication kernel for energy-efficient sparse-blas on FPGAs](https://dl.acm.org/citation.cfm?id=2554785)
    * [1510]. [Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding](https://arxiv.org/abs/1510.00149)
    * [1602]. [EIE: Efficient Inference Engine on Compressed Deep Neural Network](https://arxiv.org/abs/1602.01528)
    * [1705]. [SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks](https://arxiv.org/abs/1708.04485)
    * [1710]. [Efficient Methods and Hardware for Deep Learning, Ph.D. Thesis](https://purl.stanford.edu/qf934gh3708)
* Structured pruning
    * [1512]. [Structured Pruning of Deep Convolutional Neural Networks](https://arxiv.org/abs/1512.08571)
    * [1608]. [Learning Structured Sparsity in Deep Neural Networks](https://arxiv.org/abs/1608.03665)
    * [1705]. [Exploring the Regularity of Sparse Structure in Convolutional Neural Networks](https://arxiv.org/abs/1705.08922)


### Compact Network Architectures
* Before Training
    * use 1*1 convolutional layer to reduce the number of channels
        * [1512]. [Rethinking the inception architecture for computer vision](https://arxiv.org/abs/1512.00567)
        * [1610]. [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
        * [1704]. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
    * Bottleneck:
        * [1312]. [Network in network](https://arxiv.org/abs/1312.4400)
        * [1409]. [Going deeper with convolutions](https://arxiv.org/abs/1409.4842)
        * [1512]. [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)
        * [1602]. [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size](https://arxiv.org/abs/1602.07360)
* After Training
    * Canonical Polyadic (CP) decomposition
        * [1404]. [Exploiting linear structure within convolutional networks for efficient evaluation](https://arxiv.org/abs/1404.0736)
        * [1412]. [Speeding-up convolutional neural networks using fine-tuned cp-decomposition](https://arxiv.org/abs/1412.6553)
    * Tucker decomposition
        * [1511]. [Compression of deep convolutional neural networks for fast and low power mobile applications](https://arxiv.org/abs/1511.06530)


### Knowledge Distillation
* [0600]. [Model compression](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)
* [1312]. [Do deep nets really need to be deep?](https://arxiv.org/abs/1312.6184)
* [1412]. [Fitnets: Hints for thin deep nets](https://arxiv.org/abs/1412.6550)
* [1503]. [Distilling the knowledge in a neural network](https://arxiv.org/abs/1503.02531)
* Sequence-Level Knowledge Distillation.
* Like What You Like: Knowledge Distill via Neuron Selectivity Transfer.


# A Bit Hardware
* [1402]. [Computing's Energy Porblem (and what we can do about it)](http://ieeexplore.ieee.org/document/6757323/)
