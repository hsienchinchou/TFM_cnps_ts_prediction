THIS IS FOR THE MASTER THESIS IN UNIVERSIDAD CARLOS III DE MADRID, SPAIN

# Conditional Neural Processes for Time Series Prediction

Conditional Neural Processes (aka CNPs) have been recently proposed as a scalable Bayesian alternative to probabilistic prior function modelling methods (typically Gaussian Processes). CNPs are able to provide probabilistic estimates with a complexity that grows linearly with the number of training points. The goal of this thesis is to develop an open-source library for CNPs, evaluate their performance in high-dimensional settings with a massive amount of datasets, and replace the computation of embedded layers with the techniques from Recurrent Neural Network in order to fit and perform forecast on time series datasets.

## TFM_cnps_ts_prediction

This Python package constitutes a toolbox for neural processes, based on tensorflow.

* Garnelo et al., _Neural Processes_, [arXiv:1807.01622 [cs, stat]](http://arxiv.org/abs/1807.01622) (2018)
* Garnelo et al., _Conditional Neural Processes_, [arXiv:1807.01613 [cs, stat]](http://arxiv.org/abs/1807.01613) (2018)
