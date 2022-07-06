# PyTorch Implementation of SurvNAM

PyTorch implementation of neural additive models in
[Neural Additive Models (PyTorch)](https://github.com/kherud/neural-additive-models-pt)
is adopted for this implementation of SurvNAM.

For neural additive models, check out:
- [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912).
- [TensorFlow OG Implementation](https://github.com/google-research/google-research/tree/master/neural_additive_models)

For random survival forests (RSF):
- [julianspaeth/random-survival-forest](https://github.com/julianspaeth/random-survival-forest)

## Dependencies

```
scikit-learn>=1.0.2
numpy>=1.21.5
pandas>=1.3.5
tqdm>=4.54.0
setuptools>=61.2.0
```

## Usage


In Python:

``` python
from nam import NeuralAdditiveModel

model = NeuralAdditiveModel(input_size=x_train.shape[-1],
                            shallow_units=100,
                            hidden_units=(64, 32, 32),
                            shallow_layer=ExULayer,
                            hidden_layer=ReLULayer,
                            hidden_dropout=0.1,
                            feature_dropout=0.1)
logits, feature_nn_outputs = model.forward(x)
```

## Citation

If you use this code in your research, please cite the following paper:

### SurvNAM
> Utkin, L. V., Satyukov, E. D., & Konstantinov, A. V. (2022). 
> SurvNAM: The machine learning survival model explanation. 
> Neural Networks, 147, 81-102.

```
@article{utkin2022survnam,
    title={SurvNAM: The machine learning survival model explanation},
    author={Utkin, Lev V and Satyukov, Egor D and Konstantinov, Andrei V},
    journal={Neural Networks},
    volume={147},
    pages={81--102},
    year={2022},
    publisher={Elsevier}
}
```
### Neural Additive Models (NAM)
> Agarwal, R., Frosst, N., Zhang, X., Caruana, R., & Hinton, G. E. (2020).
> Neural additive models: Interpretable machine learning with neural nets.
> arXiv preprint arXiv:2004.13912

```
@article{agarwal2020neural,
    title={Neural additive models: Interpretable machine learning with neural nets},
    author={Agarwal, Rishabh and Frosst, Nicholas and Zhang, Xuezhou and
    Caruana, Rich and Hinton, Geoffrey E},
    journal={arXiv preprint arXiv:2004.13912},
    year={2020}
}
```