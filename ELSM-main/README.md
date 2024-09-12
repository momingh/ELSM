# ELSM

The source code of ``Enhancing Large Models based Sequential Recommendation with
Multimodal Graph Convolution Network``.

Our code is implemented based on ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations``, [here](https://github.com/facebookresearch/generative-recommenders) is their project address. We mainly modified ``\ELSM\modeling\sequential\embedding_modules.py`` to implement our method



## Getting started

#### Data preparation

To reproduce the results in our paper please download the processed **Pantry** and **Scientific** data from [here](https://drive.google.com/drive/folders/116QDSVlrsR6IvTR7_1Q5lkptSLzaM1bk?usp=sharing). Then place the downloaded ``tmp`` file in  home directory.

#### Run model training.

You can use the following instructions to train the model and reproduce our results.

```
For Scientific
CUDA_VISIBLE_DEVICES=0 python3 train.py --gin_config_file=configs/scientific-hstu-sampled-softmax-n512-large-final.gin --master_port=12345
```

```
For Pantry
CUDA_VISIBLE_DEVICES=0 python3 train.py --gin_config_file=configs/pantry-hstu-sampled-softmax-n512-large-final.gin --master_port=12345
```



