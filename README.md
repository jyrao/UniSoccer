# UniSoccer: Towards Universal Soccer Video Understanding
This repository contains the official PyTorch implementation of UniSoccer: https://arxiv.org/abs/2412.01820/.

The code will be released soon...

<div align="center">
   <img src="./architecture.png">
</div>

## Some Information
[Project Page](https://jyrao.github.io/UniSoccer/)  $\cdot$ [Paper](https://arxiv.org/abs/2412.01820/)(Soon) $\cdot$ [Dataset]()(Soon) $\cdot$ [Checkpoints]()(Soon) 

## News
- [2024.12] Our pre-print paper is released on arXiv.

## Requirements
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/) (If use A100)
- transformers >= 4.42.3
- pycocoevalcap >= 1.2

A suitable [conda](https://conda.io/) environment named `UniSoccer` can be created and activated with:

```
conda env create -f environment.yaml
conda activate UniSoccer
```

## Train

To be updated soon...

## Inference

To be updated soon...


## Citation
If you use this code and data for your research or project, please cite:

	@misc{rao2024unisoccer,
            title   = {Towards Universal Soccer Video Understanding},
            author  = {Rao, Jiayuan and Wu, Haoning and Jiang, Hao and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
            journal = {arXiv preprint arXiv:2412.01820},
            year    = {2024},
      }

## TODO
- [x] Release Paper
- [ ] Release Checkpoints
- [ ] Release Dataset
- [ ] Code of Visual Encoder Pretraining
- [ ] Code of Downstream Tasks
- [ ] Code of Inference
- [ ] Code of Evaluation


## Acknowledgements
Many thanks to the code bases from [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) and [MatchTime](https://github.com/jyrao/MatchTime), and source data from [SoccerNet-Caption](https://arxiv.org/abs/2304.04565) and [MatchTime](https://github.com/jyrao/MatchTime).


## Contact
If you have any questions, please feel free to contact jy_rao@sjtu.edu.cn or haoningwu3639@gmail.com.
