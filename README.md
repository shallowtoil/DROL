# DROL 
This is the repo for paper "Discriminative and Robust Online Learning for Siamese Visual Tracking" [[paper](https://arxiv.org/abs/1909.02959)] [[results](https://drive.google.com/open?id=1iXtaxr1zkWvKf6AAMwN98ymaId9ixU4z)], presented as poster at AAAI 2020.

## Introduction

The proposed Discriminative and Robust Online Learning (DROL) module is designed to working with a variety of off-the-shelf siamese trackers. Our method is extensively evaluated over serveral mainstream benchmarks and is believed to induce a consistant performance gain over the given baseline. Such models include but not limited to, as paper evaluated:

- [SiamRPN++](https://arxiv.org/abs/1812.11703) (DROL-RPN)
- [SiamMask](https://arxiv.org/abs/1812.05050) (DROL-MASK)
- [SiamFC](https://arxiv.org/abs/1606.09549) (DROL-FC)

## Model Zoo

The corresponding offline-trained models are availabe at [PySOT Model Zoo](MODEL_ZOO.md).


## Get Started

### Installation

 - Please find installation instructions for PyTorch and PySOT in [`INSTALL.md`](INSTALL.md).
 - Add DROL to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/drol:$PYTHONPATH
```

### Download models
Download models in [PySOT Model Zoo](MODEL_ZOO.md) and put the model.pth in the corresponding  directory in experiments.

### Test tracker
```bash
cd experiments/siamrpn_r50_l234_dwxcorr
python -u ../../tools/test.py 	\
	--snapshot model.pth 	\ # model path
	--dataset VOT2018 	\ # dataset name
	--config config.yaml	  # config file
```

### Eval tracker
assume still in experiments/siamrpn_r50_l234_dwxcorr_8gpu
``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset VOT2018        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

### Others
 - For `DROL-RPN`, we have seperate config file thus each own experiment file folder for `vot`/`votlt`/`otb`/`others`, where `vot` is used for `VOT-20XX-baseline` benchmark, `votlt` for `VOT-20XX-longterm` benchmark, `otb` for `OTB2013/15` benchmark, and `others` is default setting thus for all the other benchmarks, including but not limited to `LaSOT`/`TrackingNet`/`UAV123`.
 - For `DROL-FC/DROL-Mask`, only experiments on `vot/otb` are evaluated as described in the paper. Similar to the repo of `PySOT`, we use config file for `vot` as default setting.

 - Since this repo is a grown-up modification of [PySOT](https://github.com/STVIR/pysot), we recommend to refer to PySOT for more technical issues.


## References
- Jinghao Zhou, Peng Wang, Haoyang Sun, '[Discriminative and Robust Online Learning For Siamese Visual Tracking](http://arxiv.org/abs/1909.02959)', Proc. AAAI Conference on Artificial Intelligence (AAAI), 2020.

### Ackowledgement
- [pysot](https://github.com/STVIR/pysot)
- [pytracking](https://github.com/visionml/pytracking)