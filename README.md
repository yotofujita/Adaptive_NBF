# Offline adaptive neural beamforming 

## Environment
```bash
conda create -y -c default -c conda-forge -c pytorch -c nvidia -n nbf \
	pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1
conda install -y \
	chardet numpy opt_einsum pysoundfile tqdm scipy resampy h5py pytorch-lightning \
	typing-extensions cupy cudnn nccl tensorboard 
pip install speechbrain 
```
