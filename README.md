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

## Data
- Librispeech: /n/rd7/librispeech/
- WSJ: /n/rd7/wsj/
- CHiME3: /n/rd7/chime/CHiME3


## Run training 
```
PYTHONPATH=$(pwd) gpujob -d 0 python train.py --config-name=pretrain_mask-based-wpd_doa-aware-lstm_LibriMixDemandDual2sec7ch
```

## Run adaptation evaluation 
```
# w/o adaptation 

PYTHONPATH=$(pwd) gpujob -d 1 /n/work3/fujita/miniconda3/envs/nbf/bin/python train.py --config-name=finetune_mask-based-wpd_doa-aware-lstm_iter-wpe-fastmnmf-doaest_LibriMixDemandTest name='n_spks\=2_rt60\=0.5_noise-snr\=30.0' id=17 testonly=true


# w/ adaptation 

PYTHONPATH=$(pwd) gpujob -d 1 /n/work3/fujita/miniconda3/envs/nbf/bin/python train.py --config-name=finetune_mask-based-wpd_doa-aware-lstm_iter-wpe-fastmnmf-doaest_LibriMixDemandTest name='n_spks\=4_rt60\=0.5_noise-snr\=30.0' id=35 data_module.total_s=240
```