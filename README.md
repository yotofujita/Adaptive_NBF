# Offline adaptive neural beamforming 

## Environment
```bash
conda create -y -c default -c conda-forge -c pytorch -c nvidia -n nbf \
	pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 deepspeed==0.15.2
conda install -y \
	chardet numpy opt_einsum pysoundfile tqdm scipy resampy h5py pytorch-lightning \
	typing-extensions cupy cudnn nccl tensorboard 
pip install speechbrain 
```

## Run training 
```
PYTHONPATH=$(pwd) python train.py --config-name=pretrain_mask-based-wpd_doa-aware-lstm_LibriMixDemandDual2sec7ch
```

## Run adaptation evaluation 
```
# w/o adaptation 

PYTHONPATH=$(pwd) python train.py --config-name=finetune_mask-based-wpd_doa-aware-lstm_iter-wpe-fastmnmf-doaest_LibriMixDemandTest name='n_spks\=2_rt60\=0.5_noise-snr\=30.0' id=17 testonly=true


# w/ adaptation 

PYTHONPATH=$(pwd) python train.py --config-name=finetune_mask-based-wpd_doa-aware-lstm_iter-wpe-fastmnmf-doaest_LibriMixDemandTest name='n_spks\=4_rt60\=0.5_noise-snr\=30.0' id=35 data_module.total_s=240
```

### Eval fastmnmf
PYTHONPATH=$(pwd) /home/fujita/miniconda3/envs/nbf/bin/python eval.py --config-name=eval_iter-wpe-fastmnmf-oracle_LibriMixDemandTest name='noise-snr\=30.0' id=0