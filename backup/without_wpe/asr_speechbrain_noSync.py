import csv
import json
import os
import re

from argparse import ArgumentParser

import numpy as np
import scipy.signal
import torch
import torchaudio

from speechbrain.dataio.preprocess import AudioNormalizer
from speechbrain.pretrained import EncoderDecoderASR


PRETRAINED_BASEDIR = "/n/work3/sekiguch/data_for_paper/IROS2022/speechbrain_ASR/"
MIC_INDEX = 1   # the top center microphone
SRC_INDEX = 0


""" The ASR system uses the pretrained model from Speechbrain
    Model:                                      Test clean WER: Test other WER:
    asr-crdnn-transformerlm-librispeech         2.90            8.51
    asr-transformer-transformerlm-librispeech   2.46            5.86
"""

if __name__ == "__main__":
    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument(
        "--wav_list", type=str, default=None)
    ARGS, _ = ARG_PARSER.parse_known_args()

    print(ARGS)

    with open(ARGS.wav_list, newline='') as f:
        reader = csv.reader(f)
        wav_list = list(reader)

    for model_net in ["transformer"]:
        model_name = "asr-{}-transformerlm-librispeech".format(model_net)
        asr_model = None
        for spk_id, transcript_file, test_wav_file, ref_mix_wav_file, \
                ref_json_file in wav_list:
            _tmp = transcript_file.split('/')
            _tmp[-1] = "hyp_noSync_{}_".format(model_net) + _tmp[-1]
            transcript_file = '/'.join(_tmp)

            if os.path.isfile(transcript_file):
                continue

            print("+>", transcript_file)
            with open(ref_json_file) as g:
                json_data = json.load(g)
            test_sig_SM, sr = torchaudio.load(
                test_wav_file, channels_first=False)
            # Index 0 contains the mixture
            np_mix_sig_S = test_sig_SM[:, 0].numpy()
            for utt_idx in json_data:
                if os.path.isfile(transcript_file):
                    with open(transcript_file, 'r') as f:
                        _tmp_spk_id_utt_idx = [
                            re.search('\([0-9]*_[0-9]*\)', e).group(0)
                            for e in f.readlines()]
                    if "({}_{})".format(spk_id, utt_idx) in \
                            _tmp_spk_id_utt_idx:
                        continue
                if asr_model is None:
                    asr_model = EncoderDecoderASR.from_hparams(
                        source=os.path.join(PRETRAINED_BASEDIR, model_name),
                        savedir=os.path.join("pretrained_models", model_name),
                        run_opts={"device": "cuda"},)
                    # The default AudioNormalizer uses mix="avg-to-mono" '''
                    asr_model.audio_normalizer = AudioNormalizer(
                        sample_rate=16000, mix="keep")
                str_idx = json_data[utt_idx]['start']
                end_idx = str_idx + json_data[utt_idx]['length']
                print("+->", spk_id, test_wav_file, ref_json_file)
                _test_sig_SM = test_sig_SM[str_idx:end_idx]
                # Do resampling when `sr != sample_rate`
                _test_sig_SM = asr_model.audio_normalizer(_test_sig_SM, sr)
                # Select 1 source
                sig_S = _test_sig_SM[:, SRC_INDEX]
                # Fake a batch:
                batch = sig_S.unsqueeze(0).cuda()
                rel_length = torch.tensor([1.0]).cuda()
                pred_words, pred_tokens = asr_model.transcribe_batch(
                    batch, rel_length)
                print(spk_id, utt_idx, pred_words)
                assert len(pred_words) == 1
                with open(transcript_file, 'a') as f:
                    _ = f.write("{} ({}_{})\n".format(
                        pred_words[0], spk_id, utt_idx))
