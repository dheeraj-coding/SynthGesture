import datetime
import logging
import math
import os
import pickle
import random
import sys
import copy

import librosa
import soundfile as sf
import lmdb
import numpy as np
import time

import pyarrow
import torch
from torch.utils.data import DataLoader

import utils
from data_loader.lmdb_data_loader_expressive import SpeechMotionDataset, default_collate_fn, word_seq_collate_fn
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from train_expressive import evaluate_testset
from utils.data_utils_expressive import extract_melspectrogram, remove_tags_marks, convert_dir_vec_to_pose
from utils.train_utils_expressive import create_video_and_save, set_logger

from data_loader.data_preprocessor_expressive import DataPreprocessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_gestures_hierarchy(args, g1, g2, g3, g4, g5, g6, audio_encoder, lang_model, audio, words,
                                target_1, target_2, target_3, target_4, target_5, target_6, audio_sr=16000, vid=None,
                                fade_out=False):
    out_list = []
    n_frames = args.n_poses
    clip_length = len(audio) / audio_sr

    # pre seq
    pre_seq = torch.zeros((1, n_frames, len(args.mean_dir_vec) + 1))

    sr = 16000
    spectrogram = None
    if True:
        # audio to spectrogram
        spectrogram = extract_melspectrogram(audio, sr)

    # divide into synthesize units and do synthesize
    unit_time = args.n_poses / args.motion_resampling_framerate
    stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
    spectrogram_sample_length = utils.data_utils_expressive.calc_spectrogram_length_from_motion_length(args.n_poses,
                                                                                                       args.motion_resampling_framerate)
    audio_sample_length = int(unit_time * audio_sr)
    end_padding_duration = 0

    # prepare speaker input
    if args.z_type == 'speaker':
        if not vid:
            vid = random.randrange(g1.z_obj.n_words)
        # print('vid:', vid)
        vid = torch.LongTensor([vid]).to(device)
    else:
        vid = None

    # print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

    out_dir_vec = None
    start = time.time()
    for i in range(0, num_subdivision):
        start_time = i * stride_time
        end_time = start_time + unit_time

        # prepare spectrogram input
        in_spec = None
        if True:
            # prepare spec input
            audio_start = math.floor(start_time / clip_length * spectrogram.shape[0])
            audio_end = audio_start + spectrogram_sample_length
            in_spec = spectrogram[:, audio_start:audio_end]
            in_spec = torch.from_numpy(in_spec).float().unsqueeze(0).to(device)
            # print(in_spec.shape)

        # prepare audio input
        audio_start = math.floor(start_time / clip_length * len(audio))
        audio_end = audio_start + audio_sample_length
        in_audio = audio[audio_start:audio_end]
        if len(in_audio) < audio_sample_length:
            if i == num_subdivision - 1:
                end_padding_duration = audio_sample_length - len(in_audio)
            in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
        in_audio = torch.from_numpy(copy.copy(in_audio)).unsqueeze(0).to(device).float()

        # prepare text input
        word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time, end_time=end_time)
        extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
        word_indices = np.zeros(len(word_seq) + 2)
        word_indices[0] = lang_model.SOS_token
        word_indices[-1] = lang_model.EOS_token
        frame_duration = (end_time - start_time) / n_frames
        for w_i, word in enumerate(word_seq):
            # print(word[0], end= ', ')
            idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
            extended_word_indices[idx] = lang_model.get_word_index(word[0])
            word_indices[w_i + 1] = lang_model.get_word_index(word[0])
        # print(' ')
        in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to(device)
        in_text = torch.LongTensor(word_indices).unsqueeze(0).to(device)

        # prepare pre seq
        if i > 0:
            pre_seq[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
            pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            target_1[:, 0:args.n_pre_poses, :] = torch.cat(
                (out_dir_vec[:, -args.n_pre_poses:, :3 * 3], out_dir_vec[:, -args.n_pre_poses:, -5 * 3:]), dim=2)
            target_2[:, 0:args.n_pre_poses, :] = torch.cat((out_dir_vec[:, -args.n_pre_poses:, :4 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 20 * 3:21 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, -5 * 3:]), dim=2)
            target_3[:, 0:args.n_pre_poses, :] = torch.cat((out_dir_vec[:, -args.n_pre_poses:, :5 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 20 * 3:22 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, -5 * 3:]), dim=2)
            target_4[:, 0:args.n_pre_poses, :] = torch.cat((out_dir_vec[:, -args.n_pre_poses:, :6 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 8 * 3:9 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 11 * 3:12 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 14 * 3:15 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 17 * 3:18 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 20 * 3:23 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 25 * 3:26 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 28 * 3:29 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 31 * 3:32 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 34 * 3:35 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, -5 * 3:]), dim=2)
            target_5[:, 0:args.n_pre_poses, :] = torch.cat((out_dir_vec[:, -args.n_pre_poses:, :7 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 8 * 3:10 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 11 * 3:13 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 14 * 3:16 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 17 * 3:19 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 20 * 3:24 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 25 * 3:27 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 28 * 3:30 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 31 * 3:33 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, 34 * 3:36 * 3],
                                                            out_dir_vec[:, -args.n_pre_poses:, -5 * 3:]), dim=2)
            target_6[:, 0:args.n_pre_poses, :] = out_dir_vec[:, -args.n_pre_poses:, :]
        pre_seq = pre_seq.float().to(device)
        pre_seq_partial = pre_seq[0, 0:args.n_pre_poses, :-1].unsqueeze(0)

        # synthesize
        _, _, _, _, linear_blend_feat = audio_encoder(in_spec, vid)

        pre_seq_1 = target_1.new_zeros((target_1.shape[0], target_1.shape[1], target_1.shape[2] + 1))
        pre_seq_1[:, 0:args.n_pre_poses, :-1] = target_1[:, 0:args.n_pre_poses, :]
        pre_seq_1[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        out_dir_vec_1, *_ = g1(pre_seq_1.float(), in_text_padded, linear_blend_feat[0],
                               vid)  # out shape (batch x seq x dim)

        pre_seq_2 = target_2.new_zeros((target_2.shape[0], target_2.shape[1], target_2.shape[2] + 1))
        pre_seq_2[:, 0:args.n_pre_poses, :-1] = target_2[:, 0:args.n_pre_poses, :]
        pre_seq_2[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_2[:, args.n_pre_poses:, :3 * 3] = out_dir_vec_1[:, args.n_pre_poses:, :3 * 3]
        pre_seq_2[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_1[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_2, *_ = g2(pre_seq_2.float(), in_text_padded, linear_blend_feat[1],
                               vid)  # out shape (batch x seq x dim)

        pre_seq_3 = target_3.new_zeros((target_3.shape[0], target_3.shape[1], target_3.shape[2] + 1))
        pre_seq_3[:, 0:args.n_pre_poses, :-1] = target_3[:, 0:args.n_pre_poses, :]
        pre_seq_3[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_3[:, args.n_pre_poses:, :4 * 3] = out_dir_vec_2[:, args.n_pre_poses:, :4 * 3]
        pre_seq_3[:, args.n_pre_poses:, 5 * 3:6 * 3] = out_dir_vec_2[:, args.n_pre_poses:, 4 * 3:5 * 3]
        pre_seq_3[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_2[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_3, *_ = g3(pre_seq_3.float(), in_text_padded, linear_blend_feat[2],
                               vid)  # out shape (batch x seq x dim)

        pre_seq_4 = target_4.new_zeros((target_4.shape[0], target_4.shape[1], target_4.shape[2] + 1))
        pre_seq_4[:, 0:args.n_pre_poses, :-1] = target_4[:, 0:args.n_pre_poses, :]
        pre_seq_4[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_4[:, args.n_pre_poses:, :5 * 3] = out_dir_vec_3[:, args.n_pre_poses:, :5 * 3]
        pre_seq_4[:, args.n_pre_poses:, 10 * 3:12 * 3] = out_dir_vec_3[:, args.n_pre_poses:, 5 * 3:7 * 3]
        pre_seq_4[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_3[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_4, *_ = g4(pre_seq_4.float(), in_text_padded, linear_blend_feat[3],
                               vid)  # out shape (batch x seq x dim)

        pre_seq_5 = target_5.new_zeros((target_5.shape[0], target_5.shape[1], target_5.shape[2] + 1))
        pre_seq_5[:, 0:args.n_pre_poses, :-1] = target_5[:, 0:args.n_pre_poses, :]
        pre_seq_5[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_5[:, args.n_pre_poses:, :6 * 3] = out_dir_vec_4[:, args.n_pre_poses:, :6 * 3]
        pre_seq_5[:, args.n_pre_poses:, 7 * 3:8 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 6 * 3:7 * 3]
        pre_seq_5[:, args.n_pre_poses:, 9 * 3:10 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 7 * 3:8 * 3]
        pre_seq_5[:, args.n_pre_poses:, 11 * 3:12 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 8 * 3:9 * 3]
        pre_seq_5[:, args.n_pre_poses:, 13 * 3:14 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 9 * 3:10 * 3]
        pre_seq_5[:, args.n_pre_poses:, 15 * 3:18 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 10 * 3:13 * 3]
        pre_seq_5[:, args.n_pre_poses:, 19 * 3:20 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 13 * 3:14 * 3]
        pre_seq_5[:, args.n_pre_poses:, 21 * 3:22 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 14 * 3:15 * 3]
        pre_seq_5[:, args.n_pre_poses:, 23 * 3:24 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 15 * 3:16 * 3]
        pre_seq_5[:, args.n_pre_poses:, 25 * 3:26 * 3] = out_dir_vec_4[:, args.n_pre_poses:, 16 * 3:17 * 3]
        pre_seq_5[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_4[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec_5, *_ = g5(pre_seq_5.float(), in_text_padded, linear_blend_feat[4],
                               vid)  # out shape (batch x seq x dim)

        pre_seq_6 = target_6.new_zeros((target_6.shape[0], target_6.shape[1], target_6.shape[2] + 1))
        pre_seq_6[:, 0:args.n_pre_poses, :-1] = target_6[:, 0:args.n_pre_poses, :]
        pre_seq_6[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq_6[:, args.n_pre_poses:, :7 * 3] = out_dir_vec_5[:, args.n_pre_poses:, :7 * 3]
        pre_seq_6[:, args.n_pre_poses:, 8 * 3:10 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 7 * 3:9 * 3]
        pre_seq_6[:, args.n_pre_poses:, 11 * 3:13 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 9 * 3:11 * 3]
        pre_seq_6[:, args.n_pre_poses:, 14 * 3:16 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 11 * 3:13 * 3]
        pre_seq_6[:, args.n_pre_poses:, 17 * 3:19 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 13 * 3:15 * 3]
        pre_seq_6[:, args.n_pre_poses:, 20 * 3:24 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 15 * 3:19 * 3]
        pre_seq_6[:, args.n_pre_poses:, 25 * 3:27 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 19 * 3:21 * 3]
        pre_seq_6[:, args.n_pre_poses:, 28 * 3:30 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 21 * 3:23 * 3]
        pre_seq_6[:, args.n_pre_poses:, 31 * 3:33 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 23 * 3:25 * 3]
        pre_seq_6[:, args.n_pre_poses:, 34 * 3:36 * 3] = out_dir_vec_5[:, args.n_pre_poses:, 25 * 3:27 * 3]
        pre_seq_6[:, args.n_pre_poses:, -5 * 3:] = out_dir_vec_5[:, args.n_pre_poses:, -5 * 3:]
        out_dir_vec, *_ = g6(pre_seq_6.float(), in_text_padded, linear_blend_feat[5],
                             vid)  # out shape (batch x seq x dim)

        out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

        # smoothing motion transition
        if len(out_list) > 0:
            last_poses = out_list[-1][-args.n_pre_poses:]
            out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last 4 frames

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(out_seq)

    print('generation took {:.2} s'.format((time.time() - start) / num_subdivision))

    # aggregate results
    out_dir_vec = np.vstack(out_list)

    # additional interpolation for seq2seq
    if args.model == 'seq2seq':
        n_smooth = args.n_pre_poses
        for i in range(num_subdivision):
            start_frame = args.n_pre_poses + i * (args.n_poses - args.n_pre_poses) - n_smooth
            if start_frame < 0:
                start_frame = 0
                end_frame = start_frame + n_smooth * 2
            else:
                end_frame = start_frame + n_smooth * 3

            # spline interp
            y = out_dir_vec[start_frame:end_frame]
            x = np.array(range(0, y.shape[0]))
            w = np.ones(len(y))
            w[0] = 5
            w[-1] = 5

            coeffs = np.polyfit(x, y, 3)
            fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
            interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
            interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

            out_dir_vec[start_frame:end_frame] = interpolated_y

    # fade out to the mean pose
    if fade_out:
        n_smooth = args.n_pre_poses
        start_frame = len(out_dir_vec) - int(end_padding_duration / audio_sr * args.motion_resampling_framerate)
        end_frame = start_frame + n_smooth * 2
        if len(out_dir_vec) < end_frame:
            out_dir_vec = np.pad(out_dir_vec, [(0, end_frame - len(out_dir_vec)), (0, 0)], mode='constant')
        out_dir_vec[end_frame - n_smooth:] = np.zeros((len(args.mean_dir_vec)))  # fade out to mean poses

        # interpolation
        y = out_dir_vec[start_frame:end_frame]
        x = np.array(range(0, y.shape[0]))
        w = np.ones(len(y))
        w[0] = 5
        w[-1] = 5
        coeffs = np.polyfit(x, y, 2, w=w)
        fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
        interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
        interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

        out_dir_vec[start_frame:end_frame] = interpolated_y

    return out_dir_vec


def align_words(audio, text):
    # resample audio to 8K
    audio_8k = librosa.resample(audio, 16000, 8000)
    wave_file = 'output/temp.wav'
    sf.write(wave_file, audio_8k, 8000, 'PCM_16')

    # run gentle to align words
    aligner = gentle.ForcedAligner(gentle_resources, text, nthreads=2, disfluency=False,
                                   conservative=False)
    gentle_out = aligner.transcribe(wave_file, logging=logging)
    words_with_timestamps = []
    for i, gentle_word in enumerate(gentle_out.words):
        if gentle_word.case == 'success':
            words_with_timestamps.append([gentle_word.word, gentle_word.start, gentle_word.end])
        elif 0 < i < len(gentle_out.words) - 1:
            words_with_timestamps.append([gentle_word.word, gentle_out.words[i - 1].end, gentle_out.words[i + 1].start])

    return words_with_timestamps


def main(checkpoint_path, option, data_path, audio_file_path, text):
    args, g1, g2, g3, g4, g5, g6, audio_encoder, loss_fn, lang_model, speaker_model, pose_dim = utils.train_utils_expressive.load_checkpoint_hierarchy(
        checkpoint_path, device)

    g1 = g1.to(device)
    g2 = g2.to(device)
    g3 = g3.to(device)
    g4 = g4.to(device)
    g5 = g5.to(device)
    g6 = g6.to(device)
    audio_encoder = audio_encoder.to(device)

    mean_pose = np.array(args.mean_pose).squeeze()
    mean_dir_vec = np.array(args.mean_dir_vec).squeeze()

    vocab_cache_path = os.path.join(data_path, 'vocab_cache.pkl')

    with open(vocab_cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    if args.model == 'seq2seq':
        collate_fn = word_seq_collate_fn
    else:
        collate_fn = default_collate_fn

    random.seed()

    if option:
        n_generations = int(option)
    else:
        n_generations = 5

    n_saved = 0
    test_data_path = os.path.join(data_path, 'test')
    clip_duration_range = [5, 12]
    save_path = 'output/generation_results_expressive'
    lmdb_env = lmdb.open(test_data_path, readonly=True, lock=False)

    with lmdb_env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
        while n_saved < n_generations:
            # select video
            key = random.choice(keys)

            buf = txn.get(key)
            video = pyarrow.deserialize(buf)
            vid = video['vid']
            clips = video['clips']

            # select clip
            n_clips = len(clips)
            if n_clips == 0:
                continue
            clip_idx = random.randrange(n_clips)

            clip_poses = clips[clip_idx]['skeletons_3d']
            # clip_audio = clips[clip_idx]['audio_raw']
            clip_audio = librosa.load(audio_file_path, mono=True, sr=16000, res_type='kaiser_fast')
            # clip_words = clips[clip_idx]['words']
            clip_words = align_words(clip_audio, text)
            clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]

            clip_poses = utils.data_utils_expressive.resample_pose_seq(clip_poses, clip_time[1] - clip_time[0],
                                                                       args.motion_resampling_framerate)
            target_dir_vec = utils.data_utils_expressive.convert_pose_seq_to_dir_vec(torch.from_numpy(clip_poses))
            target_dir_vec = target_dir_vec.reshape(target_dir_vec.shape[0], -1)
            target_dir_vec -= torch.from_numpy(mean_dir_vec).float()

            tmp_target_dir_vec = target_dir_vec[:args.n_poses].unsqueeze(0).to(device)

            target_1 = torch.cat((tmp_target_dir_vec[:, :, :3 * 3], tmp_target_dir_vec[:, :, -5 * 3:]), dim=2)
            target_2 = torch.cat((tmp_target_dir_vec[:, :, :4 * 3], tmp_target_dir_vec[:, :, 20 * 3:21 * 3],
                                  tmp_target_dir_vec[:, :, -5 * 3:]), dim=2)
            target_3 = torch.cat((tmp_target_dir_vec[:, :, :5 * 3], tmp_target_dir_vec[:, :, 20 * 3:22 * 3],
                                  tmp_target_dir_vec[:, :, -5 * 3:]), dim=2)
            target_4 = torch.cat((tmp_target_dir_vec[:, :, :6 * 3], tmp_target_dir_vec[:, :, 8 * 3:9 * 3],
                                  tmp_target_dir_vec[:, :, 11 * 3:12 * 3], tmp_target_dir_vec[:, :, 14 * 3:15 * 3],
                                  tmp_target_dir_vec[:, :, 17 * 3:18 * 3], tmp_target_dir_vec[:, :, 20 * 3:23 * 3],
                                  tmp_target_dir_vec[:, :, 25 * 3:26 * 3], tmp_target_dir_vec[:, :, 28 * 3:29 * 3],
                                  tmp_target_dir_vec[:, :, 31 * 3:32 * 3], tmp_target_dir_vec[:, :, 34 * 3:35 * 3],
                                  tmp_target_dir_vec[:, :, -5 * 3:]), dim=2)
            target_5 = torch.cat((tmp_target_dir_vec[:, :, :7 * 3], tmp_target_dir_vec[:, :, 8 * 3:10 * 3],
                                  tmp_target_dir_vec[:, :, 11 * 3:13 * 3], tmp_target_dir_vec[:, :, 14 * 3:16 * 3],
                                  tmp_target_dir_vec[:, :, 17 * 3:19 * 3], tmp_target_dir_vec[:, :, 20 * 3:24 * 3],
                                  tmp_target_dir_vec[:, :, 25 * 3:27 * 3], tmp_target_dir_vec[:, :, 28 * 3:30 * 3],
                                  tmp_target_dir_vec[:, :, 31 * 3:33 * 3], tmp_target_dir_vec[:, :, 34 * 3:36 * 3],
                                  tmp_target_dir_vec[:, :, -5 * 3:]), dim=2)
            target_6 = tmp_target_dir_vec

            # check duration
            clip_duration = clip_time[1] - clip_time[0]
            if clip_duration < clip_duration_range[0] or clip_duration > clip_duration_range[1]:
                continue

            # synthesize
            for selected_vi in range(len(clip_words)):  # make start time of input text zero
                clip_words[selected_vi][1] -= clip_time[0]  # start time
                clip_words[selected_vi][2] -= clip_time[0]  # end time

            vid_idx = random.sample(range(0, speaker_model.n_words), 1)[0]
            out_dir_vec = generate_gestures_hierarchy(args, g1, g2, g3, g4, g5, g6, audio_encoder, lang_model,
                                                      clip_audio, clip_words,
                                                      target_1, target_2, target_3, target_4, target_5, target_6,
                                                      vid=vid_idx, fade_out=False)

            target_dir_vec = np.squeeze(target_dir_vec.detach().cpu().numpy())
            out_dir_vec = np.squeeze(out_dir_vec)

            # make a video
            sentence_words = []
            for word, _, _ in clip_words:
                sentence_words.append(word)
            sentence = ' '.join(sentence_words)

            os.makedirs(save_path, exist_ok=True)

            filename_prefix = '{}_{}_{}'.format(vid, vid_idx, clip_idx)
            filename_prefix_for_video = filename_prefix
            aux_str = '({}, time: {}-{})'.format(vid, str(datetime.timedelta(seconds=clip_time[0])),
                                                 str(datetime.timedelta(seconds=clip_time[1])))
            create_video_and_save(
                save_path, 0, filename_prefix_for_video, 0, target_dir_vec, out_dir_vec,
                mean_dir_vec, sentence, audio=clip_audio, aux_str=aux_str,
                clipping_to_shortest_stream=True, delete_audio_file=False)

            # save pkl
            out_dir_vec = out_dir_vec + mean_dir_vec
            out_poses = convert_dir_vec_to_pose(torch.from_numpy(out_dir_vec).to(device)).cpu().numpy()

            save_dict = {
                'sentence': sentence, 'audio': clip_audio.astype(np.float32),
                'out_dir_vec': out_dir_vec, 'out_poses': out_poses,
                'aux_info': '{}_{}_{}'.format(vid, vid_idx, clip_idx),
                'human_dir_vec': target_dir_vec + mean_dir_vec,
            }
            with open(os.path.join(save_path, '{}.pkl'.format(filename_prefix)), 'wb') as f:
                pickle.dump(save_dict, f)

            n_saved += 1


if __name__ == "__main__":
    checkpoint_path = sys.argv[1]
    data_path = sys.argv[2]
    n_samples = 2
    audio_file_path = sys.argv[3]
    text = sys.argv[4]

    main(checkpoint_path, n_samples, data_path, audio_file_path, text)
