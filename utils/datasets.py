import os
import numpy as np
import librosa
import torch
import torch.utils.data as torch_data

from utilities import read_audio
import config
import mu_law


class VCTKDataset(torch_data.Dataset):
    def __init__(self, dataset_dir, data_type):
        
        audios_dir = os.path.join(dataset_dir, 'wav48')
        
        # Hold out persons for validate
        validate_persons = ['p225', 'p234', 'p246', 'p256', 'p266', 
                            'p276', 'p287', 'p301', 'p312', 'p330']
        
        self.sample_rate = config.sample_rate
        self.quantize_bins = config.quantize_bins
        
        audio_names = []
    
        for root, dirs, files in os.walk(audios_dir):
            for name in files:
                if name.endswith('.wav'):
                    audio_name = os.path.join(root, name)
                    audio_names.append(audio_name)
                    
        audio_names = sorted(audio_names)
        
        if data_type == 'train':
            audio_names = [name for name in audio_names if not 
                self.name_in_persons(name, validate_persons)]
            
        elif data_type == 'validate':
            audio_names = [name for name in audio_names if 
                self.name_in_persons(name, validate_persons)]        
            
        self.audio_names = audio_names
        
    
    def __getitem__(self, index):
        
        # Read audio
        (audio, fs) = read_audio(self.audio_names[index], target_fs=self.sample_rate)
        audio /= max(1., np.max(np.abs(audio)))
        
        # Cut silence
        frame_length = 2048
        hop_length = 512
        threshold = 0.01
        energy = librosa.feature.rmse(audio, frame_length=frame_length, 
                                      hop_length=hop_length, center=True)[0]
        frames = np.nonzero(energy > threshold)[0]
        indices = librosa.core.frames_to_samples(frames, hop_length=hop_length)
        
        # Abandon too short clips
        if len(indices) < 2:
            audio = np.zeros(10000)
        else:
            audio = audio[indices[0] : indices[-1]]
        
        if len(audio) < 10000:
            audio = np.zeros(10000)
        else:
            audio = audio[0 : 70000]    # To not over use 12 GB GPU RAM
        
        # Mu-law
        _mulaw = mu_law.MuLaw(mu=self.quantize_bins)
        _quantize = mu_law.Quantize(quantize=self.quantize_bins)
        
        audio = _mulaw.transform(audio)
        audio = _quantize.transform(audio)
        audio = torch.LongTensor(audio)
        
        # Get global condition
        self.audio_names[index]
        global_condition = int(self.audio_names[index].split('/')[-1][1:4])
        global_condition = torch.tensor(global_condition)

        return audio, global_condition
        
    def __len__(self):
        return len(self.audio_names)

    def name_in_persons(self, name, validate_persons):
        for person in validate_persons:
            if person in name:
                return True
        return False
        
        
def get_dataset(dataset):
    if dataset == 'vctk':
        return VCTKDataset
        
    else:
        raise Exception('Incorrect dataset!')