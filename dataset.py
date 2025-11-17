import torch 
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json 
import os
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import random


'''
/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/train-100_mapping.json
'''

class Librispeech(Dataset):
    def __init__(self, txt_file, speaker_map_path=None, sampling_rate=16000):
        super().__init__()

        with open(txt_file, 'r') as f:
            self.metadata = [line.strip() for line in f if line.strip()]
        
        random.shuffle(self.metadata)

        
        self.sampling_rate = sampling_rate
        if speaker_map_path is not None:
            with open(speaker_map_path, 'r') as f:
                self.speaker_to_index = json.load(f)
        else:
            self.speaker_to_index = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        audio_path = self.metadata[idx] #mixture_ID,mixture_path,source_1_path,source_2_path,speaker_1_ID,speaker_2_ID,length

        labels = int(audio_path.split('/')[-3])


        audio, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            audio = resampler(audio)
            sr = self.sampling_rate
        
        if audio.ndim > 1:
            audio = audio.squeeze(0)
        if str(labels) in self.speaker_to_index:
            label_index = self.speaker_to_index[str(labels)]
        else:
            label_index = labels
        
        labels_tensor = torch.tensor(label_index, dtype=torch.long)

        return audio, labels_tensor


def librimix_collate(batch):

    audio, labels = zip(*batch)
    '''
    audio: [B, T]

    '''
    audio_padded = pad_sequence(audio, batch_first=True, padding_value=0.0)


    labels = torch.stack(labels)

    return audio_padded, labels



class LibriDataModule(pl.LightningDataModule):

    def __init__(self, data_root, train_speaker_map_path,
                train_batch_size=32,
                val_batch_size=8,
                num_workers=0,
                sample_rate=16000):
        
        super().__init__()
        self.data_root = data_root #/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/
        self.train_speaker_map_path = train_speaker_map_path
        # self.val_speaker_map_path = val_speaker_map_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.sampling_rate = sample_rate

        self.persistent_workers = True if self.num_workers > 0 else False



    def setup(self, stage=None):


        train_meta = os.path.join(self.data_root, "libri_train100.txt")
        val_meta = os.path.join(self.data_root, "libri_dev_clean.txt")


        self.train_dataset = Librispeech(
            txt_file=train_meta,
            speaker_map_path=self.train_speaker_map_path,

        )
        self.val_dataset = Librispeech(
            txt_file=val_meta,
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers
        )

if __name__ == "__main__":

    dm = LibriDataModule(
        data_root="/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech",
        train_speaker_map_path="/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/LibriSpeech/train-100_mapping.json",
        batch_size=4,
        num_workers=0,
        sample_rate=16000,
    )

    dm.setup("fit")

    loader = dm.train_dataloader()

    audio_batch, labels = next(iter(loader))

    save_dir = "/home/sidharth./codebase/wavlm_single_embedding/dataset_samples/"
    os.makedirs(save_dir, exist_ok=True)

    for i, (audio, label) in enumerate(zip(audio_batch, labels)):
        # Remove padding (keep only nonzero values)
        valid_len = (audio != 0).nonzero(as_tuple=True)[0].max().item() + 1
        audio = audio[:valid_len]

        out_path = os.path.join(save_dir, f"sample_{i}_spk_{label.item()}.wav")
        torchaudio.save(out_path, audio.unsqueeze(0), 16000)

        print(f"Saved: {out_path}")
