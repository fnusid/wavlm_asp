import torch 
import torchaudio
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json 
import os
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence




class MyLibri2Mix(Dataset):
    def __init__(self, metadata_path, speaker_map_path, num_speakers=2, sampling_rate=16000):
        super().__init__()

        self.metadata = pd.read_csv(metadata_path)


        self.num_speakers = num_speakers

        with open(speaker_map_path, 'r') as f:
            self.speaker_to_index = json.load(f)
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
   
        row = self.metadata.iloc[idx] #mixture_ID,mixture_path,source_1_path,source_2_path,speaker_1_ID,speaker_2_ID,length

        mix_path = row['mixture_path']

        mix_audio, _ = torchaudio.load(mix_path)

        source_audios = []
        for i in range(self.num_speakers):
            s_path = row[f"source_{i+1}_path"]
            s_audio,_ = torchaudio.load(s_path)
            source_audios.append(s_audio)
        
        sources_tensor = torch.cat(source_audios, dim = 0) #[B,2,T]

        speaker_indices = []
        for i in range(self.num_speakers):
            speaker_id = str(row[f"speaker_{i+1}_ID"])
            if speaker_id in self.speaker_to_index:
                index = self.speaker_to_index[speaker_id]
            else:
                index = int(speaker_id)


            speaker_indices.append(index)
        
        labels_tensor = torch.tensor(speaker_indices, dtype=torch.long)

        return mix_audio.squeeze(0), sources_tensor, labels_tensor



def librimix_collate(batch):

    mix, source, labels = zip(*batch)
    '''
    mix: [B, T]
    source: [B, 2, T]

    '''
    mix_padded = pad_sequence(mix, batch_first=True, padding_value=0.0)

    #permute the sources to [T, 2]

    sources_permuted = [s.permute(1,0) for s in source]
    sources_padded = pad_sequence(sources_permuted, batch_first=True, padding_value = 0.0)

    sources_padded = sources_padded.permute(0,2,1)

    labels = torch.stack(labels)

    return mix_padded, sources_padded, labels



class LibriMixDataModule(pl.LightningDataModule):

    def __init__(self, data_root, speaker_map_path, 
                batch_size=32,
                num_workers=0,
                num_speakers=2,
                sample_rate=16000):
        
        super().__init__()
        self.data_root = data_root
        self.speaker_map_path = speaker_map_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_speakers = num_speakers
        self.sampling_rate = sample_rate

        self.persistent_workers = True if self.num_workers > 0 else False

        self.base_data_path = os.path.join(self.data_root, f"Libri{self.num_speakers}Mix", "wav16k", "min")

        self.metadata_path = os.path.join(self.base_data_path, "metadata")


    def setup(self, stage=None):



        train_meta = os.path.join(self.metadata_path, "mixture_train-100_mix_clean.csv")
        val_meta = os.path.join(self.metadata_path, "mixture_dev_mix_clean.csv")

        self.train_dataset = MyLibri2Mix(
            metadata_path=train_meta,
            speaker_map_path=self.speaker_map_path,
            num_speakers=self.num_speakers
        )
        self.val_dataset = MyLibri2Mix(
            metadata_path=val_meta,
            speaker_map_path=self.speaker_map_path,
            num_speakers=self.num_speakers
        )



    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=librimix_collate,
            persistent_workers=self.persistent_workers
        )


if __name__ == '__main__':
\
    # breakpoint()
    data_root = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix"
    speaker_map_path = "/mnt/disks/data/datasets/Datasets/LibriMix/LibriMix/Libri2Mix/wav16k/min/metadata/train100_mapping.json"

    dataset = LibriMixDataModule(
        data_root, speaker_map_path, 
                batch_size=4,
                num_workers=0,
                num_speakers=2,
                sample_rate=16000
    )
    dataset.setup()
    dl = dataset.train_dataloader()

    for i, (mix, source, labels) in enumerate(dl):
        breakpoint()
        for b in range(mix.shape[0]):
            torchaudio.save(f"/home/sidharth./codebase/streamable_architecture/dataset_sanity_check/mix_{b}.wav", mix[b].unsqueeze(0), 16000)
            torchaudio.save(f"/home/sidharth./codebase/streamable_architecture/dataset_sanity_check/source1_{b}.wav", source[b,0,:].unsqueeze(0), 16000)
            torchaudio.save(f"/home/sidharth./codebase/streamable_architecture/dataset_sanity_check/source2_{b}.wav", source[b,1,:].unsqueeze(0), 16000)


        breakpoint()
