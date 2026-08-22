import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SceneFeatureDataset(Dataset):

    def __init__(self, video_root, audio_root,normalize=False, video_mean=None, video_std=None,audio_mean=None, audio_std=None):

        self.samples = []
        self.video_root=video_root
        self.audio_root=audio_root
        self.normalize=normalize
        self.video_mean = video_mean
        self.video_std = video_std
        self.audio_mean = audio_mean
        self.audio_std = audio_std

        for movie in os.listdir(video_root):

            video_movie = os.path.join(video_root, movie)
            audio_movie = os.path.join(audio_root, movie)

            for clip in os.listdir(video_movie):

                v_clip = os.path.join(video_movie, clip)
                a_clip = os.path.join(audio_movie, clip)

                if not os.path.exists(a_clip):
                      continue
                
                # check if clip contains npy files
                v_files = [f for f in os.listdir(v_clip) if f.endswith(".npy")]
                a_files = [f for f in os.listdir(a_clip) if f.endswith(".npy")]

                if len(v_files) == 0 or len(a_files) == 0:
                    continue

                
                self.samples.append((movie, v_clip, a_clip))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        movie, video_clip, audio_clip = self.samples[idx]
        clip_name = os.path.basename(video_clip)
        print("clip name:",clip_name)

        video_features = []
        audio_features = []

        scene_files = sorted([f for f in os.listdir(video_clip) if f.endswith(".npy")])

        for f in scene_files:

            v = np.load(os.path.join(video_clip, f))   # (3,768)
            a = np.load(os.path.join(audio_clip, f))   # (1,1024)

            # Aggregate scene features
            v = v.mean(axis=0)        # (768)
            a = a.mean(axis=0)          # (1024)

            video_features.append(v)
            audio_features.append(a)

        video_features = torch.tensor(np.stack(video_features), dtype=torch.float32)
        audio_features = torch.tensor(np.stack(audio_features), dtype=torch.float32)
        if self.normalize:
            video_features = (video_features - self.video_mean) / (self.video_std + 1e-8)
            audio_features = (audio_features - self.audio_mean) / (self.audio_std + 1e-8)
        
        print("--printing video feature status--")
        print(self.normalize, self.video_mean is None)
        return video_features, audio_features, clip_name, movie
    
    
def collate_fn(batch):

    videos = [item[0] for item in batch]
    audios = [item[1] for item in batch]
    clip_names = [item[2] for item in batch]

    scene_lengths = [v.shape[0] for v in videos]

    max_len = max(scene_lengths)

    video_dim = videos[0].shape[1]
    audio_dim = audios[0].shape[1]

    padded_videos = []
    padded_audios = []
    masks = []

    for v, a in zip(videos, audios):

        pad_len = max_len - v.shape[0]

        if pad_len > 0:

            v_pad = torch.zeros(pad_len, video_dim)
            a_pad = torch.zeros(pad_len, audio_dim)

            v = torch.cat([v, v_pad], dim=0)
            a = torch.cat([a, a_pad], dim=0)

        padded_videos.append(v)
        padded_audios.append(a)

        mask = torch.zeros(max_len)
        mask[:v.shape[0]-pad_len] = 1
        masks.append(mask)

    padded_videos = torch.stack(padded_videos)
    padded_audios = torch.stack(padded_audios)
    masks = torch.stack(masks)

    return padded_videos, padded_audios, masks, clip_names