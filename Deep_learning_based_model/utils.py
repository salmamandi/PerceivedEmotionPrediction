import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

def load_labels(label_file):

    data = pd.read_csv(label_file)

    df = data[["movie_clip", "valence", "arousal"]].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))

    df[["valence", "arousal"]] = scaler.fit_transform(
        df[["valence", "arousal"]]
    )

    label_dict = {}

    for _, row in df.iterrows():

        clip = row["movie_clip"]

        label_dict[clip] = (row["valence"], row["arousal"])

    return label_dict

def compute_metrics(pred, gt):

    mse = np.mean((pred - gt) ** 2)

    mae = np.mean(np.abs(pred - gt))

    mean_pred = np.mean(pred)
    mean_gt = np.mean(gt)

    cov = np.mean((pred - mean_pred) * (gt - mean_gt))

    var_pred = np.var(pred)
    var_gt = np.var(gt)

    ccc = (2 * cov) / (var_pred + var_gt + (mean_pred - mean_gt) ** 2 + 1e-8)

    return mse, mae, ccc

def compute_classification_metrics(gt, pred):

    acc = accuracy_score(gt, pred)
    prec = precision_score(gt, pred, average="macro",zero_division=0)
    rec = recall_score(gt, pred, average="macro",zero_division=0)
    f1 = f1_score(gt, pred,average="macro", zero_division=0)

    return acc, prec, rec, f1

def bin_value_binary(x,thres):
    
    if x <= thres:     
        return 0
    else:
        return 1  
    
def load_binary_labels(label_file,val_thresh_file,aro_thresh_file):

    data = pd.read_csv(label_file)
    val_thres_data=pd.read_csv(val_thresh_file)
    aro_thres_data=pd.read_csv(aro_thresh_file)
    val_movie_df=val_thres_data.loc[:,["movie_name","mid_val"]]
    aro_movie_df=aro_thres_data.loc[:,["movie_name","mid_val"]]

    df = data[["movie_clip", "valence", "arousal"]].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))

    df[["valence", "arousal"]] = scaler.fit_transform(
        df[["valence", "arousal"]]
    )



    label_dict = {}

    for _, row in df.iterrows():

        clip = row["movie_clip"]
        print(f"clip name:{clip}")
        movie_name=clip.split("_")[0]
        if(movie_name=="MeBeforeYou"):
            continue
        print(f"movie name:{movie_name}")
        val_thres=val_movie_df[val_movie_df["movie_name"]==movie_name]["mid_val"].round(2).values[0]
        aro_thres=aro_movie_df[aro_movie_df["movie_name"]==movie_name]["mid_val"].round(2).values[0]
        val_bin = bin_value_binary(row["valence"], val_thres)
        aro_bin = bin_value_binary(row["arousal"], aro_thres)
        label_dict[clip] = (val_bin,aro_bin)

    return label_dict


def compute_mean_std(dataset):

    video_list = []
    audio_list = []

    for i in range(len(dataset)):
        video, audio, mask, _ = dataset[i]

        video_list.append(video)
        audio_list.append(audio)

    video_all = torch.cat(video_list, dim=0)
    audio_all = torch.cat(audio_list, dim=0)

    video_mean = video_all.mean(dim=0)
    video_std = video_all.std(dim=0)

    audio_mean = audio_all.mean(dim=0)
    audio_std = audio_all.std(dim=0)

    return video_mean, video_std, audio_mean, audio_std