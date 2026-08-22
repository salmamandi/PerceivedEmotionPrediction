import torch
from torch.utils.data import DataLoader
from Salma_perceive_emo_pred_model import config
from Salma_perceive_emo_pred_model.utils import load_binary_labels,compute_classification_metrics,bin_value_binary
from Salma_perceive_emo_pred_model.dataset import SceneFeatureDataset, collate_fn
#from Salma_perceive_emo_pred_model.model import PerceivedVA_Model
from Salma_perceive_emo_pred_model.lightweight_model import PerceivedVA_Model
import pandas as pd
from torch.utils.data import DataLoader, random_split
import os
import numpy as np

video_root = config.VIDEO_FEATURE_ROOT
audio_root = config.AUDIO_FEATURE_ROOT

movies = sorted(os.listdir(video_root))
dataset = SceneFeatureDataset(video_root, audio_root)

label_dict = load_binary_labels(config.LABEL_FILE,config.VALENCE_THRESH_FILE,config.AROUSAL_THRESH_FILE)
val_thresh_df = pd.read_csv(config.VALENCE_THRESH_FILE)
aro_thresh_df = pd.read_csv(config.AROUSAL_THRESH_FILE)


final_results=[]
all_acc_v = []
all_pre_v = []
all_recall_v = []
all_f1_v=[]

all_acc_a = []
all_pre_a = []
all_recall_a = []
all_f1_a=[]

with open("difficult_clips.txt", "r") as f:
    difficult_clips = set(line.strip() for line in f)
#print(difficult_clips)

for test_movie in movies:


    test_indices = []

    for i, sample in enumerate(dataset.samples):

        movie = sample[0]
        clip = sample[1]
        print(f"difficult clip name:{clip}")

        if clip in difficult_clips:
           print(f"skipping difficult clip:{clip}")
           continue


        if movie == test_movie:
            test_indices.append(i)

    test_subset = torch.utils.data.Subset(dataset, test_indices)

    video_mean = np.load("saved_data_normalize_variables/"+test_movie+"/video_mean.npy")
    video_std  = np.load("saved_data_normalize_variables/"+test_movie+"/video_std.npy")

    audio_mean = np.load("saved_data_normalize_variables/"+test_movie+"/audio_mean.npy")
    audio_std  = np.load("saved_data_normalize_variables/"+test_movie+"/audio_std.npy")

    dataset_norm = SceneFeatureDataset(
    video_root, audio_root,
    normalize=True,
    video_mean=video_mean,
    video_std=video_std,
    audio_mean=audio_mean,
    audio_std=audio_std
     )
    
    test_dataset = torch.utils.data.Subset(dataset_norm, test_subset.indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    model = PerceivedVA_Model(config.VIDEO_DIM, config.AUDIO_DIM)

    model.load_state_dict(torch.load(f"{config.MODEL_SAVE_PATH}/model_{test_movie}.pth", map_location=device))

    model.to(device)
    model.eval()
    pred_v_all = []
    pred_a_all = []
    gt_v_all = []
    gt_a_all = []

    val_thresh = val_thresh_df[val_thresh_df["movie_name"] == test_movie]["mid_val"].values[0]
    aro_thresh = aro_thresh_df[aro_thresh_df["movie_name"] == test_movie]["mid_val"].values[0]
    with torch.no_grad():

            for video, audio, mask, clip_names in test_loader:
                video = video.float().to(device)
                audio = audio.float().to(device)
                mask = mask.to(device)

                pred_v, pred_a, delta = model(video, audio, mask)

                for i, clip in enumerate(clip_names):

                    gt_v, gt_a = label_dict[clip]

                    pred_v_all.append(pred_v[i].item())
                    pred_a_all.append(pred_a[i].item())

                    gt_v_all.append(gt_v)
                    gt_a_all.append(gt_a)
    # Convert to numpy
   
    pred_v_all = np.array(pred_v_all)
    pred_a_all = np.array(pred_a_all)
    gt_v_all = np.array(gt_v_all)
    gt_a_all = np.array(gt_a_all)

    pred_v_bin = [bin_value_binary(v, val_thresh) for v in pred_v_all]
    pred_a_bin = [bin_value_binary(a, aro_thresh) for a in pred_a_all]

    acc_v, prec_v, rec_v, f1_v = compute_classification_metrics(gt_v_all, pred_v_bin)

    acc_a, prec_a, rec_a, f1_a = compute_classification_metrics(gt_a_all, pred_a_bin)

    print("\nResults for movie:", test_movie)

    print("Valence -> Accuracy:", acc_v,
        "Precision:", prec_v,
        "Recall:", rec_v,
        "F1:", f1_v)

    print("Arousal -> Accuracy:", acc_a,
        "Precision:", prec_a,
        "Recall:", rec_a,
        "F1:", f1_a)

    all_acc_v.append(acc_v)
    all_pre_v.append(prec_v)
    all_recall_v.append(rec_v)
    all_f1_v.append(f1_v)

    all_acc_a.append(acc_a)
    all_pre_a.append(prec_a)
    all_recall_a.append(rec_a)
    all_f1_a.append(f1_a)

    final_results.append([test_movie,acc_v,prec_v,rec_v,f1_v,acc_a,prec_a,rec_a,f1_a])

df_results=pd.DataFrame(final_results,columns=["movie_name","acc_v","prec_v","rec_v","f1_v","acc_a","prec_a","rec_a","f1_a"])
df_results.to_csv("salma_percive_emotion_model_results/results_binary_lightweight_normalize.csv", index=False)    



    