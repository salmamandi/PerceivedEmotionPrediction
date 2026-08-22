import torch
from torch.utils.data import DataLoader
from Salma_perceive_emo_pred_model import config
from Salma_perceive_emo_pred_model.utils import load_labels,compute_metrics,compute_mean_std
from Salma_perceive_emo_pred_model.dataset import SceneFeatureDataset, collate_fn
#from Salma_perceive_emo_pred_model.model import PerceivedVA_Model
from Salma_perceive_emo_pred_model.lightweight_model import PerceivedVA_Model
from Salma_perceive_emo_pred_model.loss import compute_loss
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split
import os
import numpy as np

video_root = config.VIDEO_FEATURE_ROOT
audio_root = config.AUDIO_FEATURE_ROOT

movies = sorted(os.listdir(video_root))

dataset = SceneFeatureDataset(video_root, audio_root)
label_dict = load_labels(config.LABEL_FILE)

final_results=[]
all_mse_v = []
all_mae_v = []
all_ccc_v = []

all_mse_a = []
all_mae_a = []
all_ccc_a = []

for test_movie in movies:

    print("Testing on movie:", test_movie)

    train_indices = []
    test_indices = []

    for i, sample in enumerate(dataset.samples):

        movie = sample[0]

        if movie == test_movie:
            test_indices.append(i)
        else:
            train_indices.append(i)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)


    train_size = int(0.8 * len(train_subset))
    val_size = len(train_subset) - train_size

    train_subset, val_subset = random_split(train_subset, [train_size, val_size],generator=torch.Generator().manual_seed(42))
    
    video_mean, video_std, audio_mean, audio_std = compute_mean_std(train_subset)
    print("mean and std deviation size:",video_mean.shape, video_std.shape)
    dataset_norm = SceneFeatureDataset(
    video_root, audio_root,
    normalize=True,
    video_mean=video_mean,
    video_std=video_std,
    audio_mean=audio_mean,
    audio_std=audio_std
     )
     
    train_indices_final = [train_indices[i] for i in train_subset.indices]
    val_indices_final   = [train_indices[i] for i in val_subset.indices] 

    train_dataset = torch.utils.data.Subset(dataset_norm, train_indices_final)
    val_dataset = torch.utils.data.Subset(dataset_norm, val_indices_final)
    test_dataset = torch.utils.data.Subset(dataset_norm, test_subset.indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device for training:", device)

    model = PerceivedVA_Model(config.VIDEO_DIM, config.AUDIO_DIM).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # comment this if loop if no test movie based model is not saved
    # if test_movie in ["AboutTime","Gifted","LadyBird","TheBlindSide"]:

    #     test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     collate_fn=collate_fn
    #      )

    #     model.load_state_dict(
    #     torch.load(f"{config.MODEL_SAVE_PATH}/model_{test_movie}.pth")
    #     )
    #     model.eval()

    #     val_loss = 0

    #     pred_v_all = []
    #     pred_a_all = []
    #     gt_v_all = []
    #     gt_a_all = []

    #     with torch.no_grad():

    #         for video, audio, mask, clip_names in test_loader:

    #             pred_v, pred_a, delta = model(video, audio, mask)

    #             for i, clip in enumerate(clip_names):

    #                 gt_v, gt_a = label_dict[clip]

    #                 pred_v_all.append(pred_v[i].item())
    #                 pred_a_all.append(pred_a[i].item())

    #                 gt_v_all.append(gt_v)
    #                 gt_a_all.append(gt_a)
                
    #     # Convert to numpy
    #     pred_v_all = np.array(pred_v_all)
    #     pred_a_all = np.array(pred_a_all)
    #     gt_v_all = np.array(gt_v_all)
    #     gt_a_all = np.array(gt_a_all)

    #     # Compute metrics
    #     mse_v, mae_v, ccc_v = compute_metrics(pred_v_all, gt_v_all)
    #     mse_a, mae_a, ccc_a = compute_metrics(pred_a_all, gt_a_all)

    #     all_mse_v.append(mse_v)
    #     all_mae_v.append(mae_v)
    #     all_ccc_v.append(ccc_v)

    #     all_mse_a.append(mse_a)
    #     all_mae_a.append(mae_a)
    #     all_ccc_a.append(ccc_a)

    #     print("\nResults for movie:", test_movie)

    #     print("Valence -> MSE:", mse_v, "MAE:", mae_v, "CCC:", ccc_v)
    #     print("Arousal -> MSE:", mse_a, "MAE:", mae_a, "CCC:", ccc_a)
    #     final_results.append([test_movie,mse_v,mae_v,ccc_v,mse_a,mae_a,ccc_a])
    #     continue

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader=DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    
    best_loss= float("inf")

    
    # training
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss=0
        count=0

        for video, audio, mask, clip_names in train_loader:

            video = video.float().to(device)
            audio = audio.float().to(device)
            mask=mask.to(device)

            pred_v, pred_a,delta = model(video, audio,mask)

            gt_v = []
            gt_a = []

            for clip in clip_names:

                valence, arousal = label_dict[clip]
                gt_v.append(valence)
                gt_a.append(arousal)

            
            gt_v = torch.tensor(gt_v, dtype=torch.float32).to(device)
            gt_a = torch.tensor(gt_a, dtype=torch.float32).to(device)

            loss = compute_loss(pred_v, pred_a, gt_v, gt_a, delta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+=loss.item()
            count+=1
            
        epoch_loss /= count
        print("Epoch:", epoch, "Train Loss:", epoch_loss)

        # validation
        model.eval()

        val_loss = 0

        with torch.no_grad():

            for video, audio, mask, clip_names in val_loader:
                video=video.float().to(device)
                audio=audio.float().to(device)
                mask=mask.to(device)

                pred_v, pred_a, delta = model(video, audio, mask)

                gt_v = []
                gt_a = []

                for name in clip_names:
                    v, a = label_dict[name]
                    gt_v.append(v)
                    gt_a.append(a)

                gt_v = torch.tensor(gt_v, dtype=torch.float32).to(device)
                gt_a = torch.tensor(gt_a, dtype=torch.float32).to(device)

                loss = compute_loss(pred_v, pred_a, gt_v, gt_a, delta)

                val_loss += loss.item()

        val_loss/=len(val_loader)        

        if val_loss < best_loss:

            best_loss = val_loss
            os.makedirs("saved_data_normalize_variables/"+test_movie, exist_ok=True)
            np.save("saved_data_normalize_variables/"+test_movie+"/video_mean.npy",video_mean)
            np.save("saved_data_normalize_variables/"+test_movie+"/audio_mean.npy",audio_mean)
            np.save("saved_data_normalize_variables/"+test_movie+"/video_std.npy",video_std)
            np.save("saved_data_normalize_variables/"+test_movie+"/audio_std.npy",audio_std)

            torch.save(
                model.state_dict(),
                f"{config.MODEL_SAVE_PATH}/model_{test_movie}.pth"
            )
    

    # for testing
    model.load_state_dict(
    torch.load(f"{config.MODEL_SAVE_PATH}/model_{test_movie}.pth")
     )
    
    model.to(device)
    model.eval()

    val_loss = 0

    pred_v_all = []
    pred_a_all = []
    gt_v_all = []
    gt_a_all = []

    with torch.no_grad():

        for video, audio, mask, clip_names in test_loader:

            video=video.float().to(device)
            audio=audio.float().to(device)
            mask=mask.to(device)

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

    # Compute metrics
    mse_v, mae_v, ccc_v = compute_metrics(pred_v_all, gt_v_all)
    mse_a, mae_a, ccc_a = compute_metrics(pred_a_all, gt_a_all)

    all_mse_v.append(mse_v)
    all_mae_v.append(mae_v)
    all_ccc_v.append(ccc_v)

    all_mse_a.append(mse_a)
    all_mae_a.append(mae_a)
    all_ccc_a.append(ccc_a)

    print("\nResults for movie:", test_movie)

    print("Valence -> MSE:", mse_v, "MAE:", mae_v, "CCC:", ccc_v)
    print("Arousal -> MSE:", mse_a, "MAE:", mae_a, "CCC:", ccc_a)
    final_results.append([test_movie,mse_v,mae_v,ccc_v,mse_a,mae_a,ccc_a])

print("\n==============================")
print("Final Average Results (LOOCV)")
print("==============================")

print("Valence:")
print("MSE:", np.mean(all_mse_v))
print("MAE:", np.mean(all_mae_v))
print("CCC:", np.mean(all_ccc_v))

print("\nArousal:")
print("MSE:", np.mean(all_mse_a))
print("MAE:", np.mean(all_mae_a))
print("CCC:", np.mean(all_ccc_a))    
df_results=pd.DataFrame(final_results,columns=["movie_name","mse_v","mae_v","ccc_v","mse_a","mae_a","ccc_a"])
df_results.to_csv("salma_percive_emotion_model_results/saving_results_lightweight_continuous_value.csv", index=False)