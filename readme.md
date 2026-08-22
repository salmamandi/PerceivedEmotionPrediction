PerceivedEmotionPrediction.ipynb is the main code file.

audio_only folder contain movie wise audio clips extracted from movie clips. 
Folder Link: https://drive.google.com/drive/folders/1B_4QV4bW3ZmV-_NjtxC96RiKBJebWGrk?usp=sharing

To run this code, we need the following csv files:

genre_emotion.csv ⇒ summarize average perceived emotion in terms of valence and arousal. Genre of each movie represented using one hot encoder.

True_music_voice_valence_final.csv ⇒ This file contain information focusing on valence value. Please find the meaning of the following columns.

true_val: represent ground truth perceived valence in binary (low/high) based on absolute value of valence.

Val_bin_pred: audio tool predicted valence.

Aro_bin_pred: audio tool predicted arousal.

Audio_val: audio predicted binary valence based on Val_bin_pred.

Val_actual: video tool predicted valence.

Aro_actual: video tool predicted arousal.

Video_val: video tool predicted binary valence based on Val_actual.

Music_probability: what is the probability that clip contain music.

Voice_ratio: number of frames containing voice/total number of frames in clip.

Movie_bin2_valence_info.csv ⇒ Each movie clip has its ground truth valence and arousal, which are continous value ranging from 0 to 1. To convert the continuous value into binary label, we need to choose a threshold. A fixed threshold of 0.5 (i.e., values greater than 0.5 are labeled as high and the remaining values as low) is not always suitable. For some movies, the ground-truth valence or arousal values are concentrated around 0.5. In such cases, two values that are numerically very close (e.g., 0.49 and 0.51) would be assigned to different classes, even though they represent nearly identical emotional intensities. Consequently, a prediction of 0.51 for a ground-truth value of 0.49 would be considered incorrect despite the negligible difference between the two values. To address this limitation, we determine a movie-specific threshold by analyzing the distribution of ground-truth valence and arousal values for each movie individually. Specifically, we apply K-means clustering with (k=2) to partition the ground-truth values into two clusters corresponding to low and high emotion levels. The decision threshold is then defined as the boundary between the two clusters (e.g., the midpoint between the cluster centroids). This approach maximizes the separation between the two groups (high inter-cluster distance) while minimizing the variation within each group (low intra-cluster distance), resulting in a threshold that better reflects the underlying distribution of emotion values for each movie. The resulting threshold is subsequently used to assign binary ground-truth labels for the valence and arousal values of all clips within that movie. The column “mid_edge” contain movie wise threshold.

Subtitle_video_bin_pred2.csv ⇒ video tool predicted valence and arousal.

Audio_bin_pred2.csv ⇒ audio tool predicted valence and arousal.
<-------------------------------Details of deep learning based model---------------------------->
Salma_perceive_emo_pred_model: this folder contain all code files 
train2.py: It is used to train the model
predict_only.py: It is used to predict the model



