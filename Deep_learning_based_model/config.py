# Feature directories
VIDEO_FEATURE_ROOT = "checkpoints_scene_srt/mvit_v1_scene_features"
AUDIO_FEATURE_ROOT = "checkpoints_scene_srt/wav2vec2_scene_features"
MODEL_SAVE_PATH="saved_models_lightweight"
#MODEL_SAVE_PATH="saved_models"

# Label file
LABEL_FILE = "Salma_perceive_emo_pred_model/genre_emotion.csv"
VALENCE_THRESH_FILE="movie_cluster2_valence_info.csv"
AROUSAL_THRESH_FILE="movie_cluster2_arousal_info.csv"

# Feature dimensions
VIDEO_DIM = 768
AUDIO_DIM = 1024

# Model parameters
HIDDEN_DIM = 64 # changing from 256 to 64

# Training parameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 20