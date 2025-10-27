import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk
import warnings
import os

# hide all Python warnings
warnings.filterwarnings("ignore")

# hide PyTorch distributed & AMP warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# hide tqdm or DataLoader warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# disable torch deprecation spam
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)

torch.set_num_threads(24)

# pylint: disable=W0105
"""
    This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
    YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
    In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
    If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
    In addition, you will need to add the extra datasets following the VCTK as an example.
"""
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# Name of the run for the Trainer
RUN_NAME = "YourTTS-Hindi-Finetune"
OUT_PATH = "/home/himanshubeniwal/photon_project/vansh/outputs/yourtts_hindi_finetune"
RESTORE_PATH = "/home/himanshubeniwal/photon_project/vansh/models/model_file.pth"


# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 16
batch_group_size = 64


# Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 16000

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 10

### Hindi dataset configuration
DATASET_PATH = "/home/himanshubeniwal/photon_project/vansh/hindi_data/clean_normalized"
META_FILE = os.path.join(DATASET_PATH, "metadata.csv.final.cleaned")

from TTS.config.shared_configs import BaseDatasetConfig
hindi_config = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="hindi_tts",
    meta_file_train=META_FILE,
    meta_file_val=None,
    path=DATASET_PATH,
    language="hi",
)

DATASETS_CONFIG_LIST = [hindi_config]


### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = "/home/himanshubeniwal/photon_project/vansh/models/model_se.pth"
SPEAKER_ENCODER_CONFIG_PATH = "/home/himanshubeniwal/photon_project/vansh/models/config_se.json"


D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
for dataset_conf in DATASETS_CONFIG_LIST:
    # Check if the embeddings weren't already computed, if not compute it
    embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)


# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    # use_speaker_encoder_as_loss=True,
    # Useful parameters to enable multilingual training
    # use_language_embedding=True,
    # embedded_language_dim=4,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="""
            - Original YourTTS trained using VCTK dataset
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=96,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=12,
    eval_split_max_size=256,
    print_step=25,
    plot_step=100,
    log_model_step=1000,
    save_step=1000,
    save_n_checkpoints=3,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
           "अआइईउऊऋॠएऐओऔऑअंअःऩऱऴॡ"
           "ािीुूृॄेैोौंःँ़्–—‘’“”…"
           "कखगघङचछजझञटठडढणतथदधन"
           "पफबभमयरलवशषसह"
           "क़ख़ग़ज़ड़ढ़फ़"
           "ॉॅॆॊ"
           "।॥,!?;:'\"-.०१२३४५६७८९0123456789₹$ ",

        punctuations="।॥,!?;:'\"-.₹$ ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    ),



    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=True,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=True,
    test_sentences=[
    [
        "यह मॉडल हिंदी में बहुत अच्छा बोल रहा है।",
        "ljspeech",   # or your Hindi speaker name
        None,
        "hi",
    ],
    [
        "मुझे सुनने में बहुत अच्छा लग रहा है।",
        "ljspeech",
        None,
        "hi",
    ],
    [
        "भारत की विविध भाषाओं में आवाज़ संश्लेषण करना रोमांचक है।",
        "ljspeech",
        None,
        "hi",
    ],
],

    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
# Init the model
model = Vits.init_from_config(config)

# 🔹 Distributed setup
import torch.distributed as dist
import os

if "RANK" in os.environ:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    print(f"[DDP] Rank {dist.get_rank()} initialized on GPU {local_rank}")
else:
    local_rank = 0
    print("[DDP] Running on single process")

# Limit to 100 epochs for overnight run
config.num_epochs = 100

# 🚀 Initialize Trainer — manually tell it which GPU to use (prevents RuntimeError)
trainer = Trainer(
    TrainerArgs(
        restore_path=RESTORE_PATH,
        skip_train_epoch=SKIP_TRAIN_EPOCH,
        # 🔹 Important: prevent auto-GPU detection
        gpu=local_rank,
    ),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# Optional: Auto-stop after ~5 hours
import time
start_time = time.time()
MAX_RUNTIME_HOURS = 5

while True:
    trainer.fit()
    elapsed = (time.time() - start_time) / 3600
    if elapsed > MAX_RUNTIME_HOURS:
        print(f"⏰ Reached {MAX_RUNTIME_HOURS} hours — stopping training early.")
        break

if dist.is_initialized():
    dist.destroy_process_group()

