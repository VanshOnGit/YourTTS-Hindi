import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk

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

RUN_NAME = "YourTTS-Hindi-Finetune"
OUT_PATH = "./outputs/yourtts_hindi_finetune"
RESTORE_PATH = "./models/model_file.pth"


SKIP_TRAIN_EPOCH = False

BATCH_SIZE = 16
batch_group_size = 64


SAMPLE_RATE = 16000

MAX_AUDIO_LEN_IN_SECONDS = 10

DATASET_PATH = "./clean_normalized"
META_FILE = os.path.join(DATASET_PATH, "./clean_normalized/metadata.csv.final.cleaned")

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
SPEAKER_ENCODER_CHECKPOINT_PATH = "./models/model_se.pth"
SPEAKER_ENCODER_CONFIG_PATH = "./models/config_se.json"


D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

for dataset_conf in DATASETS_CONFIG_LIST:
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


audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

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
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=2,
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
                   "अआइईउऊऋएऐओऔऑअंअः"
                   "ािीुूृेैोौंःँ़्"
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
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
    [
        "यह मॉडल हिंदी में बहुत अच्छा बोल रहा है।",
        "ljspeech",   
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

    use_weighted_sampler=True,
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    speaker_encoder_loss_alpha=9.0,
)

train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

model = Vits.init_from_config(config)

trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
