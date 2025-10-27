import sys, os
sys.path.append(os.path.abspath("./TTS"))
import streamlit as st
from TTS.utils.synthesizer import Synthesizer
import torch
import os, requests

def download_from_hf(url, filename):
    if not os.path.exists(filename):
        print(f"⬇️ Downloading {filename} ...")
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded {filename}")
    else:
        print(f"✅ {filename} already exists, skipping download")

BASE_URL = "https://huggingface.co/datasets/VanshOnHF/yourtts-hindi-model/resolve/main/hindi_best_model/"

download_from_hf(BASE_URL + "best_model_75171.pth", "best_model_75171.pth")
download_from_hf(BASE_URL + "config.json", "config.json")
download_from_hf(BASE_URL + "speakers.pth", "speakers.pth")

MODEL_PATH = "best_model_75171.pth"
CONFIG_PATH = "config.json"
SPEAKER_PATH = "speakers.pth"

REF_MALE = "./ref_audio/ref_male.wav"
REF_FEMALE = "./ref_audio/ref_female.wav"


@st.cache_resource
def load_synth():
    synth = Synthesizer(
        tts_checkpoint=MODEL_PATH,
        tts_config_path=CONFIG_PATH,
        use_cuda=False
    )
    spk_embeds = torch.load(SPEAKER_PATH, map_location="cpu")
    return synth, spk_embeds

synth, spk_embeds = load_synth()


st.set_page_config(page_title="Hindi TTS", page_icon="🎙️", layout="centered")
st.title("🎙️ Hindi Text-to-Speech (YourTTS Fine-tuned)")
st.markdown("Type Hindi text below and choose **Male** or **Female** voice to synthesize speech.")

text = st.text_area(
    "📝 Enter Hindi text:",
    "यह मॉडल हिंदी में बहुत अच्छा बोल रहा है।",
    height=120
)

voice_type = st.radio("🗣️ Choose voice type:", ["Male", "Female"])
ref_audio = REF_MALE if voice_type == "Male" else REF_FEMALE

if st.button("🎧 Generate Speech"):
    with st.spinner("Synthesizing audio..."):
        wav = synth.tts(text=text, speaker_wav=ref_audio, language_name="hi")

        out_dir = f"hindi_outputs/{voice_type.lower()}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"hindi_{voice_type.lower()}.wav")
        synth.save_wav(wav, out_path)

        st.audio(out_path)
        st.success(f"✅ {voice_type} voice generated and saved to `{out_path}`")

st.caption("Model: Fine-tuned YourTTS | Language: Hindi | Voices cloned from reference audios.")
