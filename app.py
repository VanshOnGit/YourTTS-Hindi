import sys, os
# --------------------------------------------
# 👇 Add this line to make Streamlit find your local TTS folder
sys.path.append(os.path.abspath("./TTS"))
# --------------------------------------------

import streamlit as st
from TTS.utils.synthesizer import Synthesizer
import torch

# -----------------------
# Paths
# -----------------------
MODEL_PATH = "./hindi_best_model/best_model_75171.pth"
CONFIG_PATH = "./hindi_best_model/config.json"
SPEAKER_PATH = "./hindi_best_model/speakers.pth"
REF_MALE = "./ref_audio/ref_male.wav"
REF_FEMALE = "./ref_audio/ref_female.wav"

# -----------------------
# Load Synthesizer (cached so it loads once)
# -----------------------
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

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Hindi TTS", page_icon="🎙️", layout="centered")
st.title("🎙️ Hindi Text-to-Speech (YourTTS Fine-tuned)")
st.markdown("Type Hindi text below and choose **Male** or **Female** voice to synthesize speech.")

# Input text
text = st.text_area(
    "📝 Enter Hindi text:",
    "यह मॉडल हिंदी में बहुत अच्छा बोल रहा है।",
    height=120
)

# Voice selection
voice_type = st.radio("🗣️ Choose voice type:", ["Male", "Female"])
ref_audio = REF_MALE if voice_type == "Male" else REF_FEMALE

# Generate button
if st.button("🎧 Generate Speech"):
    with st.spinner("Synthesizing audio..."):
        wav = synth.tts(text=text, speaker_wav=ref_audio, language_name="hi")

        # Save output file
        out_dir = f"hindi_outputs/{voice_type.lower()}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"hindi_{voice_type.lower()}.wav")
        synth.save_wav(wav, out_path)

        st.audio(out_path)
        st.success(f"✅ {voice_type} voice generated and saved to `{out_path}`")

# Optional info
st.caption("Model: Fine-tuned YourTTS | Language: Hindi | Voices cloned from reference audios.")
