"""Use this script to generate speech from text using the GLaDOS model.
The script requires the `trimmed_requirements.py` file to be present in the same directory.
"""

import pathlib

import torch
import scipy.io.wavfile

from trimmed_requirements import get_all, prepare_text


def run_tts(model_dir, text, output_path, device, alpha: float = 1.0):
    try:
        emb, glados, vocoder, device = get_all(model_dir=model_dir, device=device)
    except FileNotFoundError as e:
        print(e)
        print("Please download the models from the release page.")
        raise e

    x = prepare_text(text, model_dir=model_dir)

    with torch.no_grad():
        tts_output = glados.generate_jit(x, emb, alpha)
        mel = tts_output["mel_post"].to(device)
        audio = vocoder(mel)
        audio = audio.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype("int16")
        scipy.io.wavfile.write(output_path, 22050, audio)


if __name__ == "__main__":
    # Load miniconda: source ~/miniconda3/bin/activate
    # Create an env: conda env create --file conda_requirements.yaml --prefix ./glados_tts_env
    # Activate the script: conda activate ./glados_tts_env
    # Run this example with: python text_to_speech_glados.py

    # example
    run_tts(
        model_dir=pathlib.Path("./models/"),
        text=""" Power-up complete.
        Thank you for participating in this Aperture Science computer-aided enrichment
        activity. Goodbye.""",
        output_path=pathlib.Path("./glados_hello.wav"),
        device="cpu",
    )
