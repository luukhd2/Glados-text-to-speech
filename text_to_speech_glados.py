"""Use this script to generate speech from text using the GLaDOS model.
The script requires the `trimmed_requirements.py` file to be present in the same directory.
"""

import pathlib

import torch
import scipy.io.wavfile

from trimmed_requirements import get_all, prepare_text


def run_tts(model_dir, text, output_path, device, alpha: float = 1.0):
    """
    Generate speech from text using the GLaDOS model.

    Args:
    model_dir: pathlib.Path, path to the directory containing the models.
               by default, the models are expected to be in the './models' directory.
    text: str, the text to be converted to speech.
    output_path: pathlib.Path, path to save the generated speech. I recommend using the .wav format.
    device: str, the device to run the model on. Use 'cpu' for CPU and 'cuda' for GPU.
    alpha: float, the alpha value to use for controlling the speed of speech. Default is 1.0.

    Returns:
    None
    """
    try:
        emb, glados, vocoder, device = get_all(model_dir=model_dir, device=device)
    except FileNotFoundError as e:
        print("Check model directory", model_dir)
        print("They should download automatically with git lfs.")
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
