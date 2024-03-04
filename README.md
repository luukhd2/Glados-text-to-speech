# Glados-text-to-speech
 TTS using the Glados voice from Portal 1. 
 This is a trimmed down version taking just the GLaDOS model of https://github.com/keithito/tacotron
 
Thanks to them for their code and training the models. I made this repo for people who just want the GLaDOS voice and a simple set-up and downloadable weights.
 
![alt text](https://github.com/luukhd2/Glados-text-to-speech/blob/main/im.png?raw=true)

# Setup:
1. Clone this repository and cd into the directory.
2. Create and activate the conda environment with:
```conda env create --file conda_requirements.yaml --prefix ./glados_tts_env```

3. Activate the environment with:
```conda activate ./glados_tts_env```

# Ready to run:
Now you can run the example script with:
```python ./text_to_speech_glados.py```
