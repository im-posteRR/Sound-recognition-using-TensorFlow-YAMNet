# YAMNet Sound Recognition

This project uses TensorFlow's YAMNet model to detect sounds from audio files.
It filters general categories and outputs the most specific detected sound.
It’s ideal for identifying animal noises, environmental sounds, and other audio events.

## Features

- Loads audio files and resamples to 16kHz.
- Uses YAMNet to predict the top sound categories.
- Ignores general categories and outputs the most specific sound detected.

## Requirements

- Python 3.9+
- numpy
- tensorflow
- tensorflow-hub
- librosa
- soundfile (optional)

## Installation

```bash
pip install numpy tensorflow tensorflow-hub librosa soundfile
```

<audio controls src="Meow.wav" title="Title"></audio>
