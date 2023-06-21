![image](https://github.com/Dschogo/whisperx-webui/assets/36862419/1a81177c-1865-4b08-b11a-d0ae086eeedf)# WhisperX webui

build with gradio and WhisperX

## Installation

- clone
- install requirements
- install torch (for cuda `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`) (cpu: `pip install torch torchvision torchaudio`)

To use the go.bat (start script) you have to install it in a venv (`python -m venv venv`) - or edit the bat file.
To use diarization, you have to specify a hf_token for the env.

For example via a .env file:

```env
    hf_token=your_token
```


## Features

- Transcribe with whisperx - 70x realtime
- Multiple files at once (single language per batch)
- All cli options and output formats
- easy to use, and view/ copy 

![image](https://github.com/Dschogo/whisperx-webui/assets/36862419/ff144328-51b6-45c1-a89a-ad4b63ea1699)

![image](https://github.com/Dschogo/whisperx-webui/assets/36862419/75442dee-a4fb-4151-a27c-9d0bcb87a25d)


## Todo

- [ ] Rest of options available in cli
