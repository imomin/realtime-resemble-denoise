# realtime-resemble-noise-cancellation

An attempt to create real-time noise cancellation using open source [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) and [Resemble Denoiser in ONNX
](https://github.com/skeskinen/resemble-denoise-onnx-inference) project. Thanks for your contribution.

# Installation
This project has been tested on Python 3.10. Install the necessary Python packages.

pip install -r requirements.txt

[Download file ONNX](https://github.com/skeskinen/resemble-denoise-onnx-inference/blob/master/denoiser.onnx) and place in the same directory.

In order to use this in real-time, you will need to install virtual audio driver. I am using [Blackhole](https://github.com/ExistentialAudio/BlackHole/tree/master) on my Mac.

# 
The audio data is channeled from the mic (input device) -> denoice app -> blackhole (output device) -> 3rd Party App with blackhole as mic (input device).
