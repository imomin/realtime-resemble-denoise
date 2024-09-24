# realtime-resemble-noise

Work in progress, first attempt to create real-time noise cancellation using open source [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) and [Resemble Denoiser in ONNX
](https://github.com/skeskinen/resemble-denoise-onnx-inference). [Download file ONNX](https://github.com/skeskinen/resemble-denoise-onnx-inference/blob/master/denoiser.onnx) Thanks for your contribution.

In order to use this in real-time, you will need to install virtual audio driver. I am using [Blackhole](https://github.com/ExistentialAudio/BlackHole/tree/master) on my Mac.

The audio data is channeled from the mic (input device) -> denoice app -> blackhole (output device) -> 3rd Party App with blackhole as mic (input device).
