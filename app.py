import numpy as np
import torch
import pyaudio
import threading
import queue
import time
import onnxruntime
import tkinter as tk
from tkinter import ttk
import logging
from librosa import stft, istft
from resampy.core import resample

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# STFT Parameters
STFT_HOP_LENGTH = 420
WIN_LENGTH = N_FFT = 4 * STFT_HOP_LENGTH

def _stft(x):
    s = stft(x, window='hann', win_length=WIN_LENGTH, n_fft=N_FFT, hop_length=STFT_HOP_LENGTH,
             center=True, pad_mode='reflect')
    s = s[..., :-1]  # Remove the last frame
    mag = np.abs(s)
    phi = np.angle(s)
    cos = np.cos(phi)
    sin = np.sin(phi)
    return mag, cos, sin

def _istft(mag: np.array, cos: np.array, sin: np.array):
    real = mag * cos
    imag = mag * sin
    s = real + imag * 1.0j
    s = np.pad(s, ((0, 0), (0, 0), (0, 1)), mode='edge')  # Restore last frame
    x = istft(s, window='hann', win_length=WIN_LENGTH, hop_length=STFT_HOP_LENGTH, n_fft=N_FFT)
    return x

def model(onnx_session, wav: np.array) -> np.array:
    padded_wav = np.pad(wav, ((0, 0), (0, 441)))
    mag, cos, sin = _stft(padded_wav)
    
    ort_inputs = {
        "mag": mag.astype(np.float32),
        "cos": cos.astype(np.float32),
        "sin": sin.astype(np.float32),
    }

    sep_mag, sep_cos, sep_sin = onnx_session.run(None, ort_inputs)

    o = _istft(sep_mag, sep_cos, sep_sin)
    o = o[:wav.shape[-1]]
    return o

def run_model(onnx_session, wav: np.array, sample_rate: int, batch_process_chunks: bool = False) -> np.array:
    assert wav.ndim == 1, 'Input should be 1D (mono) wav'

    if sample_rate != 44100:
        wav = resample(wav, sample_rate, 44100, filter='kaiser_best', parallel=True)

    chunk_length = int(44100 * 30)  # 30 seconds per chunk
    hop_length = chunk_length  # Non-overlapping

    num_chunks = 1 + (wav.shape[-1] - 1) // hop_length
    n_pad = (num_chunks - wav.shape[-1] % hop_length) % num_chunks
    wav = np.pad(wav, (0, n_pad))
    
    chunks = wav.reshape((num_chunks, -1))
    abs_max = np.clip(np.max(np.abs(chunks), axis=-1, keepdims=True), a_min=1e-7, a_max=None)
    chunks /= abs_max

    if batch_process_chunks:
        res_chunks = model(onnx_session, chunks)
    else:
        res_chunks = np.array([model(onnx_session, c[None]) for c in chunks]).squeeze(axis=1)
    res_chunks *= abs_max

    res = res_chunks.reshape(-1)
    return res[:wav.shape[-1]], 44100

class AudioDenoiserApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Denoiser")

        # Audio Configuration
        self.CHUNK = 4096  # Number of samples per frame
        self.FORMAT = pyaudio.paInt16  # 16-bit resolution
        self.CHANNELS = 1  # Mono audio
        self.RATE = 44100  # Sampling rate in Hz

        self.device = "cpu"  # Use "cuda" if GPU is available and compatible

        # Queues for communication between threads
        self.read_queue = queue.Queue(maxsize=300)      # Holds raw audio frames from the microphone
        self.process_queue = queue.Queue(maxsize=300)   # Holds processed audio frames ready for playback

        # Flag to control threading
        self.stop_flag = threading.Event()
        self.threads = []

        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()

        # Get default input and output device indices
        try:
            self.default_input_device_index = self.pa.get_default_input_device_info()['index']
            self.default_output_device_index = self.pa.get_default_output_device_info()['index']
        except IOError:
            logging.error("No default input/output device found.")
            self.default_input_device_index = None
            self.default_output_device_index = None

        # Set up ONNX runtime session
        self.session = self.setup_onnx_session()

        # Initialize GUI components
        self.create_widgets()

    def setup_onnx_session(self):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        opts.log_severity_level = 4

        try:
            session = onnxruntime.InferenceSession(
                'denoiser.onnx',
                providers=["CPUExecutionProvider"],
                sess_options=opts,
            )
            logging.info("ONNX model loaded successfully.")
            return session
        except Exception as e:
            logging.error(f"Failed to load ONNX model: {e}")
            return None

    def create_widgets(self):
        # Input device dropdown
        self.input_device_var = tk.StringVar()
        self.input_label = ttk.Label(self.master, text="Input Device:")
        self.input_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

        self.input_dropdown = ttk.Combobox(self.master, textvariable=self.input_device_var, state="readonly", width=50)
        self.input_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.populate_input_devices()

        # Output device dropdown
        self.output_device_var = tk.StringVar()
        self.output_label = ttk.Label(self.master, text="Output Device:")
        self.output_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

        self.output_dropdown = ttk.Combobox(self.master, textvariable=self.output_device_var, state="readonly", width=50)
        self.output_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.populate_output_devices()

        # Refresh Devices button
        self.refresh_button = ttk.Button(self.master, text="Refresh Audio Devices", command=self.update_device_lists)
        self.refresh_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        # Toggle button
        self.toggle_button = ttk.Button(self.master, text="Start", command=self.toggle_processing)
        self.toggle_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

    def populate_input_devices(self):
        input_devices, _ = self.get_audio_devices()
        if not input_devices:
            logging.warning("No input devices found.")
        # Set default input device
        if self.default_input_device_index is not None:
            default_input_device_name = f"{self.default_input_device_index}: {self.pa.get_device_info_by_index(self.default_input_device_index).get('name')}"
            if default_input_device_name in input_devices:
                self.input_device_var.set(default_input_device_name)
            else:
                self.input_device_var.set(input_devices[0] if input_devices else "")
        self.input_dropdown['values'] = input_devices

    def populate_output_devices(self):
        _, output_devices = self.get_audio_devices()
        if not output_devices:
            logging.warning("No output devices found.")
        # Set default output device
        if self.default_output_device_index is not None:
            default_output_device_name = f"{self.default_output_device_index}: {self.pa.get_device_info_by_index(self.default_output_device_index).get('name')}"
            if default_output_device_name in output_devices:
                self.output_device_var.set(default_output_device_name)
            else:
                self.output_device_var.set(output_devices[0] if output_devices else "")
        self.output_dropdown['values'] = output_devices

    def get_audio_devices(self):
        """Get lists of input and output audio devices."""
        input_devices = []
        output_devices = []
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            device_name = f"{i}: {device_info.get('name')}"
            if device_info.get('maxInputChannels') > 0:
                input_devices.append(device_name)
            if device_info.get('maxOutputChannels') > 0:
                output_devices.append(device_name)
        return input_devices, output_devices

    def update_device_lists(self):
        """Update the device lists and refresh dropdowns."""
        logging.info("Refreshing audio devices...")
        input_devices, output_devices = self.get_audio_devices()

        # Update input devices
        current_input = self.input_device_var.get()
        self.input_dropdown['values'] = input_devices
        if current_input not in input_devices:
            # Reset to default or first available
            if self.default_input_device_index is not None:
                default_input_device_name = f"{self.default_input_device_index}: {self.pa.get_device_info_by_index(self.default_input_device_index).get('name')}"
                if default_input_device_name in input_devices:
                    self.input_device_var.set(default_input_device_name)
                elif input_devices:
                    self.input_device_var.set(input_devices[0])
                else:
                    self.input_device_var.set("")
            elif input_devices:
                self.input_device_var.set(input_devices[0])
            else:
                self.input_device_var.set("")

        # Update output devices
        current_output = self.output_device_var.get()
        self.output_dropdown['values'] = output_devices
        if current_output not in output_devices:
            # Reset to default or first available
            if self.default_output_device_index is not None:
                default_output_device_name = f"{self.default_output_device_index}: {self.pa.get_device_info_by_index(self.default_output_device_index).get('name')}"
                if default_output_device_name in output_devices:
                    self.output_device_var.set(default_output_device_name)
                elif output_devices:
                    self.output_device_var.set(output_devices[0])
                else:
                    self.output_device_var.set("")
            elif output_devices:
                self.output_device_var.set(output_devices[0])
            else:
                self.output_device_var.set("")

    def toggle_processing(self):
        if self.toggle_button.config('text')[-1] == 'Start':
            self.toggle_button.config(text='Stop')
            self.start_processing()
        else:
            self.toggle_button.config(text='Start')
            self.stop_processing()

    def start_processing(self):
        if self.session is None:
            logging.error("ONNX session is not initialized.")
            return

        self.stop_flag.clear()

        # Reinitialize the queues
        self.read_queue = queue.Queue(maxsize=300)
        self.process_queue = queue.Queue(maxsize=300)

        # Get selected input and output device indices
        input_selection = self.input_device_var.get()
        output_selection = self.output_device_var.get()

        if input_selection == "":
            logging.error("Please select an input device.")
            return
        if output_selection == "":
            logging.error("Please select an output device.")
            return

        input_device_index = int(input_selection.split(":")[0])
        output_device_index = int(output_selection.split(":")[0])

        try:
            # Open PyAudio input stream (microphone)
            self.input_stream = self.pa.open(format=self.FORMAT,
                                            channels=self.CHANNELS,
                                            rate=self.RATE,
                                            input=True,
                                            frames_per_buffer=self.CHUNK,
                                            input_device_index=input_device_index,
                                            start=False)

            # Open PyAudio output stream
            self.output_stream = self.pa.open(format=self.FORMAT,
                                             channels=self.CHANNELS,
                                             rate=self.RATE,
                                             output=True,
                                             frames_per_buffer=self.CHUNK,
                                             output_device_index=output_device_index,
                                             start=False)

            # Start the streams
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            logging.info("Audio streams started.")

            # Start threads
            self.threads = []
            read_thread = threading.Thread(target=self.read_audio, name="ReadThread")
            process_thread = threading.Thread(target=self.process_audio, name="ProcessThread")
            playback_thread = threading.Thread(target=self.playback_audio, name="PlaybackThread")

            for t in [read_thread, process_thread, playback_thread]:
                t.daemon = True
                t.start()
                self.threads.append(t)

            logging.info("Audio processing threads started.")

        except Exception as e:
            logging.error(f"Failed to start audio processing: {e}")
            self.stop_flag.set()
            self.toggle_button.config(text="Start")

    def stop_processing(self):
        logging.info("Stopping audio processing...")
        self.stop_flag.set()

        # Wait for threads to finish
        for t in self.threads:
            t.join(timeout=1)

        # Close the streams
        try:
            if hasattr(self, 'input_stream') and self.input_stream.is_active():
                self.input_stream.stop_stream()
                self.input_stream.close()
                logging.info("Input stream closed.")
        except Exception as e:
            logging.error(f"Error closing input stream: {e}")

        try:
            if hasattr(self, 'output_stream') and self.output_stream.is_active():
                self.output_stream.stop_stream()
                self.output_stream.close()
                logging.info("Output stream closed.")
        except Exception as e:
            logging.error(f"Error closing output stream: {e}")

        # Reset streams
        self.input_stream = None
        self.output_stream = None

        # Clear the threads list
        self.threads = []

    def read_audio(self):
        """Thread function to read audio frames from the microphone."""
        logging.info("Read thread started.")
        try:
            while not self.stop_flag.is_set():
                try:
                    audio_data = self.input_stream.read(self.CHUNK, exception_on_overflow=False)
                    self.read_queue.put(audio_data, timeout=0.5)
                except queue.Full:
                    logging.warning("Read queue is full. Dropping frame.")
                except Exception as e:
                    logging.error(f"Error reading audio input: {e}")
                    self.stop_flag.set()
                    break
            # Signal the processing thread that reading is done
            self.read_queue.put(None)
            logging.info("Read thread terminated.")
        except Exception as e:
            logging.error(f"Exception in read_audio thread: {e}")
            self.stop_flag.set()

    def process_audio(self):
        """Thread function to process audio frames."""
        logging.info("Process thread started.")
        try:
            while not self.stop_flag.is_set():
                try:
                    audio_data = self.read_queue.get(timeout=1)
                    if audio_data is None:
                        # End of audio data
                        self.process_queue.put(None)
                        break
                    start_time = time.perf_counter()
                    
                    # Convert bytes to numpy array and normalize
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Process audio using the denoiser
                    clean_audio, new_sr = run_model(self.session, audio_np, self.RATE, batch_process_chunks=True)
                    
                    # Convert back to numpy and int16
                    clean_audio_np = clean_audio * 32768.0
                    clean_audio_np = np.clip(clean_audio_np, -32768, 32767).astype(np.int16)
                    clean_audio_bytes = clean_audio_np.tobytes()

                    processing_time = time.perf_counter() - start_time
                    logging.info(f"Processing time: {processing_time*1000:.2f} ms")

                    try:
                        self.process_queue.put(clean_audio_bytes, timeout=0.5)
                    except queue.Full:
                        logging.warning("Process queue is full. Dropping frame.")
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error in process_audio: {e}")
                    self.stop_flag.set()
                    break

            # Signal main thread that processing is done
            self.process_queue.put(None)
            logging.info("Process thread terminated.")
        except Exception as e:
            logging.error(f"Exception in process_audio thread: {e}")
            self.stop_flag.set()

    def playback_audio(self):
        """Thread to play back audio."""
        logging.info("Playback thread started.")
        try:
            while not self.stop_flag.is_set():
                try:
                    clean_audio_bytes = self.process_queue.get(timeout=1)
                    if clean_audio_bytes is None:
                        # End of audio data
                        break
                    self.output_stream.write(clean_audio_bytes, exception_on_underflow=False)
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error playing audio: {e}")
                    self.stop_flag.set()
                    break
            logging.info("Playback thread terminated.")
        except Exception as e:
            logging.error(f"Exception in playback_audio thread: {e}")
            self.stop_flag.set()

    def on_closing(self):
        if self.toggle_button['text'] == 'Stop':
            self.stop_processing()
        self.pa.terminate()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioDenoiserApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
