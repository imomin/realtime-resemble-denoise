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

# Audio configuration
CHUNK = 4096 #4096 #8192  # Number of samples per frame
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate in Hz

# Queues for communication between threads
read_queue = queue.Queue(maxsize=300)      # Holds raw audio frames from the microphone
process_queue = queue.Queue(maxsize=300)   # Holds processed audio frames ready for playback

# Flags to control threading
processing_stop_flag = threading.Event()
playback_stop_flag = threading.Event()

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Get default input and output device indices
try:
    default_input_device_info = pa.get_default_input_device_info()
    default_input_device_index = default_input_device_info['index']
    default_input_device_name = f"{default_input_device_index}: {default_input_device_info['name']}"
except IOError:
    logging.error("No default input device found.")
    default_input_device_index = None
    default_input_device_name = ""

try:
    default_output_device_info = pa.get_default_output_device_info()
    default_output_device_index = default_output_device_info['index']
    default_output_device_name = f"{default_output_device_index}: {default_output_device_info['name']}"
except IOError:
    logging.error("No default output device found.")
    default_output_device_index = None
    default_output_device_name = ""

# Set up ONNX runtime session
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
except Exception as e:
    logging.error(f"Failed to load ONNX model: {e}")
    session = None

def get_audio_devices():
    """Get lists of input and output audio devices."""
    input_devices = []
    output_devices = []
    for i in range(pa.get_device_count()):
        device_info = pa.get_device_info_by_index(i)
        device_name = f"{i}: {device_info.get('name')}"
        if device_info.get('maxInputChannels') > 0:
            input_devices.append(device_name)
        if device_info.get('maxOutputChannels') > 0:
            output_devices.append(device_name)
    return input_devices, output_devices

def update_device_lists():
    """Update the device lists and refresh dropdowns."""
    logging.info("Refreshing audio devices...")
    input_devices, output_devices = get_audio_devices()

    # Update input devices
    current_input = input_device_var.get()
    input_dropdown['values'] = input_devices
    if current_input not in input_devices:
        # Reset to default or first available
        if default_input_device_name in input_devices:
            input_device_var.set(default_input_device_name)
        elif input_devices:
            input_device_var.set(input_devices[0])
        else:
            input_device_var.set("")
        logging.info("Input device selection updated.")

    # Update output devices
    current_output = output_device_var.get()
    output_dropdown['values'] = output_devices
    if current_output not in output_devices:
        # Reset to default or first available
        if default_output_device_name in output_devices:
            output_device_var.set(default_output_device_name)
        elif output_devices:
            output_device_var.set(output_devices[0])
        else:
            output_device_var.set("")
        logging.info("Output device selection updated.")

def start_processing():
    global input_stream, output_stream, threads, processing_stop_flag, playback_stop_flag, read_queue, process_queue

    if session is None:
        logging.error("ONNX session is not initialized.")
        return

    logging.info("Starting audio processing...")
    processing_stop_flag.clear()
    playback_stop_flag.clear()

    # Reinitialize the queues
    read_queue = queue.Queue(maxsize=300)
    process_queue = queue.Queue(maxsize=300)

    # Get selected input and output device indices
    input_selection = input_device_var.get()
    output_selection = output_device_var.get()

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
        input_stream = pa.open(format=FORMAT,
                               channels=CHANNELS,
                               rate=RATE,
                               input=True,
                               frames_per_buffer=CHUNK,
                               input_device_index=input_device_index,
                               start=False)

        # Open PyAudio output stream
        output_stream = pa.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                output=True,
                                frames_per_buffer=CHUNK,
                                output_device_index=output_device_index,
                                start=False)

        # Start the streams
        input_stream.start_stream()
        output_stream.start_stream()
        logging.info("Audio streams started.")

        # Start threads
        threads = []
        read_thread = threading.Thread(target=read_audio, name="ReadThread")
        process_thread = threading.Thread(target=process_audio, name="ProcessThread")
        playback_thread = threading.Thread(target=playback_audio, name="PlaybackThread")

        for t in [read_thread, process_thread, playback_thread]:
            t.daemon = True
            t.start()
            threads.append(t)

        logging.info("Audio processing threads started.")

    except Exception as e:
        logging.error(f"Failed to start audio processing: {e}")
        stop_processing()

def stop_processing():
    global input_stream, output_stream, threads, processing_stop_flag, playback_stop_flag

    logging.info("Stopping audio processing...")
    processing_stop_flag.set()  # Signal to stop read and process threads

    # Wait for read and process threads to finish
    for t in threads[:2]:  # Assuming first two threads are read and process
        t.join(timeout=1)
        if t.is_alive():
            logging.warning(f"{t.name} did not terminate gracefully.")

    # At this point, playback_audio thread is still running to drain the queue
    playback_stop_flag.set()  # Signal playback_audio to stop after draining

    # Wait for playback_audio thread to finish
    playback_thread = threads[2]
    playback_thread.join(timeout=5)
    if playback_thread.is_alive():
        logging.warning("PlaybackThread did not terminate gracefully.")

    # Close the streams
    try:
        if input_stream.is_active():
            input_stream.stop_stream()
        input_stream.close()
        logging.info("Input stream closed.")
    except Exception as e:
        logging.error(f"Error closing input stream: {e}")

    try:
        if output_stream.is_active():
            output_stream.stop_stream()
        output_stream.close()
        logging.info("Output stream closed.")
    except Exception as e:
        logging.error(f"Error closing output stream: {e}")

    # Clear the threads list
    threads = []
    logging.info("Audio processing stopped.")

def toggle_processing():
    if toggle_button.config('text')[-1] == 'Start':
        toggle_button.config(text='Stop')
        start_processing()
    else:
        toggle_button.config(text='Start')
        stop_processing()

def read_audio():
    """Thread function to read audio frames from the microphone."""
    logging.info("Read thread started.")
    try:
        while not processing_stop_flag.is_set():
            try:
                audio_data = input_stream.read(CHUNK, exception_on_overflow=False)
                read_queue.put(audio_data, timeout=0.5)
            except queue.Full:
                logging.warning("Read queue is full. Dropping frame.")
            except Exception as e:
                logging.error(f"Error reading audio input: {e}")
                processing_stop_flag.set()
                break
        # Signal the processing thread that reading is done
        read_queue.put(None)
        logging.info("Read thread terminated.")
    except Exception as e:
        logging.error(f"Exception in read_audio thread: {e}")
        processing_stop_flag.set()

def process_audio():
    """Thread function to process audio frames."""
    logging.info("Process thread started.")
    try:
        while not processing_stop_flag.is_set() or not read_queue.empty():
            try:
                audio_data = read_queue.get(timeout=1)
                if audio_data is None:
                    # End of audio data
                    process_queue.put(None)
                    break
                start_time = time.perf_counter()
                
                # Convert bytes to numpy array and normalize
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Process audio using the denoiser
                clean_audio, new_sr = run_model(session, audio_np, RATE, batch_process_chunks=False)

                # Convert back to numpy and int16
                clean_audio_np = clean_audio * 32768.0
                clean_audio_np = np.clip(clean_audio_np, -32768, 32767).astype(np.int16)
                clean_audio_bytes = clean_audio_np.tobytes()

                processing_time = time.perf_counter() - start_time
                logging.info(f"Processing time: {processing_time*1000:.2f} ms")

                try:
                    process_queue.put(clean_audio_bytes, timeout=0.5)
                except queue.Full:
                    logging.warning("Process queue is full. Dropping frame.")
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in process_audio: {e}")
                processing_stop_flag.set()
                break

        # Signal main thread that processing is done
        process_queue.put(None)
        logging.info("Process thread terminated.")
    except Exception as e:
        logging.error(f"Exception in process_audio thread: {e}")
        processing_stop_flag.set()

def playback_audio():
    """Thread to play back audio."""
    logging.info("Playback thread started.")
    try:
        while not (processing_stop_flag.is_set() and process_queue.empty()):
            try:
                clean_audio_bytes = process_queue.get(timeout=1)
                if clean_audio_bytes is None:
                    # End of audio data
                    break
                output_stream.write(clean_audio_bytes, exception_on_underflow=False)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error playing audio: {e}")
                break
        logging.info("Playback thread terminating after draining queue.")
    except Exception as e:
        logging.error(f"Exception in playback_audio thread: {e}")

# Set up GUI
root = tk.Tk()
root.title("Audio Denoiser")

# Initialize device lists
input_devices, output_devices = get_audio_devices()

# Input device dropdown
input_device_var = tk.StringVar()
# Set default input device
if default_input_device_name in input_devices:
    input_device_var.set(default_input_device_name)
elif input_devices:
    input_device_var.set(input_devices[0])
else:
    input_device_var.set("")

input_label = ttk.Label(root, text="Input Device:")
input_label.grid(row=0, column=0, padx=5, pady=5, sticky='e')

input_dropdown = ttk.Combobox(root, textvariable=input_device_var, values=input_devices, state="readonly", width=50)
input_dropdown.grid(row=0, column=1, padx=5, pady=5)

# Output device dropdown
output_device_var = tk.StringVar()
# Set default output device
if default_output_device_name in output_devices:
    output_device_var.set(default_output_device_name)
elif output_devices:
    output_device_var.set(output_devices[0])
else:
    output_device_var.set("")

output_label = ttk.Label(root, text="Output Device:")
output_label.grid(row=1, column=0, padx=5, pady=5, sticky='e')

output_dropdown = ttk.Combobox(root, textvariable=output_device_var, values=output_devices, state="readonly", width=50)
output_dropdown.grid(row=1, column=1, padx=5, pady=5)

# Refresh Devices button
refresh_button = ttk.Button(root, text="Refresh Audio Devices", command=update_device_lists)
refresh_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Toggle button
toggle_button = ttk.Button(root, text="Start", command=toggle_processing)
toggle_button.grid(row=3, column=0, columnspan=2, padx=5, pady=10)

def on_closing():
    if toggle_button['text'] == 'Stop':
        stop_processing()
    pa.terminate()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Run the Tkinter event loop
root.mainloop()
