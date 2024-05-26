import numpy as np
import cv2
import scipy.io.wavfile
import cupy as cp
from tkinter import filedialog, Tk
import os
import subprocess

def map_pixel_to_frequency(channel, min_freq, max_freq):
    frequencies = min_freq + (max_freq - min_freq) * (channel / 255.0)
    return frequencies

def encode_image():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if not file_path:
        print("No file selected.")
        return
    
    img = cv2.imread(file_path)
    if img is None:
        print("Image not found.")
        return

    height, width, _ = img.shape

    frequency_ranges = {
        'r': (200, 4200),
        'g': (8400, 12400),
        'b': (4300, 8300)
    }

    b_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    r_channel = img[:, :, 2]

    r_frequencies = map_pixel_to_frequency(r_channel, frequency_ranges['r'][0], frequency_ranges['r'][1])
    g_frequencies = map_pixel_to_frequency(g_channel, frequency_ranges['g'][0], frequency_ranges['g'][1])
    b_frequencies = map_pixel_to_frequency(b_channel, frequency_ranges['b'][0], frequency_ranges['b'][1])

    channels = {
        'r': r_frequencies,
        'g': g_frequencies,
        'b': b_frequencies
    }

    SAMPLE_RATE = 192000
    DURATION = 0.002

    combined_waveform = []

    for color, frequencies in channels.items():
        freqs_gpu = cp.asarray(frequencies.flatten())
        t = cp.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        waveforms_gpu = cp.sin(2 * cp.pi * freqs_gpu[:, cp.newaxis] * t)
        full_waveform_gpu = waveforms_gpu.reshape(-1)
        full_waveform_gpu /= cp.max(cp.abs(full_waveform_gpu))

        combined_waveform.append(cp.asnumpy(full_waveform_gpu))

    combined_waveform = np.concatenate(combined_waveform)
    wav_file_path = "combined_sound.wav"
    scipy.io.wavfile.write(wav_file_path, SAMPLE_RATE, combined_waveform)

    print("Waveform saved to combined_sound.wav")
    
if __name__ == "__main__":
    encode_image()
