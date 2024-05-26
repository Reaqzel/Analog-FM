import numpy as np
import scipy.io.wavfile
import scipy.fftpack
import cv2
import os
from tkinter import filedialog, Tk


SAMPLE_RATE = 192000
DURATION = 0.002

def freq_to_pixel(freq, min_freq, max_freq):
    pixel_val = (freq - min_freq) / (max_freq - min_freq) * 255
    return np.clip(pixel_val, 0, 255)

def decode_image():
    
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        print("No file selected.")
        return

    sample_rate, audio = scipy.io.wavfile.read(file_path)
    samples_per_pixel = int(DURATION * SAMPLE_RATE)
    num_pixels_per_channel = len(audio) // 3 // samples_per_pixel
    image_size = int(np.sqrt(num_pixels_per_channel))

    channels = {
        'r': (200, 4200),
        'g': (8400, 12400),
        'b': (4300, 8300)
    }

    reconstructed_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    for color, (MIN_FREQ, MAX_FREQ) in zip(['r', 'g', 'b'], channels.values()):
        offset = {'r': 0, 'g': 1, 'b': 2}[color] * num_pixels_per_channel * samples_per_pixel
        channel_audio = audio[offset:offset + num_pixels_per_channel * samples_per_pixel]

        for i in range(image_size):
            for j in range(image_size):
                start_idx = (i * image_size + j) * samples_per_pixel
                end_idx = start_idx + samples_per_pixel

                pixel_audio = channel_audio[start_idx:end_idx]
                fft_result = scipy.fftpack.fft(pixel_audio)
                freqs = scipy.fftpack.fftfreq(len(pixel_audio), 1 / SAMPLE_RATE)

                peak_freq = freqs[np.argmax(np.abs(fft_result))]
                reconstructed_image[i, j, {'r': 2, 'g': 1, 'b': 0}[color]] = freq_to_pixel(peak_freq, MIN_FREQ, MAX_FREQ)

    cv2.imwrite('Reconstructed_Image.png', reconstructed_image)
    print("Image reconstructed and saved as Reconstructed_Image.png")

if __name__ == "__main__":
    decode_image()
