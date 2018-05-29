from scipy import signal
import numpy as np

def highpass_filter(y, sr, filter_stop_freq=400, filter_pass_freq=500, filter_order=1001):
    """
    :param y:  audio signal to be filtered
    :param sr: audio signal sampling rate
    :param filter_stop_freq: frequency below which the filter MUST act like a stop filter
    :param filter_pass_freq: frequency above which the filter MUST act like a pass filter
    :param filter_order: the length (number of taps) of the filter. The higher the order, usually, the better the fit
    :return: a filtered audio signal
    """
    # High-pass filter
    nyquist_rate = sr / 2.
    desired = (0, 0, 1, 1)
    bands = (0, filter_stop_freq, filter_pass_freq, nyquist_rate)
    filter_coefs = signal.firls(filter_order, bands, desired, nyq=nyquist_rate)

    # Apply high-pass filter
    y = np.ravel(y)
    filtered_audio = signal.filtfilt(filter_coefs, [1], y, padtype='constant', padlen=len(y)-1)
    return filtered_audio

# from essentia.standard import MonoLoader, MonoWriter
# import numpy as np
# from extract_audio_car import save_oscilogram
# from librosa.output import write_wav
#
# audio = MonoLoader(filename='/home/migue/dubstep.wav')()
# a = highpass_filter
# args = {'filter_stop_freq':1000, 'filter_pass_freq':1200, 'filter_order':2001}
# args = {}
# # save_oscilogram(audio, "dubstepWithoutFilter.jpg")
#
# audio = a(audio, 44100, **args)
# audio = np.array(audio)
# # save_oscilogram(audio, "dubstepWithFilter.jpg")
# write_wav("p2.wav", audio, 44100)