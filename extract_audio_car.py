import speech_recognition as sr
from essentia.standard import *
from essentia import Pool
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt
import numpy as np


def save_plots(data, save_fig_path):
    figure = plt.figure()
    figure.add_axes([0, 0, 1, 1])  # para que no pinte bordes blancos alrededor de la figura
    imshow(data, aspect='auto', origin='lower')
    # plt.plot(data)
    plt.axis('off')
    figure.savefig(save_fig_path)
    plt.close(figure)


def save_oscilogram(audio, path="oscilogram.jpg"):
    # audio = MonoLoader(filename=audio_file)()
    plt.rcParams['figure.figsize'] = (15, 6)  # set plot sizes to something larger than default
    figure = plt.figure()
    figure.add_axes([0, 0, 1, 1])  # para que no pinte bordes blancos alrededor de la figura
    plot(audio)
    figure.savefig(path)
    plt.close(figure)


def beats_extractor(audio, audio_file):
    # Compute beat positions and BPM
    rhythm_extractor = RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
    return bpm, beats


def sound_to_text(audio, audio_file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)  # read the entire audio file
        # recognize speech using Sphinx
        return r.recognize_sphinx(audio)
    except:
        return -1


def spectrogram(audio, audio_file, save_fig=True, save_fig_path=None):
    if audio_file.endswith('.wav'):
        w = Windowing(type='hann')
        spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
        pool = Pool()
        ## NOTAS -> 1 segundo de un fichero wav son aprox 90 frames y la intensidad esta dada en Hz
        for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
            win = w(frame)
            spec = spectrum(win)
            pool.add('spec', spec)
        aggrPool = PoolAggregator(defaultStats=['mean'])(pool)
        a = sum(aggrPool['spec.mean'].T)/aggrPool['spec.mean'].T.shape[0]
        # a = aggrPool['spec.mean'].T
        # b = np.zeros(pool['spec'].T.shape)
        b = np.array(pool['spec'].T)
        # for iterator1, i in enumerate(pool['spec'].T):
        #     for iterator2, j in enumerate(i):
        #         # if j > a[iterator1]/2:
        #         if j > a/2 and j > 0.015:
        #             b[iterator1][iterator2] = j
        # b = np.array([i for i in b if i.max() > 0.01])
        # no para el nuevo dataset de bats
        b = remove_initial_zeros(b)
        b = b.tolist()
        b.reverse()
        b = remove_initial_zeros(b)
        b.reverse()
        b = np.array(b)
        if save_fig:
            if not save_fig_path:
                save_fig_path = audio_file.replace('.wav', '_spec.jpg')
            save_plots(b, save_fig_path)
        return b[:200, :200].tolist()


def melCoefs(audio, audio_file, save_fig=True):
    if audio_file.endswith('.wav'):
        mfcc_pool = __mfccs__(audio)
        # b = np.compress([True]*40, np.array(mfcc_pool['mfcc'].T), axis=1)
        # b = np.array(mfcc_pool['mfcc'].T)
        # b = mfcc(audio, sr=44100)
        b = np.array(mfcc_pool['mfcc'].T)
        data = b
        print(b.shape)
        # b = np.delete(b, range(40, b.shape[1], 1))
        if save_fig:
            # data = mfcc_pool['mfcc'].T
            save_fig_path = audio_file.replace('.wav', '_mfcc.jpg')
            save_plots(data, save_fig_path)
        return b.tolist()


def meanMFCCs(audio, audio_file=None):
    mfcc_pool = __mfccs__(audio)
    aggrPool = PoolAggregator(defaultStats=['mean', 'stdev'])(mfcc_pool)
    a = aggrPool['mfcc.mean'].T
    return np.array(a).tolist()


def __mfccs__(audio):
    w = Windowing(type='hann')
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = MFCC()
    mfcc_pool = Pool()
    ## NOTAS -> 1 segundo de un fichero wav son aprox 90 frames y la intensidad esta dada en Hz
    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        spec = spectrum(w(frame))
        mfcc_bands, mfcc_coeffs = mfcc(spec)
        mfcc_pool.add('mfcc', mfcc_coeffs[1:])
        mfcc_pool.add('mfcc_bands', mfcc_bands)
    return mfcc_pool


def remove_initial_zeros(arr):
    for i, a in enumerate(arr):
        # if max(a) > 0.01:
        if max(a) > 0: # cambio para el nuevo dataset de bats
            break
    return arr[i:]


FUNCS_ = {'beats': beats_extractor, 'text': sound_to_text, 'spectrogram': spectrogram, 'mfccs': melCoefs, 'meanMFCC': meanMFCCs}


def do_from_name(func_name, audio, audio_file):
    # try:
    #     return FUNCS_[func_name](audio_file)
    # except:
    #     return -1
    return FUNCS_[func_name](audio, audio_file)