from pydub import AudioSegment
from pydub.silence import split_on_silence
from os import makedirs, listdir, path
from os.path import exists
from essentia.standard import MonoLoader


def segmentate(audios_path):
    classes = listdir(audios_path)

    for i, animal in enumerate(classes):
        if animal.find(' ') != -1: continue  # not an animal but a set of words
        current_path = path.join(audios_path, animal)
        sounds = listdir(current_path)
        for sound in sounds:
            if not sound.lower().endswith('.wav'): continue
            AUDIO_FILE = path.join(current_path, sound)
            sound_file = AudioSegment.from_wav(AUDIO_FILE)
            try:
                audio_chunks = split_on_silence(
                    sound_file,
                    min_silence_len=2,
                    # consider it silent if quieter than -16 dBFS
                    silence_thresh=min(-16, sound_file.dBFS*7),
                    keep_silence=0
                )
            except:
                continue
            for j, chunk in enumerate(audio_chunks):
                out_file = "splitAudio/{1}/{2}chunk{0}.wav".format(j, animal, sound[:-4])
                if chunk.duration_seconds < 0.5:
                    continue
                #print(chunk.duration_seconds)
                if not exists("splitAudio/{0}".format(animal)):
                    makedirs("splitAudio/{0}".format(animal))
                #print("exporting", out_file)
                chunk.export(out_file, format="wav")

def split_audio(audio_path, split_list):
    audio = MonoLoader(filename=audio_path)()
    start = 0
    res = []
    #split list es una secuencia de segundos. cada elemento debe ser mayor que el anterior
    for i in split_list:
        # end = int(i * 44100) # porque es la frecuencia con la que se muestrea el audio
        end = int(i) + start
        # end = int(i)
        if end > audio.size or end <= start:
            return res, audio
        res.append(audio[start:end])
        start = end
    return res, audio

# segmentate("/home/migue/sonidos animales")