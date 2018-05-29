import os
import librosa
import numpy as np

p = "/media/migue/5E7A52CD7A52A197/Users/black beard/Desktop/Birds/xcmeta.csv"
audios_path = "/media/migue/5E7A52CD7A52A197/Users/black beard/Desktop/xccoverbl_WAV/"

with open(p) as file:
    s = file.read()
    s = s.split('\n')
    s = [i.split('\t') for i in s]

id_pos = s[0].index('id')
gen_pos = s[0].index('gen')
sp_pos = s[0].index('sp')
en_pos = s[0].index('en')
type_pos = s[0].index('type')

save_path = "/media/migue/5E7A52CD7A52A197/Users/black beard/Desktop/xccoverbl_WAV_classes/"
os.makedirs(save_path, exist_ok=True)
audios_files = os.listdir(audios_path)

for i in s[1:]:
    try:
        p = os.path.abspath(save_path + i[en_pos])
        os.makedirs(p, exist_ok=True)
        audio = i[id_pos]
        for j in audios_files:
            if j.startswith("xc"+audio):
                to_save, sr = librosa.load(audios_path + j, sr=44100)
                # librosa.output.write_wav(p, np.array(to_save, dtype=float), 44100)
                librosa.output.write_wav(p + '/' + j, to_save, 44100)
    except Exception as e:
        print(e)