import os
import sys

import librosa
import nolds
import numpy as np
import pandas as pd
import parselmouth
from google.cloud import storage
from parselmouth.praat import call
from pyentrp import entropy
from pysndfx import AudioEffectsChain
from scipy import signal
from scipy.io import wavfile


def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID)
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    meanF0 = call(pitch, "Get mean", 0, 0, unit)
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    voice_report = call([sound, pitch, pointProcess], "Voice report", 0.0, 0.0, f0min, f0max, 1.3, 1.6, 0.03, 0.45)

    return meanF0, stdevF0, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, voice_report


def reduce_noise(name):
    y, sr = librosa.load(name)

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    threshold_h = round(np.median(cent)) * 1.5
    threshold_l = round(np.median(cent)) * 0.1
    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(gain=-12.0,
                                                                                                      frequency=threshold_h,
                                                                                                      slope=0.5)  # .limiter(gain=6.0)
    y_clean = less_noise(y)

    y_trimmed, index = librosa.effects.trim(y_clean, top_db=20, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y_clean) - librosa.get_duration(y_trimmed)

    newfile = name[len(name) - 29:len(name)]
    newfilename = newfile[0:len(newfile) - 4]
    destination = "/opt/lampp/htdocs/testing/uploads/" + newfilename + '.wav'
    os.remove(destination)
    librosa.output.write_wav(destination, y_trimmed, sr)


AudioFile_path = sys.argv[1]
reduce_noise(AudioFile_path)
sample_rate, samples = wavfile.read(AudioFile_path)
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)
sound = parselmouth.Sound(AudioFile_path)
DFA = nolds.dfa(times)
PPE = entropy.shannon_entropy(times)

(meanF0, stdevF0, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer,
 apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, voice_report) = measurePitch(sound, 75, 500, "Hertz")

voice_report = voice_report.strip()

new_v = []

new_v = voice_report.split('\n')

hnr = float(new_v[32][34:len(new_v[32]) - 3])
nhr = float(new_v[31][34:len(new_v[31])])

df_1 = pd.DataFrame(np.column_stack(
    [localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer,
     aqpq5Shimmer, apq11Shimmer, ddaShimmer, nhr, hnr, DFA, PPE]),
    columns=['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer',
             'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR',
             'HNR', 'DFA', 'PPE'])

df = pd.read_csv('Net_Model/parkinson_dataset_1.csv')
X = df.iloc[:, 6:21].values
Y = df.iloc[:, 4:6].values
vertical_stack = pd.concat([df.iloc[:, 6:21], df_1], axis=0)
X_new = vertical_stack.iloc[:, 0:15].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
X_new = sc.fit_transform(X_new)
y_new = sc.fit_transform(Y)

from keras.models import load_model

best_model = load_model('Net_Model/weights-improvement-998-0.0021.hdf5', compile=False)
Y = best_model.predict(X_new[5874:5875])
Y_pred_org = sc.inverse_transform(Y)

MOTOR_UPDRS = Y_pred_org[0][0]
TOTAL_UPDRS = Y_pred_org[0][1]

Result = "Patient's Motor Updrs Value : %s and Total Updrs Value : %s" % (MOTOR_UPDRS, TOTAL_UPDRS)

# For creating File
filename = AudioFile_path[len(AudioFile_path) - 29:len(AudioFile_path)]
filename = filename[0:len(filename) - 4]
ele = [x for x in filename.split("_")]
userID = ele[1]
RecordingID = ele[2]
Resultfilename = "Results_{0}_{1}.txt".format(userID, RecordingID)
f = open("Results/" + Resultfilename, "w")
f.write(Result)
f.close()
print("File is created !!!!!!!")

# For Uploading to Storage Bucket
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pd-health-2020-827a1294ce8a.json"
client = storage.Client()
bucket = client.get_bucket("pd-health-2020.appspot.com")
imagePath = "Results/" + Resultfilename
imageBlob = bucket.blob(userID + "/" + RecordingID + "/" + Resultfilename)
imageBlob.upload_from_filename(imagePath)
print("File Uploaded in Firebase Storage !!!!!")