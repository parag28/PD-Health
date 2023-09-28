import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import nolds
from scipy import signal
from scipy.io import wavfile
from pyentrp import entropy
import sys

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    #harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    #hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    voice_report = call([sound,pitch,pointProcess], "Voice report", 0.0, 0.0, f0min, f0max, 1.3, 1.6, 0.03, 0.45)

    return meanF0, stdevF0, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, voice_report

AudioFile_path = sys.argv[1]
sample_rate, samples = wavfile.read(AudioFile_path)
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)
sound = parselmouth.Sound(AudioFile_path)
DFA = nolds.dfa(times)
PPE = entropy.shannon_entropy(times)
(meanF0, stdevF0, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer, voice_report) = measurePitch(sound, 75, 500, "Hertz")

voice_report = voice_report.strip()

hnr = voice_report[984:989]
nhr = voice_report[941:953]

# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler()
# DFA = sc.fit_transform(DFA)
# PPE = sc.fit_transform(PPE)

df_1 = pd.DataFrame(np.column_stack([localJitter,localabsoluteJitter,rapJitter,ppq5Jitter,ddpJitter,localShimmer,localdbShimmer,apq3Shimmer,aqpq5Shimmer,apq11Shimmer,ddaShimmer,nhr,hnr,DFA,PPE]),
                               columns=['Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','DFA','PPE'])
        
df = pd.read_csv('/home/subhranil/Pictures/PDProject/Net_Model/parkinson_dataset_1.csv')
X = df.iloc[:, 6:21].values
Y = df.iloc[:, 4:6].values
vertical_stack = pd.concat([df.iloc[:, 6:21], df_1], axis=0)
X_new = vertical_stack.iloc[:, 0:15].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_new = sc.fit_transform(X_new)
y_new = sc.fit_transform(Y) 
import keras
from keras.models import load_model   
best_model = load_model('/home/subhranil/Pictures/PDProject/Net_Model/weights-improvement-998-0.0021.hdf5',compile=False)
Y = best_model.predict(X_new[5874:5875])
Y_pred_org = sc.inverse_transform(Y)
MOTOR_UPDRS = Y_pred_org[0][0]
TOTAL_UPDRS = Y_pred_org[0][1]

Result = "Patient's Motor Updrs Value : %s and Total Updrs Value : %s" %(MOTOR_UPDRS,TOTAL_UPDRS)
print(Result)