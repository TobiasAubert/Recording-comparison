import librosa
import numpy as np
from scipy.spatial.distance import cdist
from librosa.sequence import dtw

#path to the audio file

original_path = "C:/Users/tobia/OneDrive/Dokumente/Audacity/1entli.wav"
contestant_path = "C:/Users/tobia/OneDrive/Dokumente/Audacity/2entli.wav"

original_audio, sr = librosa.load(original_path, sr=None)
contestant_audio, _ = librosa.load(contestant_path , sr=sr)

## ---------Rhythm--------- ##
# ---getting the Data---
# Detect onsets (rythm) in the audio files
original_onsets = librosa.onset.onset_detect(y=original_audio, sr=sr)
contestant_onsets = librosa.onset.onset_detect(y=contestant_audio, sr=sr)

print("original_onsets", original_onsets)
print("contestant_onsets", contestant_onsets)

# Convert onset frames to times (when each note (onset) occurs in time during the recordings)
original_times = librosa.frames_to_time(original_onsets, sr=sr)
contestant_times = librosa.frames_to_time(contestant_onsets, sr=sr)

# Reshape to 2D arrays with 1 row (required by DTW) to get alignment_paht and cost matrix else [0,0]
original_times = original_times.reshape(1, -1)  # Shape (1, N)
contestant_times = contestant_times.reshape(1, -1)  # Shape (1, M)

# ---getting score not normalizer (timing and dynamic)---

# Apply Dynamic Time Warping
# mesures Correct Order of Key Presses, Tempo Changes, Measuring Correctness
cost_matrix, alignment_path = dtw(original_times[:, np.newaxis], contestant_times[:, np.newaxis], subseq=True, backtrack=True) # score lower is better (d , wp = .....) 
score = cost_matrix[-1, -1]  # The final alignment cost at the end of the path
print("score", score)


## ---------Rhythm--------- ##
# Extract chroma features
original_chroma = librosa.feature.chroma_cqt(y=original_audio, sr=sr) # cqt is robuster than stft
contestant_chroma = librosa.feature.chroma_cqt(y=contestant_audio, sr=sr)

# print("original_chroma", original_chroma)
# print("contestant_chroma", contestant_chroma)

# Reshape to 2D arrays with 1 row
original_chroma = original_chroma.reshape(1, -1)  # Shape (1, N)
contestant_chroma = contestant_chroma.reshape(1, -1)  # Shape (1, M)

# print("original_chroma_reshaped", original_chroma)
# print("contestant_chroma_reshaped", contestant_chroma)

# Use librosa's DTW function to compute alignment
D, wp = librosa.sequence.dtw(X=original_chroma, Y=contestant_chroma, metric='euclidean',  subseq=True, backtrack=True)
score_chroma = D[-1, -1]  # The final alignment cost at the end of the path
print("score_chroma", score_chroma)




# # ---TEST---
# #score split into singele components
# # missing notes
# missing_notes = len(original_times) - len(contestant_times)


# seq1 = np.array([1, 2, 3, 4, 5 , 18])
# seq2 = np.array([1, 2, 3, 4, 20 ])

# # Reshape to 2D arrays with 1 row
# seq1 = seq1.reshape(1, -1)  # Shape (1, N)
# seq2 = seq2.reshape(1, -1)  # Shape (1, M)

# cost_matrix3, alignment_path3 = dtw(seq1[:, np.newaxis], seq2[:, np.newaxis], subseq=True, backtrack=True)
# print("Alignment path: test", alignment_path3)
# print("score test", cost_matrix3[-1, -1])