# ---getting score ignoring loudness--- can use as example but loudness does not affect onset
#Use RMS to get loudness and normalize by max RMS value
# rms_original = librosa.feature.rms(y= original_audio)
# rms_contestant = librosa.feature.rms(y=contestant_audio)

# # Normalize each audio signal by its max RMS value
# original_audio_normalized = original_audio / rms_original.max()
# contestant_audio_normalized = contestant_audio / rms_contestant.max()

# # Detect onsets on the normalized audio signals
# original_onsets_normalized = librosa.onset.onset_detect(y=original_audio_normalized, sr=sr)
# contestant_onsets_normalized = librosa.onset.onset_detect(y=contestant_audio_normalized, sr=sr)

# #Convert onset frames to actual time
# original_times_normalized = librosa.frames_to_time(original_onsets_normalized, sr=sr)
# contestant_times_normalized = librosa.frames_to_time(contestant_onsets_normalized, sr=sr)

# # Reshape to 2D arrays with 1 row
# original_times_normalized = original_times_normalized.reshape(1, -1)  # Shape (1, N)
# contestant_times_normalized = contestant_times_normalized.reshape(1, -1)  # Shape (1, M)

# #Apply Dynamic Time Warping to normalized data
# cost_matrix_normalized, alignment_path_normalized = dtw(original_times_normalized[:, np.newaxis], contestant_times_normalized[:, np.newaxis], subseq=True, backtrack=True)
# score_normalized = cost_matrix_normalized[-1, -1]  # The final alignment cost at the end of the path
# print("score normalized", score_normalized)
