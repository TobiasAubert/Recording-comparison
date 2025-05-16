import os
import pretty_midi
import pandas as pd
import numpy as np
from dtw import dtw
import traceback


# save_path = "C:/Users/tobia/Desktop/Recording-comparison/src/Examples_tests/Data/blue_score.csv"
# save_path = "C:/Users/tobia/Desktop/Recording-comparison/src/Examples_tests/Data/risingsun_score.csv"

def extract_onsets(midi_file):
    """
    Loads a MIDI file and extracts a sorted numpy array of note onset times
    from all non-drum instruments.
    Ignores pitch and velocity.
    """
    pm = pretty_midi.PrettyMIDI(midi_file)
    onsets = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            onsets.append(note.start)
    return np.array(sorted(onsets))

def extract_onsets_and_pitches(midi_file):
    pm = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append((note.start, note.pitch))
    return np.array(sorted(notes, key=lambda x: x[0]))  # sort by onset time



# Change this to your actual folder path
root_folder = r'C:\Users\tobia\OneDrive - Universitaet Bern\25FS_BSc_AubertTobias_ARPiano\12_Rohdaten'

# List to collect loaded MIDI data
midi_data_list = []
data = []
rising_sun = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/House of the Rising Sun.mid"
rising_sun_rechts = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/Rechts House of the Rising Sun.mid"
blues = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/Blues No1.mid"

# Walk through all subfolders
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.lower().endswith(('.mid', '.midi')) and 'finger' not in filename.lower():
            file_path = os.path.join(dirpath, filename)

            try:
                # Extract participant ID, appointment, song, and attempt from filename
                participant_id = filename.split('_')[0]
                appointment = filename.split('_')[1]
                song = filename.split('_')[2]
                attempt = filename.split('_')[3].split('.')[0]  # Remove file extension

                # parts = filename.split('_')
                # if len(parts) < 4:
                #     print(f"⚠️  Skipping file with unexpected name format: {filename}")
                #     continue  # Datei überspringen und mit nächster fortfahren

                # participant_id = parts[0]
                # appointment = parts[1]
                # song = parts[2]
                # attempt = parts[3].split('.')[0]
                
                #claculate the score how well the song was played
                if song == "Blues":
                    reference_midi = blues
                elif song == "Stück":
                    if int(appointment) > 3:
                        reference_midi = rising_sun_rechts
                    else:
                        reference_midi = rising_sun
                
                
                performance_midi = file_path

                # Extract note onsets
                ref_onsets = extract_onsets(reference_midi)
                perf_onsets = extract_onsets(performance_midi)

            
                # Reshape the arrays for DTW (each must be 2D: number of elements x feature dimension)
                ref_onsets = ref_onsets.reshape(-1, 1)
                perf_onsets = perf_onsets.reshape(-1, 1)

            
                # Compute DTW alignment between the two onset sequences
                alignment = dtw(ref_onsets, perf_onsets)
                score = alignment.distance


                # ---- Extract pitch information for the performance
                # Extract note onsets and pitches
                ref_pitch = extract_onsets_and_pitches(reference_midi)
                perf_pitch = extract_onsets_and_pitches(performance_midi)

                # Reshape the arrays for DTW (each must be 2D: number of elements x feature dimension)
                ref_pitch_res = ref_pitch.reshape(-1, 2).copy()  # Onset time and pitch
                perf_pitch_res = perf_pitch.reshape(-1, 2).copy()  # Onset time and pitch

                
                # Compute DTW alignment between the two pitch sequences
                pitch_alignment = dtw(ref_pitch_res, perf_pitch_res)
                score_pitch = pitch_alignment.distance


            
                # prepare the data for DataFrame
                info = {
                    'Participant_ID': participant_id,
                    'Appointment': appointment,
                    'Song': song,
                    'Attempt': attempt,
                    'Score': score,
                    'Score pitch': score_pitch,
                }
                data.append(info)


                midi = pretty_midi.PrettyMIDI(file_path)
                midi_data_list.append((file_path, midi))
                print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"❌ Failed to load {file_path}:\n{traceback.format_exc()}")

 


df_songs = pd.DataFrame(data)
# Set options to show all rows and columns
pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # No limit on display width
pd.set_option('display.max_colwidth', None)  # Don't truncate column contents

# Create label column for wide format
df_songs['label'] = df_songs['Song'] + '_'  + df_songs['Appointment'].astype(str) + '-' +  df_songs['Attempt'].astype(str)

# Pivot to get one row per participant
df_pivot = df_songs.pivot(index='Participant_ID', columns='label', values='Score pitch')

# Optional: Reset index for a flat DataFrame
df_pivot.reset_index(inplace=True)



print(df_songs)
# print(df_pivot)

# With ID included
blues_with_id = df_pivot[['Participant_ID'] + [col for col in df_pivot.columns if col.startswith('Blues')]]
blues_filtered = blues_with_id.dropna(subset=['Blues_1-1'])
stueck_with_id = df_pivot[['Participant_ID'] + [col for col in df_pivot.columns if col.startswith('Stück')]]
stueck_filtered = stueck_with_id.dropna(subset=['Stück_1-1'])


blues_filtered.to_csv("C:/Users/tobia/Desktop/Recording-comparison/src/Examples_tests/Data/blues_score2.csv", index=False)
stueck_filtered.to_csv("C:/Users/tobia/Desktop/Recording-comparison/src/Examples_tests/Data/risingsun_score2.csv", index=False)

print(f"\nTotal MIDI files loaded (excluding 'Finger'): {len(midi_data_list)}")


