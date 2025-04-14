import os
import pretty_midi
import pandas as pd
import numpy as np
from dtw import dtw

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

                # Define a simple distance metric for note onsets (absolute difference)
                def onset_distance(x, y):
                    return abs(x - y)

                # Compute DTW alignment between the two onset sequences
                alignment = dtw(ref_onsets, perf_onsets)
                score = alignment.distance


                # prepare the data for DataFrame
                info = {
                    'Participant_ID': participant_id,
                    'Appointment': appointment,
                    'Song': song,
                    'Attempt': attempt,
                    'Score': score,
                }
                data.append(info)


                midi = pretty_midi.PrettyMIDI(file_path)
                midi_data_list.append((file_path, midi))
                print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")

 


df_songs = pd.DataFrame(data)
# Set options to show all rows and columns
pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # No limit on display width
pd.set_option('display.max_colwidth', None)  # Don't truncate column contents

# Create label column for wide format
df_songs['label'] = df_songs['Song'] + '_'  + df_songs['Appointment'].astype(str) + '-' +  df_songs['Attempt'].astype(str)

# Pivot to get one row per participant
df_pivot = df_songs.pivot(index='Participant_ID', columns='label', values='Score')

# Optional: Reset index for a flat DataFrame
df_pivot.reset_index(inplace=True)

print(df_songs)
# print(df_pivot)

# With ID included
blues_with_id = df_pivot[['Participant_ID'] + [col for col in df_pivot.columns if col.startswith('Blues')]]
blues_filtered = blues_with_id.dropna(subset=['Blues_1-1'])
stueck_with_id = df_pivot[['Participant_ID'] + [col for col in df_pivot.columns if col.startswith('Stück')]]
stueck_filtered = stueck_with_id.dropna(subset=['Stück_1-1'])

print(blues_filtered)
print(stueck_filtered)


print(f"\nTotal MIDI files loaded (excluding 'Finger'): {len(midi_data_list)}")


