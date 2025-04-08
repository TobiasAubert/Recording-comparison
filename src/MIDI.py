import pretty_midi
import numpy as np
from dtw import dtw

## what is calculated
# Onset: Overall Timing Accuracy, Tempo Variations


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

# Load MIDI files (replace with your file paths)
# reference_midi = "C:/Users/tobia/OneDrive/AA Uni/ISPW_Erlacher/Bacherlorarbeit/MIDI Aufnahmen/Auimimiäntli1_1.mid" 
# performance_midi = "C:/Users/tobia/OneDrive/AA Uni/ISPW_Erlacher/Bacherlorarbeit/MIDI Aufnahmen/Auimimiäntli1_2.mid"

reference_midi = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/House of the Rising Sun.mid"
performance_midi = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/BE01CL/BE01CL_1_Stück_1.mid"
performance1_midi = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/DI03CA/DI03CA_1_Stück_1.mid"
SU13BA_1_1 = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/SU13BA/SU13BA_1_Stück_1.mid"
SU13BA_2_1 = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/SU13BA/SU13BA_2_Stück_1.mid"
SU13BA_2_2 = "C:/Users/tobia/OneDrive - Universitaet Bern/25FS_BSc_AubertTobias_ARPiano/12_Rohdaten/SU13BA/SU13BA_2_Stück_2.mid"

# Extract note onsets
ref_onsets = extract_onsets(reference_midi)
perf_onsets = extract_onsets(performance_midi)
perf_onsets1 = extract_onsets(performance1_midi)
SU13BA_1_1 = extract_onsets(SU13BA_1_1)
SU13BA_2_1 = extract_onsets(SU13BA_2_1)
SU13BA_2_2 = extract_onsets(SU13BA_2_2)

# Reshape the arrays for DTW (each must be 2D: number of elements x feature dimension)
ref_onsets = ref_onsets.reshape(-1, 1)
perf_onsets = perf_onsets.reshape(-1, 1)
perf_onsets1 = perf_onsets1.reshape(-1, 1)
SU13BA_1_1 = SU13BA_1_1.reshape(-1, 1)
SU13BA_2_1 = SU13BA_2_1.reshape(-1, 1)
SU13BA_2_2 = SU13BA_2_2.reshape(-1, 1)

# Define a simple distance metric for note onsets (absolute difference)
def onset_distance(x, y):
    return abs(x - y)

# Compute DTW alignment between the two onset sequences
alignment = dtw(ref_onsets, perf_onsets)
alignment1 = dtw(ref_onsets, perf_onsets1)
alignment2 = dtw(ref_onsets, SU13BA_1_1)
alignment3 = dtw(ref_onsets, SU13BA_2_1)
alignment4 = dtw(ref_onsets, SU13BA_2_2)

print("DTW distance between the MIDI files:", alignment.distance)
print("DTW distance between the MIDI files:", alignment1.distance)
print("DTW distance between the MIDI files:", alignment2.distance)
print("DTW distance between the MIDI files:", alignment3.distance)
print("DTW distance between the MIDI files:", alignment4.distance)