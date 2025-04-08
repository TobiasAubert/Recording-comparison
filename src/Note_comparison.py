import mido
import editdistance
        
def manhattan_distance(list1, list2):
    distance = 0
    for sub1, sub2 in zip(list1, list2):
        # print(sub1, sub2)
        sub_distance = 0
        for x, y in zip(sub1, sub2):
            sub_distance += abs(x - y)
        distance += sub_distance
    return distance

def compare_midi_files2(file1, file2):
    # Load midi files
    mid1 = mido.MidiFile(file1)
    mid2 = mido.MidiFile(file2)

    # Extract notes from midi files and group adjacent pitches together
    notes1 = []
    for msg in mido.merge_tracks(mid1.tracks):
        if 'note_on' in msg.type:
            pitch = msg.note
            if notes1 and pitch == notes1[-1][-1] + 1:
                # Append pitch to last group
                notes1[-1].append(pitch)
            else:
                # Create new group for pitch
                notes1.append([pitch])

    notes2 = []
    for msg in mido.merge_tracks(mid2.tracks):
        if 'note_on' in msg.type:
            pitch = msg.note
            if notes2 and pitch == notes2[-1][-1] + 1:
                # Append pitch to last group
                notes2[-1].append(pitch)
            else:
                # Create new group for pitch
                notes2.append([pitch])

    # Calculate similarity for each group of pitches
    print("len1", notes1.__len__())
    print("len2", notes2.__len__())
    similarity_scores = []
    return manhattan_distance(notes1, notes2)

original_file_path = "C:/Users/tobia/OneDrive/AA Uni/ISPW_Erlacher/Bacherlorarbeit/MIDI Aufnahmen/Auimimiäntli2_1.mid"
generated_file_path = "C:/Users/tobia/OneDrive/AA Uni/ISPW_Erlacher/Bacherlorarbeit/MIDI Aufnahmen/Auimimiäntli1_1.mid"

similarity = compare_midi_files2(original_file_path,generated_file_path)

print(similarity) # higher it is, the less similar files are

