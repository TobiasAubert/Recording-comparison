seq_1_1 = ['G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4']
seq_1_2 = ['G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4', 'F4', 'E4', 'G4', 'G4', 'C4']


# Your expected pattern (adjust to match what they were supposed to play)
#['F4', 'C4', 'E4', 'D4', 'F4']
expected_sequence = ['G4', 'C4', 'F4', 'E4', 'G4',]
sequence_len = len(expected_sequence)


correct_sequences_1_1 = 0
for i in range(len(seq_1_1) - sequence_len + 1):
    if seq_1_1[i:i+sequence_len] == expected_sequence:
        correct_sequences_1_1 += 1

correct_sequences_1_2 = 0
for i in range(len(seq_1_2) - sequence_len + 1):
    if seq_1_2[i:i+sequence_len] == expected_sequence:
        correct_sequences_1_2 += 1

print(f"Correct sequences for 1_1: {correct_sequences_1_1}")
print(f"Correct sequences for 1_2: {correct_sequences_1_2}")


# c d e f g a h c 