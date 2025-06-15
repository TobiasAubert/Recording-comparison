
# 🎹 AR Piano Performance Analysis

This project analyzes MIDI piano recordings in the context of an **augmented reality (AR)** and classical training experiment. It compares timing, pitch, and finger sequence accuracy using MIDI processing and statistical analysis.

---

## 📁 Project Structure

```
src/
├── saving_JE13CL.py                     # Manual analysis for participant JE13CL
├── Setup.md                             # Experiment setup documentation
│
└── data_analysis_pipeline/
    ├── Data/                            # Output data files
    │   ├── blues_score.csv
    │   ├── blues_score2.csv
    │   ├── fingergeschicklichkeit.csv
    │   ├── risingsun_score.csv
    │   ├── risingsun_score2.csv
    │   └── Einteilung.txt               # Participant group assignments
    │
    ├── load_save_data/                  # Data preprocessing scripts
    │   ├── load_MIDI_finger.py          # Sequence accuracy and keystroke analysis
    │   └── load_MIDI_song.py            # Timing & pitch accuracy via DTW
    │
    └── statistical_analysis/            # Evaluation scripts
        ├── anova_fingerdex.py           # ANOVA for finger dexterity
        ├── blues.py                     # Statistics on Blues performance
        ├── sun.py                       # Statistics on Rising Sun performance
        └── statistical_analysis.py      # Combined group comparisons
```

---

## 🎯 Project Goal

To evaluate and compare **classical** and **AR-trained** participants in piano performance based on:
- 🎵 **Finger dexterity** (correct sequences played)
- ⏱️ **Timing accuracy** (DTW distance from reference)
- 🎼 **Pitch accuracy** (DTW distance from reference)
- 📊 **Statistical group differences**

---

## 🧪 Setup & Workflow

See `Setup.md` for instructions on data collection using:
- MuseScore Studio 4
- PianoVision Desktop & Meta Quest 3S
- loopMIDI + MIDI-OX (for MIDI routing)
- Cakewalk (for recording MIDI)

---

## ⚙️ How It Works

### 1. **Data Extraction**
- `load_MIDI_finger.py` analyzes note sequences from filenames containing “finger”.
- `load_MIDI_song.py` compares performance MIDI files against references using **Dynamic Time Warping (DTW)**.

### 2. **Generated Outputs (in `/Data`)**
- `*_score.csv`: raw DTW distance scores
- `*_score2.csv`: cleaned subsets (e.g. only 1st appointment)
- `fingergeschicklichkeit.csv`: sequence accuracy and keystroke count
- `Einteilung.txt`: lists participant ID and group (AR or Klassisch)

### 3. **Statistical Evaluation**
Run scripts in `statistical_analysis/` to analyze:
- Group differences (AR vs. Klassisch)
- Performance over appointments
- Song-specific accuracy

---

## 📦 Requirements

Install required Python libraries:

```bash
pip install pretty_midi dtw-python pandas numpy
```

---

## 📌 Notes

- File naming format: `ParticipantID_Appointment_Song_Attempt.mid`
- Finger dexterity is evaluated using predefined expected sequences.
