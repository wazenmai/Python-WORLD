from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal

from world import main as world_main
from cut_vocoder import cut_speech

song_dict = {}

NOTE_DICT = {
    "A0": 27.5,
    "B0": 30.868,
    "C1": 32.703,
    "D1": 36.708,
    "E1": 41.203,
    "F1": 43.654,
    "G1": 48.999,
    "A1": 55.0,
    "B1": 61.735,
    "C2": 65.406,
    "D2": 73.416,
    "E2": 82.407,
    "F2": 87.307,
    "G2": 97.999,
    "A2": 110,
    "B2": 123.47,
    "C3": 130.81,
    "D3": 146.83,
    "E3": 164.81,
    "F3": 174.61,
    "G3": 196.0,
    "A3": 220.0,
    "B3": 246.94,
    "C4": 261.63,
    "D4": 293.67,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.0,
    "A4": 440.0,
    "B4": 493.88,
    "C5": 523.25,
    "D5": 587.33,
    "E5": 659.26,
    "F5": 698.46,
    "G5": 783.99,
    "A5": 880.0,
    "B5": 987.77,
    "C6": 1046.5,
    "D6": 1174.7,
    "E6": 1318.5,
    "F6": 1396.9,
    "G6": 1568.0,
    "A6": 1760.0,
    "B6": 1975.5,
    "C7": 2093.0,
    "D7": 2349.3,
    "E7": 2637.0,
    "F7": 2703.0,
    "G7": 3136.0,
    "A7": 3520.0,
    "B7": 3951.1,
    "C8": 4186.0
}

# accidental
ACC_DICT = {
    "A0+": 29.315,
    "B0-": 29.315,
    "C1+": 34.648,
    "D1-": 34.648,
    "D1+": 38.891,
    "E1-": 38.891,
    "F1+": 46.249,
    "G1-": 46.249,
    "G1+": 51.913,
    "A1-": 51.913,
    "A1+": 58.270,
    "B1-": 58.270,
    "C2+": 69.296,
    "D2-": 69.296,
    "D2+": 77.782,
    "E2-": 77.782,
    "F2+": 92.499,
    "G2-": 92.499,
    "G2+": 103.83,
    "A2-": 103.83,
    "A2+": 116.54,
    "B2-": 116.54,
    "C3+": 138.59,
    "D3-": 138.59,
    "D3+": 155.56,
    "E3-": 155.56,
    "F3+": 185.0,
    "G3-": 185.0,
    "G3+": 207.65,
    "A3-": 207.65,
    "A3+": 233.08,
    "B3-": 233.08,
    "C4+": 277.18,
    "D4-": 277.18,
    "D4+": 311.13,
    "E4-": 311.13,
    "F4+": 369.99,
    "G4-": 369.99,
    "G4+": 415.3,
    "A4-": 415.3,
    "A4+": 466.16,
    "B4-": 466.16,
    "C5+": 544.37,
    "D5-": 544.37,
    "D5+": 622.25,
    "E5-": 622.25,
    "F5+": 739.99,
    "G5-": 739.99,
    "G5+": 830.61,
    "A5-": 830.61,
    "A5+": 923.33,
    "B5-": 923.33,
    "C6+": 1108.7,
    "D6-": 1108.7,
    "D6+": 1244.5,
    "E6-": 1244.5,
    "F6+": 1480.0,
    "G6-": 1480.0,
    "G6+": 1661.2,
    "A6-": 1661.2,
    "A6+": 1864.7,
    "B6-": 1864.7,
    "C7+": 2217.5,
    "D7-": 2217.5,
    "D7+": 2489.0,
    "E7-": 2489.0,
    "F7+": 2960.0,
    "G7-": 2960.0,
    "G7+": 3322.4,
    "A7-": 3322.4,
    "A7+": 3729.3,
    "B7-": 3729.3
}

def pitch2freq(pitch):
    if pitch[-1] == '+' or pitch[-1] == '-':
        return ACC_DICT[pitch]
    return NOTE_DICT[pitch]

def calculate_f0(f0_o, note):
    std = np.zeros(f0_o.shape)
    sum_f0 = 0
    num_f0 = 0
    # calculate average f0
    for f in f0_o:
        if f != 0:
            num_f0 += 1
            sum_f0 += f
    avg_f0 = sum_f0 / num_f0
    print("Average f0:", avg_f0)

    # calcualte the difference
    for idx, f in enumerate(f0_o):
        if f == 0:
            std[idx] = 0
        else:
            std[idx] = f - avg_f0
    
    pitch = pitch2freq(note)
    new_f0 = np.zeros(f0_o.shape)
    for idx, s in enumerate(std):
        if s == 0:
            new_f0[idx] = 0
        else:
            new_f0[idx] = pitch + s

    return new_f0

def manipulate_f0(dat, note_list):
    num_f0 = 0
    start_idx = 0
    note_num = 1
    section_num = 0
    for idx, f in enumerate(dat['f0']):
        print(idx, f)
        if f == 0 and num_f0 != 0:
            print(f"section : {start_idx} ~ {idx}")
            new_f0 = calculate_f0(dat['f0'][start_idx:idx], note_list[note_num])
            if note_num == 0:
                all_f0 = new_f0
            else:
                all_f0 = np.concatenate((all_f0, new_f0))
            section_num += 1
            start_idx = idx
            note_num += 1
            num_f0 = 0
        elif f != 0:
            num_f0 += 1
    if start_idx != len(dat['f0']) - 1:
        print(f"left zero: {start_idx}, total len: {len(dat['f0'])}")
        pad = np.zeros(len(dat['f0']) - start_idx)
        print("zero padding shape: ", pad.shape)
        all_f0 = np.concatenate((all_f0, pad))
    print("section number: ", section_num)
    return all_f0

def calculate_section(dat):
    num_f0 = 0
    start_idx = 0
    note_num = 0
    section_num = 0
    for idx, f in enumerate(dat['f0']):
        if f == 0 and num_f0 != 0:
            print(f"section : {start_idx} ~ {idx}")
            section_num += 1
            start_idx = idx
            note_num += 1
            num_f0 = 0
        elif f != 0:
            num_f0 += 1
    if start_idx != len(dat['f0']) - 1:
        print(f"left zero: {start_idx}, total len: {len(dat['f0'])}")
    print("section number: ", section_num)

def make_song_dict(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            f_name = f.split('.')[0]
            song_dict[f_name] = root + f

def main():

    # wav_path = Path('./test/test-walk.wav')
    wav_path = './test/test-star'
    wav_sections, fs = cut_speech(wav_path + '.wav')

    # input the note requirement
    notes_number = int(input("How many note are you going to compose?"))
    note_list = list(str(num) for num in input("Enter the note items separated by space: ").strip().split())[:notes_number]
    note_len = len(note_list)
    print(f"note_list: {note_list} {note_len}")
    print()

    # TODO: input the duration requirement
    duration_list = list(float(num) for num in input("Enter the duration ratio separated by space: ").strip().split())[:notes_number]
    duration_len = len(duration_list)
    print(f"duration list: {duration_list} {duration_len}")

    # initialize the world vocoder
    vocoder = world_main.World()

    for wav_num, wav_section in enumerate(wav_sections):
        x = wav_section / (2 ** 15 - 1)
        # analysis
        dat = vocoder.encode(fs, x, f0_method='harvest', is_requiem=True) # use requiem analysis and synthesis
        print(f"section {wav_num + 1} processing...")
        dat['f0'] = calculate_f0(dat['f0'], note_list[wav_num])
        if duration_list[wav_num] != 1:
            dat = vocoder.scale_duration(dat, duration_list[wav_num])
        dat = vocoder.decode(dat)
        if wav_num == 0:
            final_data = dat['out']
        else:
            final_data = np.concatenate((final_data, dat['out']))

    wavwrite(wav_path + '-resynth_1.wav', fs, (final_data * 2 ** 15).astype(np.int16))

    # for k in song_dict:

    """
    fs, x_int16 = wavread(wav_path)
    x = x_int16 / (2 ** 15 - 1)

    if 0:  # resample
        fs_new = 16000
        x = signal.resample_poly(x, fs_new, fs)
        fs = fs_new

    if 0:  # low-cut
        B = signal.firwin(127, [0.01], pass_zero=False)
        A = np.array([1.0])
        if 0:
            import matplotlib.pyplot as plt
            w, H = signal.freqz(B, A)

            fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 6))
            ax1.plot(w / np.pi, abs(H))
            ax1.set_ylabel('magnitude')
            ax2.plot(w / np.pi, np.unwrap(np.angle(H)))
            ax2.set_ylabel('unwrapped phase')
            plt.show()
        x = signal.lfilter(B, A, x)

    vocoder = world_main.World()

    # analysis
    dat = vocoder.encode(fs, x, f0_method='harvest', is_requiem=True) # use requiem analysis and synthesis
    if 0:  # global pitch scaling
        dat = vocoder.scale_pitch(dat, 1.5)
    if 0:  # global duration scaling
        dat = vocoder.scale_duration(dat, 2)
    if 0:  # fine-grained duration modification
        vocoder.modify_duration(dat, [1, 1.5], [0, 1, 3, -1])  # TODO: look into this


#--------------- Adjust pitch -------------------------------------
    # print(dat['f0'])
    # print("original_size: ", dat['f0'].shape)

    # make_song_dict('../Python-Wrapper-for-World-Vocoder/demo/cut_speech/')

    # dat['f0'] = manipulate_f0(dat, note_list)
    # print("manipulated size: ", dat['f0'].shape)

#----------------------------------------------------------------

    # dat['f0'] = np.r_[np.zeros(5), dat['f0'][:-5]]

    # synthesis
    dat = vocoder.decode(dat)
    if 0:  # audio
        import simpleaudio as sa
        snd = sa.play_buffer((dat['out'] * 2 ** 15).astype(np.int16), 1, 2, fs)
        snd.wait_done()
    if 0:  # visualize
        vocoder.draw(x, dat)

    wavwrite(wav_path.with_name(wav_path.stem + '-resynth.wav'), fs, (dat['out'] * 2 ** 15).astype(np.int16))
    """
if __name__ == '__main__':
    main()

# C3 C4 B3 A3 G3 F3 E3 D3
