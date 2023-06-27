import mido 
import os
import librosa
from midi2audio import FluidSynth
import numpy as np
import soundfile as sf

def generate_instrument(instrument, length, path):
    time = 0
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message('program_change', program=instrument, time=0))
    
    while time < length:
        note_length = np.random.randint(10, 2000)
        vel = np.random.randint(40, 120)
        note = np.random.randint(36, 98)
        track.append(mido.Message('note_on', note=note, velocity=vel, time=0))
        track.append(mido.Message('note_off', note=note, velocity=vel, time=note_length))
        
        time += note_length
    mid.save(path)


def mix_audio(audio_signals):
    mix = audio_signals[0]
    for i in range(1, len(audio_signals)):
        mix += audio_signals[i]
    return mix


def generate_song(instruments, length, song, set):
    audio_signals = []
    for ins in range(len(instruments)):
        midipath = OUTPUT_FOLDER + '/MIDI/ins' + str(ins)
        wavpath = OUTPUT_FOLDER + '/WAV/foreground/' + set + '/ins' + str(ins)
        os.makedirs(midipath, exist_ok=True)
        os.makedirs(wavpath, exist_ok=True)
        midifile = midipath + '/song' + str(song) + '.mid'
        wavfile = wavpath + '/song' + str(song).zfill(4) + '.wav'

        generate_instrument(instruments[ins], length*1000, midifile)

        fs.midi_to_audio(midifile, wavfile)
        audio = librosa.load(wavfile, sr=44100, duration=length, mono=True)
        audio = librosa.resample(audio[0], orig_sr = 44100, target_sr = sr)
        audio_signals.append(audio)
        sf.write(wavfile, audio, sr, subtype='PCM_24')
    mix = mix_audio(audio_signals)
    os.makedirs(OUTPUT_FOLDER + '/WAV/foreground/' + set + '/mix/', exist_ok=True)
    sf.write(OUTPUT_FOLDER + '/WAV/foreground/' + set + '/mix/song' + str(song).zfill(4) + '.wav', mix, sr, subtype='PCM_24')

OUTPUT_FOLDER = 'datasets/randomMIDI/PianoViolin11025/'
np.random.seed(123)
instruments = [0, 0, 0, 40]
length = 3
sr = 11025

num_train = 0
num_val = 50
num_test = 50

fs = FluidSynth()

for song in range(num_train):
    generate_song(instruments, length, song, 'train')

for song in range(num_val):
    generate_song(instruments, length, song, 'val')

for song in range(num_test):
    generate_song(instruments, length, song, 'test')