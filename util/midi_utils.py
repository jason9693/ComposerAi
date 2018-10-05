import numpy as np
import pickle

def get_pitchnames(notes):
    return sorted(set(item for item in notes))

def prepare_sequence(notes, pitchnames, n_vocab, sequence_length):
    sequence_length = sequence_length
    #pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    normed_network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    normed_network_input = normed_network_input / float(n_vocab)

    #network_output =__onehot__(network_output,len(note_to_int))

    return (network_input,normed_network_input, network_output)

def load_notes(note_saved_path):
    with open(note_saved_path, 'rb') as filepath:
        notes = pickle.load(filepath)
    return notes