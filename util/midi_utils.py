import numpy as np
import pickle

def get_pitchnames(notes):
    return sorted(set(item for item in notes))

def prepare_sequence(notes, offset, pitchnames, n_vocab, sequence_length):
    sequence_length = sequence_length
    #pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        #regarding of offset
        sequence_offset_in = offset[i:i + sequence_length]
        sequence_offset_out = offset[i + sequence_length]

        #regarding of notes or chords
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        network_input.append([[note_to_int[char], offset] for char, offset in zip(sequence_in,sequence_offset_in)])
        network_output.append([note_to_int[sequence_out], sequence_offset_out])

    n_patterns = len(network_input)

    network_input = np.array(network_input)
    #normed_network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    #print(np.array(network_input)[:,:,0])
    normed_network_input = network_input[:,:,0] / float(n_vocab)
    normed_network_input = np.transpose([normed_network_input, network_input[:,:,1]], (1,2,0))

    return (network_input,normed_network_input, network_output)

def load_notes(note_saved_path):
    with open(note_saved_path, 'rb') as filepath:
        notes = pickle.load(filepath)
    return notes

def load_offsets(offset_saved_path):
    with open(offset_saved_path, 'rb') as filepath:
        offsets = pickle.load(filepath)
    return offsets