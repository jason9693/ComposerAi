from music21 import converter, instrument, note, chord
import tensorflow as tf
import pickle
import glob
import numpy as np
import util.midi_utils as midi_utils

def get_notes_and_offsets(midi_path, note_save_path,offset_save_path = 'data/offset'):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    offset_list = []
    try:
        notes = midi_utils.load_notes(note_save_path)
        offsets = midi_utils.load_offsets(offset_save_path)
        print(notes)
        return notes, offsets
    except:
        pass
    for file in glob.glob(midi_path+"/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        prev_offset = 0
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            #print(s2.parts)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            #print(float(element.offset))
            offset_list.append(element.offset - prev_offset)
            print(element.offset - prev_offset)
            prev_offset = element.offset
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open(offset_save_path, 'wb') as filepath:
         pickle.dump(offset_list, filepath)
    with open(note_save_path, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return (notes, offset_list)

# def prepare_sequence(notes, n_vocab):
#     sequence_length = par.sequence_length
#     pitchnames = sorted(set(item for item in notes))
#
#     note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
#
#     network_input = []
#     network_output = []
#
#     for i in range(0, len(notes) - sequence_length, 1):
#         sequence_in = notes[i:i + sequence_length]
#         sequence_out = notes[i + sequence_length]
#         network_input.append([note_to_int[char] for char in sequence_in])
#         network_output.append(note_to_int[sequence_out])
#
#     n_patterns = len(network_input)
#
#     network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
#     network_input = network_input / float(n_vocab)
#
#     #network_output =__onehot__(network_output,len(note_to_int))
#
#     return (network_input, network_output)


def __onehot__(input: list, dim: int = 2):
    out = np.zeros([len(input), dim])
    out[np.arange(dim),input] = 1.0
    return out

if __name__ == '__main__':
    get_notes('../data/midi','../data/notes','../data/offset')