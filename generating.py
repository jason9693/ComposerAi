from util import tf_utils, postprocessing, midi_utils
from Model.Network import ComposeNetwork
import params as par
import tensorflow as tf

if __name__ == '__main__':
    notes = midi_utils.load_notes(par.notes_save_path)
    pitchnames = midi_utils.get_pitchnames(notes)
    n_vocab = len(set(notes))

    network_input, normed_network_input, _ = \
        midi_utils.prepare_sequence(notes, pitchnames, n_vocab, par.sequence_length)

    sess = tf.Session()
    model = ComposeNetwork(list(normed_network_input.shape[1:]),n_vocab,sess=sess,ckpt_path=par.ckpt_path)
    predict = postprocessing.generate_notes(model,network_input,pitchnames,n_vocab)
    postprocessing.create_midi(predict, par.generated_midi_path)




