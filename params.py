import datetime
#file_path parameters
notes_save_path = 'data/notes'
midi_path = 'data/midi'
offsets_save_path = 'data/offset'
generated_midi_path = 'data/midi/generated/generated/{}saved.mid'.\
    format(str(datetime.datetime.now().date())+str(datetime.datetime.now().time()))
#ckpt_path = 'ckpt/saved2018-10-05.ckpt'
ckpt_path = 'ckpt/saved.ckpt'

#model parameter
sequence_length = 100
batch_size = 64

#training hyper parameters
epochs = 100
learning_rate = 0.0001

#device hyper parameter
gpu_mode = True