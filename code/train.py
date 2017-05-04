from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.015, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_batches", 20, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 150, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")
tf.app.flags.DEFINE_boolean("evaluate_model", False, "If True, evaluate model, if False, train model.")
tf.app.flags.DEFINE_boolean("debug", True, "If True, evaluate model, if False, train model.")

FLAGS = tf.app.flags.FLAGS

MAX_PASSAGE_LENGTH = 300


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def load_file(filee):
    res = []
    for line in open(filee):
        words = [int(a) for a in line.split(" ")]        
        res.append(words)
    return res



def load_data(directory):
    train_context = load_file(directory + '/train.ids.context')
    train_answer = load_file(directory +'/train.span')
    train_question = load_file(directory +'/train.ids.question')

    dev_context = load_file(directory + '/val.ids.context')
    dev_answer = load_file(directory + '/val.span')
    dev_question = load_file(directory + '/val.ids.question')

    train_context_copy = []
    train_answer_copy = []
    train_question_copy = []

    for ii in xrange(len(train_context)):
         if int(train_answer[ii][0]) < MAX_PASSAGE_LENGTH and int(train_answer[ii][1]) < MAX_PASSAGE_LENGTH:
            train_context_copy.append(train_context[ii])
            train_answer_copy.append(train_answer[ii])
            train_question_copy.append(train_question[ii])

    dev_context_copy = []
    dev_answer_copy = []
    dev_question_copy = []

    for ii in xrange(len(dev_context)):
         if int(dev_answer[ii][0]) < MAX_PASSAGE_LENGTH and int(dev_answer[ii][1]) < MAX_PASSAGE_LENGTH:
            dev_context_copy.append(dev_context[ii])
            dev_answer_copy.append(dev_answer[ii])
            dev_question_copy.append(dev_question[ii])

    return train_context_copy,train_answer_copy,train_question_copy,dev_context_copy,dev_answer_copy,dev_question_copy


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = load_data(FLAGS.data_dir)
    #print(dataset)

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    deocder = Decoder(output_size=FLAGS.output_size)

    global_train_dir = '/tmp/cs224n-squad-train'

    qa = QASystem(encoder, deocder, global_train_dir)

    # Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    # file paths saved in the checkpoint. This allows the model to be reloaded even
    # if the location of the checkpoint files has moved, allowing usage with CodaLab.
    # This must be done on both train.py and qa_answer.py in order to work.
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    os.symlink(os.path.abspath(FLAGS.train_dir), global_train_dir)
    train_dir = global_train_dir

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        if not FLAGS.evaluate_model:
            initialize_model(sess, qa, train_dir)
        qa.runModel(sess, dataset, FLAGS.evaluate_model)

if __name__ == "__main__":
    tf.app.run()
