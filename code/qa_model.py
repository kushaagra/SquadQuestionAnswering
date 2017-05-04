from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

from evaluate import exact_match_score, f1_score, metric_max_over_ground_truths

from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)

'''
================================
        GLOBAL VARIABLES
================================
'''
MAX_PASSAGE_LENGTH  = 300
EMBED_PATH          = 'data/squad/glove.trimmed.100.npz'
MAX_Q_LENGTH        = 25
BATCH_SIZE          = 30
NUM_EPOCHS          = 10
EVALSET             = 1              # 1 for dev set ; 0 for training set
DEBUG               = False           # Use only 20 batches
DROPOUT             = 1            # keep prob
CLIP_GRADIENTS      = True
MAX_GRADIENT        = tf.cast(5, tf.float64)
LEARNING_RATE       = 0.001
#LEARNING_RATE       = [0.01, 0.005, 0.003, 0.002, 0.001]
CONSTANT_EMBEDDINGS = True

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2


def pad_sequences(data, max_length):
    '''Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.
    '''

    ret1 = []
    ret2 = []

    # Use this zero vector when padding sequences.
    zero_vector = PAD_ID

    for passage in data:
        ### YOUR CODE HERE (~4-6 lines)
        new_passage = [i for i in passage] + [zero_vector for i in range(max_length+1 - len(passage))]
        new_passage = new_passage[0:max_length]
        masking = [True for i in range(len(passage))] + [False for i in range(max_length + 1 - len(passage))]
        masking = masking[0:max_length]
        ret1.append(new_passage)
        ret2.append(masking)
        #ret.append((new_passage, masking))
        ### END YOUR CODE ###
    return (ret1,ret2)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode_passage(self, inputt, q_fw, q_bw, masks, dropoutt=1, encoder_state_input=None):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        ### Input dimension : (batch_size,max_length,embedding_size)

        
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropoutt,input_keep_prob = dropoutt)
        p_len = tf.reduce_sum(tf.cast(masks, 'int32'), 1)

        outputs, states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell , cell_bw=cell, dtype=tf.float64,\
                                                    inputs = inputt, sequence_length = p_len, scope = 'p1',\
                                                    initial_state_fw=q_fw, initial_state_bw=q_bw)
        output_fw, output_bw = outputs
        states_fw, states_bw = states


        return (output_fw, output_bw)

    def encode_question(self, inputt, masks, dropoutt, encoder_state_input=None):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        ### Input dimension : (batch_size,max_length,embedding_size)

        cell2 = tf.nn.rnn_cell.LSTMCell(num_units=self.size, state_is_tuple=True)

        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=dropoutt,input_keep_prob = dropoutt)
        
        q_len = tf.reduce_sum(tf.cast(masks, 'int32'), 1)       # [batch_size, 1]

        outputs2, states2  = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell2, cell_bw=cell2, dtype=tf.float64,\
                                                        inputs = inputt, sequence_length = q_len, scope = 'q1')
        output_fw2, output_bw2 = outputs2                       # [batch_size, MAX_Q_LENGTH, h]
        states_fw2, states_bw2 = states2                       # [batch_size, h]

        return (output_fw2, output_bw2, states_fw2, states_bw2)

    
    def base_attention(self, Q, P, encoder_state_input=None):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        ### Input dimension : (batch_size, max_length, embedding_size)

        '''
        DIMENSIONS:
        P : [batch_size, MAX_PASSAGE_LENGTH, h]
        Q : [batch_size, MAX_Q_LENGTH, h]
        '''
        Pt = tf.transpose(P, perm=[0,2,1])          # batch_size, h, MAX_PASSAGE_LENGTH
        QPt = batch_matmul(Q, Pt)                   # batch_size, MAX_Q_LENGTH, MAX_PASSAGE_LENGTH

        A_q = tf.nn.softmax(QPt)                    # batch_size, MAX_Q_LENGTH, MAX_PASSAGE_LENGTH
        C_q = batch_matmul(A_q, P)                  # batch_size, MAX_Q_LENGTH, h
        C_q_cat = tf.concat(2, [Q, C_q])                           # batch_size, MAX_Q_LENGTH, 2*h

        A_p = tf.nn.softmax(tf.transpose(QPt, perm=[0,2,1]))       # batch_size, MAX_PASSAGE_LENGTH, MAX_Q_LENGTH
        C_p = batch_matmul(A_p, C_q_cat)                  # batch_size, MAX_PASSAGE_LENGTH, 2*h

        hidden_dim = P.get_shape().as_list()[-1]
        W = tf.get_variable('W', shape=[3*hidden_dim, 2*hidden_dim],\
                                initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        b = tf.get_variable(name='b', shape=[2*hidden_dim],\
                                initializer=tf.constant_initializer(0.0), dtype = tf.float64)


        pred_P = tf.concat(2, [tf.cast(C_p, dtype=tf.float64), tf.cast(P, dtype=tf.float64)])   # batch_size, MAX_PASSAGE_LENGTH, 3*hidden_dim
        pred_P = tf.reshape(pred_P, [-1, 3*hidden_dim])                 # batch_size*MAX_PASSAGE_LENGTH, 3*hidden_dim

        pred = tf.matmul(pred_P, W) + b                                 # batch_size, MAX_PASSAGE_LENGTH*2*hidden_dim
        pred = tf.reshape(pred, [-1, MAX_PASSAGE_LENGTH, 2*hidden_dim]) # batch_size, MAX_PASSAGE_LENGTH, 2*hidden_dim

        return pred                                                     # batch_size, MAX_PASSAGE_LENGTH, 2*hidden_dim




    def base_prediction(self, pred, p_mask, dropoutt):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        ### Input dimension : (batch_size, max_length, embedding_size)

        '''
        DIMENSIONS:
        pred : [batch_size, MAX_PASSAGE_LENGTH, 2*hidden_dim]
        '''

        hidden_dim_2 = pred.get_shape().as_list()[-1]

        cell4 = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim_2/2, state_is_tuple=True)

        sentence_len = tf.reduce_sum(tf.cast(p_mask, 'int32'), 1)       # [batch_size, 1]
 
        cell3 = tf.nn.rnn_cell.DropoutWrapper(cell4, output_keep_prob=dropoutt, input_keep_prob=dropoutt)

        outt, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell4, cell_bw=cell4,\
                        dtype=tf.float64 ,inputs = pred, sequence_length=sentence_len, scope='g0')
        outt_fw, outt_bw = outt
        g0 = tf.concat(2,[outt_fw,outt_bw])
        outt2, _  = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell4, cell_bw=cell4,\
                       dtype=tf.float64 ,inputs = g0, sequence_length=sentence_len, scope='g1')
        outt2_fw,outt2_bw = outt2
        pred_afterLSTM = tf.concat(2,[outt2_fw,outt2_bw])

  
        pred_start = tf.concat(2,[pred,pred_afterLSTM])

        
        W_start = tf.get_variable('W_start', shape=[2*hidden_dim_2, 1],\
                                initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        W_end = tf.get_variable('W_end', shape=[2*hidden_dim_2, 1],\
                                initializer = tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

        pred_reshaped = tf.reshape(pred_start, [-1, 2*hidden_dim_2])            # batch_size*MAX_PASSAGE_LENGTH, 2*hidden_dim

        pred_start = tf.matmul(pred_reshaped, W_start)                  # batch_size*MAX_PASSAGE_LENGTH, 1
        pred_start = tf.reshape(pred_start, [-1, MAX_PASSAGE_LENGTH])   # batch_size*MAX_PASSAGE_LENGTH
        

        cell3 = tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim_2/2, state_is_tuple=True)

        sentence_len = tf.reduce_sum(tf.cast(p_mask, 'int32'), 1)       # [batch_size, 1]
 
        cell3 = tf.nn.rnn_cell.DropoutWrapper(cell3, output_keep_prob=dropoutt, input_keep_prob=dropoutt)

        outputs, states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell3, cell_bw=cell3,\
                        dtype=tf.float64 ,inputs = pred_afterLSTM, sequence_length=sentence_len, scope='pred_end')
        output_fw, output_bw = outputs
        pred_end = tf.concat(2, [output_fw, output_bw])
        output_end = tf.concat(2,[pred,pred_end])
        output_end_reshaped = tf.reshape(output_end, [-1, 2*hidden_dim_2])
        pred_end = tf.matmul(output_end_reshaped, W_end)                      # batch_size*MAX_PASSAGE_LENGTH, 1
        pred_end = tf.reshape(pred_end, [-1, MAX_PASSAGE_LENGTH])       # batch_size*MAX_PASSAGE_LENGTH
        
        return (pred_start, pred_end)


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size


class QASystem(object):
    def __init__(self, encoder, decoder, train_dir, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # Save your model parameters/checkpoints here
        self.train_dir = train_dir
        self.save_path = './models/{:%Y%m%d_%H%M%S}/'.format(datetime.now())
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.pretrained_embeddings = np.load(EMBED_PATH)['glove']
        self.embed_size = len(self.pretrained_embeddings[0])
        self.max_plength = MAX_PASSAGE_LENGTH
        self.max_qlength = MAX_Q_LENGTH
        # ==== set up placeholder tokens ========

        self.passage_placeholder = tf.placeholder(tf.int32,shape = (None, self.max_plength))
        self.question_placeholder = tf.placeholder(tf.int32,shape = (None,self.max_qlength))
        self.p_mask_placeholder = tf.placeholder(tf.bool , shape = (None,self.max_plength))
        self.q_mask_placeholder = tf.placeholder(tf.bool,shape = (None, self.max_qlength))
        self.answer_start_placeholder = tf.placeholder(tf.int32,shape = (None,))
        self.answer_end_placeholder = tf.placeholder(tf.int32, shape = (None,))

        self.dropout_placeholder = tf.placeholder(tf.float64,shape = ())

        self.encoder = encoder
        self.decoder = decoder
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.embedd_passage,self.embedd_question = self.setup_embeddings()
            self.x_start, self.x_end  = self.setup_system()
            self.loss, self.train_op = self.setup_loss()

        # ==== set up training/updating procedure ====
        #pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """

        output_fw_question, output_bw_question, state_fw_q, state_bw_q = self.encoder.encode_question(self.embedd_question,\
                                self.q_mask_placeholder, self.dropout_placeholder) # batch_size, LSTmSize
        output_fw_passage, output_bw_passage =  self.encoder.encode_passage(self.embedd_passage,\
                            state_fw_q, state_bw_q, self.p_mask_placeholder, self.dropout_placeholder)  # Batch_size,time_steps,lstm_size

        # output_fw_question, output_bw_question, state_fw_q, state_bw_q = self.encoder.encode_question(self.embedd_question,self.q_mask_placeholder) # batch_size, LSTmSize
        # output_fw_passage, output_bw_passage =  self.encoder.encode_passage(self.embedd_passage,\
        #                     state_fw_q, state_bw_q, self.p_mask_placeholder)  # Batch_size,time_steps,lstm_size

        with vs.variable_scope("pred_fw"):
            pred_fw = self.encoder.base_attention(output_fw_question, output_fw_passage)   # batch_size, MAX_PASSAGE_LENGTH, 2*hidden_dim
        with vs.variable_scope("pred_bw"):
            pred_bw = self.encoder.base_attention(output_bw_question, output_bw_passage)   # batch_size, MAX_PASSAGE_LENGTH, 2*hidden_dim

        pred_fw_bw = tf.concat(2, [tf.cast(pred_fw, dtype=tf.float64), tf.cast(pred_bw, dtype=tf.float64)])                                 # batch_size, MAX_PASSAGE_LENGTH, 4*hidden_dim
        
        X_start, X_end = self.encoder.base_prediction(pred_fw_bw, self.p_mask_placeholder, self.dropout_placeholder)  # Xstart,Xend is btach_size, passage length

        return (X_start, X_end)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        loss = 0.0
        with vs.variable_scope("loss"):
            loss  = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.x_start,self.answer_start_placeholder))
            loss        += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.x_end,self.answer_end_placeholder))

        train_op  = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

        # grads, variables = zip(*optimizer.compute_gradients(loss))

        # if(CLIP_GRADIENTS):
        #     grads, _ = tf.clip_by_global_norm(grads, MAX_GRADIENT)
   
        # final_grad = zip(grads, variables)

        # train_op = optimizer.apply_gradients(final_grad)

        return loss, train_op

    def relevancy(self, embedd_passage, embedd_question):
        
        p_n = tf.nn.l2_normalize(embedd_passage, dim = 2)                   # [batch_size, MAX_PASSAGE_LENGTH, embed_size]
        q_n = tf.nn.l2_normalize(embedd_question, dim = 2)                  # [batch_size, MAX_Q_LENGTH, embed_size]

        p = tf.tile(tf.expand_dims(p_n, 2), [1, 1, MAX_Q_LENGTH, 1])        # [batch_size, MAX_PASSAGE_LENGTH, MAX_Q_LENGTH, embed_size]
        q = tf.tile(tf.expand_dims(q_n, 1), [1, MAX_PASSAGE_LENGTH, 1, 1])  # [batch_size, MAX_PASSAGE_LENGTH, MAX_Q_LENGTH, embed_size]
        
        pq = tf.multiply(p, q)                                              # [batch_size, MAX_PASSAGE_LENGTH, MAX_Q_LENGTH, embed_size]
        weights = tf.reduce_max(pq, reduction_indices=[2])                  # [batch_size, MAX_PASSAGE_LENGTH, embed_size]

        relevancy_passage = tf.multiply(embedd_passage, weights)            # [batch_size, MAX_PASSAGE_LENGTH, embed_size]

        return relevancy_passage


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            
            if CONSTANT_EMBEDDINGS:
                emb_tensor = tf.constant(self.pretrained_embeddings)
            else:
                emb_tensor = tf.Variable(self.pretrained_embeddings)

            embedd_passage_wo_rel = tf.nn.embedding_lookup(emb_tensor,self.passage_placeholder)    # [batch_size,p_len,emb_size]

            embedd_question = tf.nn.embedding_lookup(emb_tensor,self.question_placeholder)         # [batch_size, q_len,emb_size]

            # embedd_passage = self.relevancy(embedd_passage_wo_rel, embedd_question)

        return (embedd_passage_wo_rel,embedd_question)
            #pass

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}
        input_feed[self.passage_placeholder] = train_x[0]
        input_feed[self.question_placeholder] = train_x[1]
        input_feed[self.p_mask_placeholder] = train_x[2]
        input_feed[self.q_mask_placeholder] = train_x[3]
        input_feed[self.answer_start_placeholder] = train_y[0]
        input_feed[self.answer_end_placeholder] = train_y[1]
        input_feed[self.dropout_placeholder] = DROPOUT

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.loss, self.train_op]

        outputs = session.run(output_feed, input_feed)

        #print(outputs)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x, ans_y):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        input_feed[self.passage_placeholder]        = test_x[0]
        input_feed[self.question_placeholder]       = test_x[1]
        input_feed[self.p_mask_placeholder]         = test_x[2]
        input_feed[self.q_mask_placeholder]         = test_x[3]
        input_feed[self.answer_start_placeholder]   = ans_y[0]
        input_feed[self.answer_end_placeholder]     = ans_y[1]
        input_feed[self.dropout_placeholder] = 1
        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.x_start, self.x_end, self.loss]
        outputs = session.run(output_feed, input_feed)

        return outputs




    def sub_decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        input_feed[self.passage_placeholder]        = test_x[0]
        input_feed[self.question_placeholder]       = test_x[1]
        input_feed[self.p_mask_placeholder]         = test_x[2]
        input_feed[self.q_mask_placeholder]         = test_x[3]
        input_feed[self.dropout_placeholder] = 1
        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.x_start, self.x_end]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def sub_answer(self, session, test_x):

        yp, yp2 = self.sub_decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)
        for i in range(BATCH_SIZE):
            if(a_e[i] < a_s[i]):
                temp = a_s[i]
                a_s[i] = a_e[i]
                a_e[i] = temp


        return(a_s, a_e)      


    def submission_evaluate_model(self,session,dataset):
        saver = tf.train.Saver()
        saver.restore(session,"train/20170320_171142/model_save")
        sub_c, sub_q, question_uuid = dataset
        
        submission_context = [i for i in sub_c]
        submission_question = [i for i in sub_q]

        #### SOME BULLSHIT
        submission_context.extend(submission_context[-1*BATCH_SIZE:])
        submission_question.extend(submission_question[-1*BATCH_SIZE:])
        question_uuid.extend(question_uuid[-1*BATCH_SIZE:])
        #### BULL SHIT OVER


        padded_context, p_mask     = pad_sequences(submission_context,MAX_PASSAGE_LENGTH)
        padded_question, q_mask    = pad_sequences(submission_question,MAX_Q_LENGTH)
        num_batches = int(len(padded_context)/BATCH_SIZE)
        answer_start_list = []
        answer_end_list = []
        print ("NUM Batches = ")
        print (num_batches)
        for num_batch_iter in xrange(num_batches):

            # a_s, a_e= self.sub_answer(session, (padded_context[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
            #                                             padded_question[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
            #                                             p_mask[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
            #                                             q_mask[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)]))

            a_s, a_e = self.sub_answer(session, (padded_context[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                                        padded_question[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                                        p_mask[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                                        q_mask[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)]))

            answer_start_list.extend(a_s)
            answer_end_list.extend(a_e)
            print ("iter = ")
            print (num_batch_iter)
        print (len(answer_start_list))
        print (len(answer_end_list))

        return (answer_start_list,answer_end_list,question_uuid)        

    def answer(self, session, test_x, ans_y):

        yp, yp2, loss = self.decode(session, test_x, ans_y)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e, loss)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost


    def evaluate_model(self, session, dataset, evalSet=0, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param eval: 0 for evaluating on training set,
                    1 for evaluating on dev set,
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        print (10*'-')
        print ('EVALUATE MODEL')
        print (10*'-')
        print (2*'\n')

        train_context, train_answer, train_question,dev_context, dev_answer, dev_question = dataset
        
        if evalSet == 0:
            padded_context, p_mask     = pad_sequences(train_context,MAX_PASSAGE_LENGTH)
            padded_question, q_mask    = pad_sequences(train_question,MAX_Q_LENGTH)
            answer_start              = [i[0] for i in train_answer]
            answer_end                = [i[1] for i in train_answer]
            num_batches = int(len(padded_context)/BATCH_SIZE)

        else:
            padded_context,p_mask     = pad_sequences(dev_context,MAX_PASSAGE_LENGTH)
            padded_question,q_mask    = pad_sequences(dev_question,MAX_Q_LENGTH)
            answer_start                = [i[0] for i in dev_answer]
            answer_end                  = [i[1] for i in dev_answer]
            num_batches = int(len(padded_context)/BATCH_SIZE)

        if DEBUG:
            num_batches = 20

        print ("Number of batches = ")
        print (num_batches)

        f1 = em = total = 0.
        for num_batch_iter in xrange(num_batches):

            a_s, a_e, loss = self.answer(session, (padded_context[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                                        padded_question[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                                        p_mask[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                                        q_mask[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)]),\
                                                        (answer_start[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                                        answer_end[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)]))

            print ("EVALUATING MODEL - Loss for batch number: ")
            print (loss)
            print (10*'-')
            print (2*'\n')

            for BATCH_SIZE_iter in xrange(BATCH_SIZE):
                num_same = 0                                        # record no. of words that are correct
                total += 1
                idx = BATCH_SIZE*num_batch_iter + BATCH_SIZE_iter
                # EXACT MATCH - start and end match
                if(a_s[BATCH_SIZE_iter] == answer_start[idx] and a_e[BATCH_SIZE_iter] == answer_end[idx]):
                    num_same = a_e[BATCH_SIZE_iter] - a_s[BATCH_SIZE_iter] + 1
                    em += 1
                    f1 += 1
                # ZERO MATCH
                elif(a_s[BATCH_SIZE_iter] > a_e[BATCH_SIZE_iter] or a_e[BATCH_SIZE_iter] <= answer_start[idx]\
                                                                or a_s[BATCH_SIZE_iter] >= answer_end[idx]):
                    num_same = 0
                # CASE
                else:
                    num_same = min(a_e[BATCH_SIZE_iter], answer_end[idx]) - max(a_s[BATCH_SIZE_iter], answer_start[idx]) + 1
                    num_predicted = a_e[BATCH_SIZE_iter] - a_s[BATCH_SIZE_iter] + 1
                    num_ground_truth = answer_end[idx] - answer_start[idx] + 1
                    if num_same > 0:
                        precision = 1.0 * num_same / num_predicted
                        recall = 1.0 * num_same / num_ground_truth
                        f1 += (2 * precision * recall) / (precision + recall)
    	    print ("Total so far = ")
    	    print (total)
    	    print ("F1 current = ")
    	    print (f1*100/total)
    	    print (em*100/total)
    	    print (f1)
    	    print (em)


        f1 = 100.0 * f1 / total
        em = 100.0 * em / total

        print (10*'-')
        print ('F1:')
        print (f1)
        print ('EM:')
        print (em)
        print (10*'-')
        print (2*'\n')

        return(f1, em)

        

    def train_model(self, session, dataset, saver):
        """
        Train and save the model parameters

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param saver:
        :return:
        """

        print (10*'-')
        print ('TRAIN MODEL')
        print (10*'-')
        print (2*'\n')

        train_context, train_answer, train_question, _, _, _ = dataset

        padded_train_context, p_mask     = pad_sequences(train_context,MAX_PASSAGE_LENGTH)
        padded_train_question, q_mask    = pad_sequences(train_question,MAX_Q_LENGTH)
        train_answer_start              = [i[0] for i in train_answer]
        train_answer_end                = [i[1] for i in train_answer]
        
        num_batches = int(len(padded_train_context)/BATCH_SIZE)

        if DEBUG:
            num_batches = 20

        print ("Number of batches = ")
        print (num_batches)

        for i in range(NUM_EPOCHS):

            start_time = time.time()
            
            print (10*'-')
            print ('EPOCH#:')
            print (i)
            print (10*'-')
            print (2*'\n')
            start_time = time.time()
            for num_batch_iter in xrange(num_batches):
                op = self.optimize(session, (padded_train_context[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                padded_train_question[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                p_mask[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                q_mask[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)]),\
                                (train_answer_start[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)],\
                                train_answer_end[BATCH_SIZE*num_batch_iter:BATCH_SIZE*(num_batch_iter+1)]))
                if(num_batch_iter%10 == 0):
                    print ('Batch number: ', num_batch_iter)
                    print (op)

            print('Time elapsed: ', time.time()-start_time)

            saver.save(session, self.save_path + 'model_save')

            self.evaluate_model(session, dataset, evalSet=EVALSET, sample=100, log=False)




    def runModel(self, session, dataset, evalModel=False):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in self.train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        
        saver = tf.train.Saver()

        if not evalModel:
            # saver.restore(session,"models/20170319_234852/model_save")
            self.train_model(session, dataset, saver)

        else:
	    # saver.restore(session,"models/20170314_174244/model_save")
            saver.restore(session, self.save_path + 'model_save')
            f1, em = self.evaluate_model(session, dataset, evalSet=EVALSET, sample=100, log=False)


        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
