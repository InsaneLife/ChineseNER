#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Time    :   2019/01/23 00:24:59
@Author  :   Aaron Chou
@Desc    :   
"""

# here put the import lib
import tensorflow as tf
import numpy as np
from .base_model import BaseModel
from .utils import Progbar


class BiRnn(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        pass

    def add_placeholder(self):
        # shape = (batch_size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='bert_word_ids')

        # shape = (batch_size)
        # 第一维：batch中每一个sentence的长度
        self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="sequence_length")

        # shape = (batch_size, max length of sentence, max length of word)
        # 第一维：batch中每一个sentence
        # 第二维：sentence中的每一个word
        # 第三维：每一个word中char id list
        # self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")
        # shape = (batch_size, max length of sentence)
        self.ner_labels = tf.placeholder(tf.int32, shape=[None, None], name="meta_ner_labels")

        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

        # self.labels_pred = tf.placeholder(dtype=tf.int32, shape=[None, None], name="labels_pred")
        # self.pred_scores = tf.placeholder(dtype=tf.float32, shape=[None], name="pred_score")

    def add_embedding_layer(self):
        with tf.variable_scope('word_embedding_layer'):
            if self.config.embeddings is None:
                print("inital embedding Randomly")
                embedding_matrix_ = tf.get_variable(name='embedding_matrix_', dtype=tf.float32,
                                                    shape=[self.config.word_size, self.config.dim_word])
            else:
                embedding_matrix_ = tf.Variable(self.config.embeddings, dtype=tf.float32,
                                                shape=[self.config.word_size, self.config.dim_word])
            self.word_embeddings_ = tf.nn.embedding_lookup(embedding_matrix_, self.word_ids)

    def add_encoder_layer(self):
        state_outputs, final_state = self.bi_lstm(self.word_embeddings_, self.sequence_length,
                                                  self.config.rnn_layer_size, self.keep_prob)
        state_dim = 2 * self.config.rnn_layer_size
        slot_outputs = tf.reshape(state_outputs, [-1, state_dim])
        nstep = tf.shape(state_outputs)[1]
        with tf.variable_scope("slot_proj"):
            slot_outputs = self.add_fc_layer(slot_outputs, state_dim, self.config.tag_size)
            if self.config.use_crf:
                nstep = tf.shape(state_outputs)[1]
                slot_outputs = tf.reshape(slot_outputs, [-1, nstep, self.config.tag_size])
            self.slot_outputs = slot_outputs

        with tf.variable_scope('slot_loss'):
            if self.config.use_crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    slot_outputs, self.ner_labels, self.sequence_length)
                self.slot_loss = tf.reduce_mean(-log_likelihood)
            else:
                slots_shape = tf.shape(slot_outputs)
                slots_reshape = tf.reshape(slot_outputs, [-1])
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ner_labels, logits=slots_reshape)
                self.slot_loss = tf.reduce_mean(crossent)
                # crossent = tf.reshape(crossent, slots_shape)
                # slot_loss = tf.reduce_sum(crossent * slot_weights, 1)
                # total_size = tf.reduce_sum(slot_weights, 1)
                # total_size += 1e-12
                # slot_loss = slot_loss / total_size
            self.loss = self.slot_loss
        with tf.variable_scope("predict"):
            if self.config.use_crf:
                self.inference_slot_output, self.slot_scores = tf.contrib.crf.crf_decode(
                    self.slot_outputs, trans_params, self.sequence_length)
            else:
                self.inference_slot_output = tf.nn.softmax(self.slot_outputs, name='slot_output')
            pass

    def build_network(self):
        self.sess = tf.InteractiveSession()
        self.add_placeholder()
        self.add_embedding_layer()
        self.add_encoder_layer()
        self.add_train_op("adam", self.learning_rate, self.loss)
        self.initialize_session()

    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) / batch_size
        prog = Progbar(target=nbatches)
        for i, (word_ids, seq_len, tag_ids) in enumerate(train.get_batch(batch_size)):
            fd = {self.word_ids: word_ids, self.sequence_length: seq_len, self.ner_labels: tag_ids,
                  self.learning_rate: self.config.learning_rate, self.keep_prob: self.config.keep_prob}
            _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])
            # if i % 100 == 0:
            #     self.file_writer.add_summary(summary, epoch * nbatches + i)
        # test_auc, test_f1, test_acc, threshold = self.run_evaluate(dev)
        # self.logger.info(
        #     "------ test_auc:{:04.4f}, test_f1:{:04.4f}, test_acc:{:04.4f}, threshold:{:04.4f} ------".format(
        #         test_auc, test_f1, test_acc, threshold))
        # return test_auc, test_f1

    def run_evaluate(self, dev):
        """
        :param dev: dataset for evaluation
        :return: f1
        """
        tag_ids_all = []
        for i, (word_ids, seq_len, tag_ids) in enumerate(dev.get_batch(self.config.batch_size)):
            tag_ids_all.extend(tag_ids)
            fd = {self.word_ids: word_ids, self.sequence_length: seq_len,
                  self.learning_rate: self.config.learning_rate, self.keep_prob: 1}
            pred_slots, pred_score = self.sess.run([self.inference_slot_output, self.slot_scores], feed_dict=fd)
        pass
