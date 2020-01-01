#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2019/01/22 20:31:04
@Author  :   Aaron Chou
@Desc    :   None
'''

# here put the import lib


import os
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell
from tensorflow.python.framework import ops
import numpy as np
from .utils import pad_sequences, get_chunks, get_tag_labels
from .bert import modeling_v1 as modeling, tokenization, optimization

 
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class BaseModel(object):

    def __init__(self, config):
        self.config = config
        self.sess = None
        self.saver = None
        self.flip_gradient = FlipGradientBuilder()
        # layer name
        self.fc_i = 0
        self.self_atte_i = 0
        self.slot_atte_i = 0
        self.slot_gete_i = 0
        self.id_cnn_i = 0
        self.bi_list_i = 0
        self.bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)

    def add_train_op(self, learning_method, learning_rate, loss, clip=-1):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        _lr_m = learning_method.lower()
        with tf.variable_scope("train_step"):
            if _lr_m == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif _lr_m == "adagrad":
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif _lr_m == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif _lr_m == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))
            with tf.control_dependencies(update_ops):
                if clip > 0:
                    grads, variables = zip(*optimizer.compute_gradients(loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, clip)
                    self.train_op = optimizer.apply_gradients(zip(grads, variables),
                                                              global_step=tf.train.get_global_step())
                else:
                    self.train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return self.train_op

    def add_bert_train_op(self, loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False):
        self.train_op = optimization.create_optimizer(
            loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        return self.train_op

    @staticmethod
    def get_params_count():
        params_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("params_count", params_count)
        return params_count

    @staticmethod
    def add_train_op2(learning_method, learning_rate, loss, task_name, clip=-1):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        _lr_m = learning_method.lower()
        with tf.variable_scope("optimizer_%s" % task_name):
            if _lr_m == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif _lr_m == "adagrad":
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif _lr_m == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif _lr_m == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))
            with tf.control_dependencies(update_ops):
                if clip > 0:
                    grads, variables = zip(*optimizer.compute_gradients(loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, clip)
                    train_op = optimizer.apply_gradients(zip(grads, variables))
                else:
                    train_op = optimizer.minimize(loss)

            return train_op

    def initialize_session(self):
        print("Initializing tf session")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        global_config = tf.ConfigProto(gpu_options=gpu_options)
        # global_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=global_config)
        # self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())
        self.saver = tf.train.Saver()

    def restore_pretrain_session(self, dir_model):
        print("Reloading the latest trained model...")
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)

        variables = tf.contrib.framework.get_variables_to_restore()
        # train_step/service_ner_loss/
        variables_to_resotre = [v for v in variables if
                                'service_ner_loss/' not in v.name and
                                'service_slot_proj/' not in v.name and
                                'cate_attention/' not in v.name and
                                'category_proj/' not in v.name
                                ]
        cur_saver = tf.train.Saver(variables_to_resotre)
        cur_saver.restore(self.sess, dir_model)

    def restore_session(self, dir_model):
        print("Reloading the latest trained model...")
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def close_session(self):
        self.sess.close()

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.logdir, self.sess.graph)

    def train(self, train, dev):
        best_score = 0
        nepoch_no_imprv = 0
        # best_f1 = 0
        # nepoch_no_f1_imprv = 0
        # self.add_summary()
        for epoch in range(self.config.nepochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))
            score = self.run_epoch(train, dev, epoch)
            self.config.learning_rate *= self.config.lr_decay
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                best_score = score
                print("- new best score!")
                # self.saved_model()
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    print("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break

    def evaluate(self, test):
        print("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.4f}".format(k, v) for k, v in metrics.items()])
        print(msg)

    def bi_lstm(self, input_embed, seq_len,
                layer_size, keep_prob, reuse=False,
                scope_name="bi_lstm"):
        self.bi_list_i += 1
        cur_name = "bi_lstm-{}".format(self.bi_list_i)
        cur_name = scope_name if reuse == tf.AUTO_REUSE and scope_name else cur_name
        with tf.variable_scope(cur_name, reuse=reuse):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob,
                                                    output_keep_prob=keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob,
                                                    output_keep_prob=keep_prob)
            state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, input_embed,
                sequence_length=seq_len, dtype=tf.float32)

            final_state = tf.concat([final_state[0][0], final_state[0][1], final_state[1][0], final_state[1][1]], 1)
            state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)

            return state_outputs, final_state

    def id_cnn(self, input_embed, filter_width,
               filter_num, repeat_times, layers_conf,
               reuse=False, scope_name=None, embed_size=None):
        self.id_cnn_i += 1
        cur_name = "id_cnn-{}".format(self.id_cnn_i)
        cur_name = scope_name if reuse == tf.AUTO_REUSE and scope_name else cur_name
        # print cur_name, reuse, scope_name, "idcnn"
        with tf.variable_scope(cur_name, reuse=reuse):
            if embed_size == None:
                embedding_shape = input_embed.get_shape()
                embedding_dim = embedding_shape[-1]
            else:
                embedding_dim = embed_size
            """
            shape of word_embeddings: [batch_size, num_steps, emb_size] 
            """
            # batch_size, 1,  num_steps, emb_size
            model_inputs = tf.expand_dims(input_embed, 1)
            initial_layer_filter_shape = [1, filter_width, embedding_dim, filter_num]
            initial_layer_w = tf.get_variable(
                "initial_layer_w", shape=initial_layer_filter_shape,
                initializer=tf.contrib.layers.xavier_initializer())
            initial_layer_b = tf.get_variable(
                "initial_layer_b",
                initializer=tf.constant(0.01, shape=[filter_num]))
            # batch_size, 1, num_steps, filter_num
            initial_layer_output = tf.nn.conv2d(
                model_inputs, initial_layer_w, strides=[1, 1, 1, 1],
                padding="SAME", name="init_layer")
            initial_layer_output = tf.nn.relu(tf.nn.bias_add(initial_layer_output, initial_layer_b), name="relu")
            atrous_input = initial_layer_output
            atrous_layers_output = []
            atrous_layers_output_dim = 0
            for block in range(repeat_times):
                for i in range(len(layers_conf)):
                    layer_name = "conv_%d" % i
                    dilation = layers_conf[i]['dilation']
                    is_last = True if i == (len(layers_conf) - 1) else False
                    with tf.variable_scope("atrous-conv-%d" % i, reuse=tf.AUTO_REUSE):
                        filter_shape = [1, filter_width, filter_num, filter_num]
                        conv_w = tf.get_variable(layer_name + "_w", shape=filter_shape,
                                                 initializer=tf.contrib.layers.xavier_initializer())
                        conv_b = tf.get_variable(layer_name + "_b", shape=[filter_num])

                        conv_output = tf.nn.convolution(atrous_input, conv_w, dilation_rate=[1, dilation],
                                                        padding="SAME", name=layer_name)

                        conv_output = tf.nn.bias_add(conv_output, conv_b)
                        conv_output = tf.nn.relu(conv_output)
                        if is_last:
                            atrous_layers_output.append(conv_output)
                            atrous_layers_output_dim += filter_num
                        atrous_input = conv_output

            cur_output = tf.concat(axis=3, values=atrous_layers_output)
            # bs, nstep, d=400
            cur_output = tf.squeeze(cur_output, [1])
            return cur_output

    def id_cnn_v2(self, input_emgbd, filter_width,
                  filter_num, repeat_times, layers_conf,
                  reuse=False, scope_name="id-cnn"):
        self.id_cnn_i += 1
        cur_name = "id_cnn-{}".format(self.id_cnn_i)
        cur_name = scope_name if reuse == tf.AUTO_REUSE and scope_name else cur_name
        # print cur_name, reuse, scope_name, "idcnn"
        with tf.variable_scope(cur_name, reuse=reuse):
            embedding_shape = input_emgbd.get_shape()
            embedding_dim = embedding_shape[-1]
            # shape of word_embeddings: [batch_size, num_steps, emb_size]
            # batch_size, 1,  num_steps, emb_size
            model_inputs = tf.expand_dims(input_emgbd, 1)
            initial_layer_filter_shape = [1, filter_width, embedding_dim, filter_num]
            initial_layer_w = tf.get_variable(
                "initial_layer_w", shape=initial_layer_filter_shape,
                initializer=tf.contrib.layers.xavier_initializer())
            initial_layer_b = tf.get_variable(
                "initial_layer_b",
                initializer=tf.constant(0.01, shape=[filter_num]))
            # batch_size, 1, num_steps, filter_num
            initial_layer_output = tf.nn.conv2d(
                model_inputs, initial_layer_w, strides=[1, 1, 1, 1],
                padding="SAME", name="init_layer")
            initial_layer_output = tf.nn.relu(tf.nn.bias_add(initial_layer_output, initial_layer_b), name="relu")
            atrous_input = initial_layer_output
            atrous_layers_output = []
            for block in range(repeat_times):
                for i in range(len(layers_conf)):
                    layer_name = "conv_%d" % i
                    dilation = layers_conf[i]['dilation']
                    is_last = True if i == (len(layers_conf) - 1) else False
                    with tf.variable_scope("atrous-conv-%d" % i, reuse=tf.AUTO_REUSE):
                        filter_shape = [1, filter_width, filter_num, filter_num]
                        conv_w = tf.get_variable(layer_name + "_w", shape=filter_shape,
                                                 initializer=tf.contrib.layers.xavier_initializer())
                        conv_b = tf.get_variable(layer_name + "_b", shape=[filter_num])

                        conv_output = tf.nn.atrous_conv2d(atrous_input, conv_w, rate=dilation,
                                                          padding="SAME", name=layer_name)

                        conv_output = tf.nn.bias_add(conv_output, conv_b)
                        conv_output = tf.nn.relu(conv_output)
                        if is_last:
                            atrous_layers_output.append(conv_output)
                        atrous_input = conv_output

            cur_output = tf.concat(axis=3, values=atrous_layers_output)
            # bs, nstep, d=400
            cur_output = tf.squeeze(cur_output, [1])
            return cur_output

    def slot_gate(self, intent_output, slot_inputs, slot_d,
                  reuse=False, scope_name="slot_gate"):
        """
        """
        self.slot_gete_i += 1
        cur_name = "slot_gete-{}".format(self.slot_gete_i)
        cur_name = scope_name if reuse == tf.AUTO_REUSE and scope_name else cur_name
        with tf.variable_scope(cur_name, reuse=reuse):
            attn_size = slot_inputs.get_shape()[2].value

            intent_gate = core_rnn_cell._linear(intent_output, attn_size, True)
            # [batch_size, hidden_size * 2]

            intent_gate = tf.reshape(intent_gate, [-1, 1, intent_gate.get_shape()[1].value])
            # [batch_size, 1, hidden_size * 2]

            gate_v = tf.get_variable("gateV", [attn_size])
            # hidden_size * 2

            slot_gate = gate_v * tf.tanh(slot_d + intent_gate)
            # state_outputs(16x33x128) + intent_gate(16x1x128)
            # [batch_size, nsetp, hidden_size * 2]

            slot_gate = tf.reduce_sum(slot_gate, [2])
            # [batch_size, nsetp]
            slot_gate = tf.expand_dims(slot_gate, -1)
            # [batch_size, nsetp, 1]

            slot_gate = slot_d * slot_gate
            # 16, 33, 128 = 16x33x128 * 16, 33, 1
            # [batch_size, nsetp, hidden_size * 2]

            slot_gate = tf.reshape(slot_gate, [-1, attn_size])
            slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])
            # [batch_size * nsetp, hidden_size * 2]
            slot_output = tf.concat([slot_gate, slot_inputs], -1)
            # [batch_size * nsetp, hidden_size * 4]
            return slot_output

    def slot_attention(self, H, reuse=False, scope_name="slot_attention"):
        """
        :param H: shape:[batch_size, nstep, hidden_size]
        :return: [batch_size, nstep, hidden_size]
        """
        self.slot_atte_i += 1
        cur_name = "slot_attention-{}".format(self.slot_atte_i)
        cur_name = scope_name if reuse == tf.AUTO_REUSE and scope_name else cur_name
        # print cur_name, reuse, scope_name, "slot_attention"
        with tf.variable_scope(cur_name, reuse=reuse):
            state_shape = H.get_shape()
            hidden_size = state_shape[2].value
            attn_size = hidden_size

            origin_shape = tf.shape(H)
            hidden = tf.expand_dims(H, 1)
            hidden_conv = tf.expand_dims(H, 2)
            # hidden shape = [batch, sentence length, 1, hidden size]
            k = tf.get_variable("AttnW", [1, 1, hidden_size, attn_size])
            hidden_features = tf.nn.conv2d(hidden_conv, k, [1, 1, 1, 1], "SAME")
            hidden_features = tf.reshape(hidden_features, origin_shape)
            hidden_features = tf.expand_dims(hidden_features, 1)
            v = tf.get_variable("AttnV", [attn_size])

            slot_inputs_shape = tf.shape(H)
            slot_inputs = tf.reshape(H, [-1, attn_size])
            y = core_rnn_cell._linear(slot_inputs, attn_size, True)
            y = tf.reshape(y, slot_inputs_shape)
            y = tf.expand_dims(y, 2)
            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])
            a = tf.nn.softmax(s)
            # a shape = [batch, input size, sentence length, 1]
            a = tf.expand_dims(a, -1)
            slot_d = tf.reduce_sum(a * hidden, [2])
            return slot_d

    def self_attention(self, H, hidden_num, head_num=1, reuse=False, scope_name=None):
        '''
            self-attention实现：
            输入是：H  shape:[batch_size, nstep, hidden_size]
            d_a: "self attention weight hidden units number"
            head_num: "self attention weight hops"
            alpha: "coefficient for self attention loss"
            tag 是命名空间的唯一名称
            输出是：[batch_size, hidden_num, hidden_size]
        '''
        state_shape = H.get_shape()
        hidden_size = state_shape[2].value
        self.self_atte_i += 1
        cur_name = "self_attention-{}".format(self.self_atte_i)
        cur_name = scope_name if reuse == tf.AUTO_REUSE and scope_name else cur_name
        # print cur_name, reuse, scope_name, "self_attention"
        with tf.variable_scope(cur_name, reuse=reuse):
            # Declare trainable variables for self attention
            with tf.name_scope("w_self_attention"):
                # shape(W_s1) = d_a * 2u
                W_s1 = tf.get_variable('W_s1', shape=[hidden_num, hidden_size],
                                       initializer=tf.contrib.layers.xavier_initializer())
                # shape(W_s2) = r * d_a
                W_s2 = tf.get_variable('W_s2', shape=[head_num, hidden_num],
                                       initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("attention"):
                # bs, head_num, nstep
                A = tf.nn.softmax(
                    tf.map_fn(
                        lambda x: tf.matmul(W_s2, x),
                        tf.tanh(
                            tf.map_fn(
                                lambda x: tf.matmul(W_s1, tf.transpose(x)),
                                H))))

                M = tf.matmul(A, H)
            return M, A

    def add_fc_layer(self, inputs, in_size, out_size, activation_function=None,
                     reuse=False, scope_name=None):
        """
        :param inputs: 二维， bs * d
        :param in_size: 输入的维度：d
        :param out_size: 输出的维度
        :param activation_function: 激活函数，如relu
        :return: bs * out_size
        """
        self.fc_i += 1
        cur_name = "fc_layer-{}".format(self.fc_i)
        cur_name = scope_name if reuse == tf.AUTO_REUSE and scope_name else cur_name
        # print cur_name, reuse, scope_name, "fc_layer"
        with tf.variable_scope(cur_name, reuse=reuse):
            wlimit = np.sqrt(6.0 / (in_size + out_size))
            # W = tf.Variable(tf.random_uniform([in_size, out_size], -wlimit, wlimit), name="W")
            # b = tf.Variable(tf.random_uniform([out_size], -wlimit, wlimit), name="bias")

            W = tf.get_variable("W", [in_size, out_size],
                                initializer=tf.random_normal_initializer(stddev=wlimit))
            b = tf.get_variable("bias", [out_size],
                                initializer=tf.random_normal_initializer(stddev=wlimit))
            Wx_plus_b = tf.matmul(inputs, W) + b
            if activation_function is None:
                fc_outputs = Wx_plus_b
            else:
                fc_outputs = activation_function(Wx_plus_b)
            return fc_outputs

    @staticmethod
    def reduce_avg(x, sequence_length, mask_zero=False, maxlen=None):
        """
        Args:
            x : [bs, nstep, d]
            sequence_length : [bs]
            mask_zero : mask padding
            maxlen: max length
        """
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.reshape(mask, (-1, maxlen, 1))
            mask = tf.cast(mask, tf.float32)
            z = tf.reduce_sum(x * mask, axis=1)
            l = tf.reduce_sum(mask, axis=1)
            # in some cases especially in the early stages of training the sum may be almost zero
            epsilon = 1e-8
            z /= tf.cast(l + epsilon, tf.float32)
        else:
            z = tf.reduce_mean(x, axis=1)
        return z

    def add_bert_placeholder(self):
        # shape = (batch_size, max length of sentence in batch)
        # 第一维：batch中的每一个sentence
        # 第二维：每一个sentence中的word id list，并根据最长sentence进行了0填充
        # bert 输入
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='bert_word_ids')
        self.bert_mask_ids = tf.placeholder(tf.int32, shape=[None, None], name='bert_mask_ids')
        self.bert_segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='bert_segment_ids')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name="bert_traing")

        # shape = (batch_size)
        # 第一维：batch中每一个sentence的长度
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # shape = (batch_size, max length of sentence, max length of word)
        # 第一维：batch中每一个sentence
        # 第二维：sentence中的每一个word
        # 第三维：每一个word中char id list
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

        # shape = (batch_size, max length of sentence)
        # 第一维：batch中的每一个sentence
        # 第二维：sentence中每一个word的长度
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

        # shape = (batch_size, max length of sentence)
        # 第一维：batch中的每一个sentence
        # 第二维：sentence中每一个word对应的tag
        self.ner_labels = tf.placeholder(tf.int32, shape=[None, None], name="meta_ner_labels")

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

        self.output = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name="output_embedding")
        self.labels_pred = tf.placeholder(dtype=tf.int32, shape=[None, None], name="labels_pred")
        self.slot_scores = tf.placeholder(dtype=tf.float32, shape=[None], name="pred_score")

    def add_bert_layer(self):
        with tf.variable_scope("bert"):
            bert_model = modeling.BertModel(
                config=self.bert_config,
                is_training=self.is_training,
                input_ids=self.word_ids,
                input_mask=self.bert_mask_ids,
                token_type_ids=self.bert_segment_ids,
                use_one_hot_embeddings=self.config.use_one_hot_embeddings)
            tvars = tf.trainable_variables()
            (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   self.config.bert_init_checkpoint)
            tf.train.init_from_checkpoint(self.config.bert_init_checkpoint, assignment)

            bert_output_seq_ori = bert_model.get_sequence_output()
            cur_shape = tf.shape(bert_output_seq_ori)
            self.bert_output_seq_ori = bert_output_seq_ori
            # bs, seq, 768
            self.bert_output_seq = tf.strided_slice(
                bert_output_seq_ori, [0, 1, 0], [cur_shape[0], cur_shape[1] - 1, cur_shape[2]], [1, 1, 1])
            self.bert_output_seq = tf.reshape(self.bert_output_seq,
                                              [-1, self.config.max_seq_len, self.bert_config.hidden_size])

            self.cls_output = bert_model.get_pooled_output()

    def get_bert_feed_dict(self, words, bert_mask, bert_seg,
                           labels=None, categories=None, actions=None,
                           kgs=None, dropout=None, learning_rate=None,
                           ser_tag=None, seq_len=None, is_training=False):
        """
        :param words: list of sentence.
               sentence: use_chars: [ ([char_ids], [char_ids], ...), (word_id, word_id, ...) ]
                         not use_chars: [word_id, word_id, ...]
        :param labels: list of sentence tags
               sentence tags: [tag1, tag2, ...]
        :param dropout: dropout的keep比例
        :param learning_rate: 学习率
        :return: dict[placeholder] = value
        """
        word_ids, sequence_lengths = pad_sequences(words, 0, self.config.bert_max_seq_len)
        bert_mask, _ = pad_sequences(bert_mask, 0, self.config.bert_max_seq_len)
        bert_seg, _ = pad_sequences(bert_seg, 0, self.config.bert_max_seq_len)

        feed = {
            self.word_ids: word_ids,
            self.bert_mask_ids: bert_mask,
            self.bert_segment_ids: bert_seg

        }

        kg_onehots, _ = pad_sequences(kgs, 0.0, self.config.max_seq_len, nlevels=2)
        feed[self.kg_onehots] = kg_onehots

        if labels is not None:
            labels, _ = pad_sequences(labels, 0, self.config.max_seq_len)
            feed[self.meta_ner_labels] = labels

        if categories is not None:
            feed[self.categories] = categories

        if actions is not None:
            feed[self.actions] = actions

        if dropout is not None:
            feed[self.dropout] = dropout

        if learning_rate is not None:
            feed[self.learning_rate] = learning_rate
        if ser_tag is not None:
            feed[self.service_ner_labels] = ser_tag
        if seq_len is not None:
            sequence_lengths = seq_len
        # sequence_lengths = [s if s < self.config.max_seq_len else self.config.max_seq_len for s in sequence_lengths]
        feed[self.is_training] = is_training
        feed[self.sequence_lengths] = sequence_lengths
        return feed, sequence_lengths

    def predict_bert_batch_service(self, words, kg_onehots, bert_mask_id, bert_seg_id, seq_len):
        """
        :param words: sentence列表
        :return:
        """
        fd, sequence_lengths = self.get_bert_feed_dict(words, bert_mask_id, bert_seg_id, kgs=kg_onehots, dropout=1.0,
                                                       seq_len=seq_len)
        service_tag_pred, cate_pred_max_idx, cate_pred_score = self.sess.run(
            [self.service_tag_pred, self.category_pred_max_idx, self.category_pred],
            feed_dict=fd)

        return service_tag_pred, sequence_lengths, cate_pred_max_idx, cate_pred_score


class FlipGradientBuilder(object):
    """
    Code: https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
    """

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y

