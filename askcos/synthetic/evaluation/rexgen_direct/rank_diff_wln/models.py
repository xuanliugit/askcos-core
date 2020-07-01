import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from askcos.synthetic.evaluation.rexgen_direct.rank_diff_wln.mol_graph_direct_useScores import max_nb
from askcos.synthetic.evaluation.rexgen_direct.rank_diff_wln.nn import *

def rcnn_wl_last(graph_inputs, hidden_size, depth, training=True):
    input_atom, input_bond, atom_graph, bond_graph, num_nbs = graph_inputs
    atom_features = tf.nn.relu(linearND(input_atom, hidden_size, "atom_embedding", init_bias=None))
    layers = []
    for i in range(depth):
        with tf.variable_scope("WL", reuse=(i>0)) as scope:
            fatom_nei = tf.gather_nd(atom_features, atom_graph)
            fbond_nei = tf.gather_nd(input_bond, bond_graph)
            h_nei_atom = linearND(fatom_nei, hidden_size, "nei_atom", init_bias=None)
            h_nei_bond = linearND(fbond_nei, hidden_size, "nei_bond", init_bias=None)
            h_nei = h_nei_atom * h_nei_bond
            mask_nei = tf.sequence_mask(tf.reshape(num_nbs, [-1]), max_nb, dtype=tf.float32)
            target_shape = tf.concat([tf.shape(num_nbs), [max_nb, 1]], 0)
            mask_nei = tf.reshape(mask_nei, target_shape)
            mask_nei.set_shape([None, None, max_nb, 1])
            f_nei = tf.reduce_sum(h_nei * mask_nei, -2)
            f_self = linearND(atom_features, hidden_size, "self_atom", init_bias=None)
            layers.append(f_nei * f_self)
            l_nei = tf.concat([fatom_nei, fbond_nei], 3)
            nei_label = tf.nn.relu(linearND(l_nei, hidden_size, "label_U2"))
            nei_label = tf.reduce_sum(nei_label * mask_nei, -2) 
            new_label = tf.concat([atom_features, nei_label], 2)
            new_label = linearND(new_label, hidden_size, "label_U1")
            atom_features = tf.nn.relu(new_label)
    #kernels = tf.concat(1, layers)
    kernels = layers[-1]
    fp = tf.reduce_sum(kernels, 1)
    return kernels, fp

def rcnn_wl_only(graph_inputs, hidden_size, depth, training=True):
    input_atom, input_bond, atom_graph, bond_graph, num_nbs = graph_inputs
    atom_features = tf.nn.relu(linearND(input_atom, hidden_size, "atom_embedding", init_bias=None))
    layers = []
    for i in range(depth):
        with tf.variable_scope("WL", reuse=(i>0)) as scope:
            fatom_nei = tf.gather_nd(atom_features, atom_graph)
            fbond_nei = tf.gather_nd(input_bond, bond_graph)

            mask_nei = tf.sequence_mask(tf.reshape(num_nbs, [-1]), max_nb, dtype=tf.float32)
            target_shape = tf.concat([tf.shape(num_nbs), [max_nb, 1]], 0)
            mask_nei = tf.reshape(mask_nei, target_shape)
            mask_nei.set_shape([None, None, max_nb, 1])

            l_nei = tf.concat([fatom_nei, fbond_nei], 3)
            nei_label = tf.nn.relu(linearND(l_nei, hidden_size, "label_U2"))
            nei_label = tf.reduce_sum(nei_label * mask_nei, -2) 
            new_label = tf.concat([atom_features, nei_label], 2)
            new_label = linearND(new_label, hidden_size, "label_U1")
            atom_features = tf.nn.relu(new_label)

    return atom_features

def wl_diff_net(graph_inputs, atom_features, hidden_size, depth):
    input_atom, input_bond, atom_graph, bond_graph, num_nbs = graph_inputs
    for i in range(depth):
        with tf.variable_scope("WL", reuse=(i>0)) as scope:
            fatom_nei = tf.gather_nd(atom_features, atom_graph)
            fbond_nei = tf.gather_nd(input_bond, bond_graph)

            mask_nei = tf.sequence_mask(tf.reshape(num_nbs, [-1]), max_nb, dtype=tf.float32)
            target_shape = tf.concat([tf.shape(num_nbs), [max_nb, 1]], 0)
            mask_nei = tf.reshape(mask_nei, target_shape)
            mask_nei.set_shape([None, None, max_nb, 1])

            l_nei = tf.concat([fatom_nei, fbond_nei], 3)
            nei_label = tf.nn.relu(linearND(l_nei, hidden_size, "label_U2"))
            nei_label = tf.reduce_sum(nei_label * mask_nei, -2) 
            new_label = tf.concat([atom_features, nei_label], 2)
            new_label = linearND(new_label, hidden_size, "label_U1")
            atom_features = tf.nn.relu(new_label)

    return tf.reduce_sum(atom_features, -2)

