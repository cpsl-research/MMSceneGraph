import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.nn.utils.rnn import PackedSequence
from mmengine.model import BaseModule
import itertools

from mmscene.registry import MODELS

from ..utils import obj_edge_vectors


@MODELS.register_module()
class DecoderRNN(BaseModule):
    def __init__(self, config, obj_classes, embed_dim, inputs_dim, hidden_dim, rnn_drop):
        super(DecoderRNN, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.embed_dim = embed_dim

        obj_embed_vecs = obj_edge_vectors(['start'] + self.obj_classes, wv_dir=self.cfg.glove_dir, wv_dim=embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes) + 1, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.hidden_size = hidden_dim
        self.inputs_dim = inputs_dim
        self.input_size = self.inputs_dim + self.embed_dim
        self.nms_thresh = 0.5
        self.rnn_drop = rnn_drop

        self.input_linearity = torch.nn.Linear(self.input_size, 6 * self.hidden_size, bias=True)
        self.state_linearity = torch.nn.Linear(self.hidden_size, 5 * self.hidden_size, bias=True)
        self.out_obj = nn.Linear(self.hidden_size, len(self.obj_classes))

    def init_weights(self):
        # Use sensible default initializations for parameters.
        block_orthogonal(self.input_linearity.weight.data, [self.hidden_size, self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.hidden_size, self.hidden_size])

        self.state_linearity.bias.data.fill_(0.0)
        # Initialize forget gate biases to 1.0 as per An Empirical
        # Exploration of Recurrent Network Architectures, (Jozefowicz, 2015).
        self.state_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

        self.input_linearity.bias.data.fill_(0.0)
        self.input_linearity.bias.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def lstm_equations(self, timestep_input, previous_state, previous_memory, dropout_mask=None):
        """
        Does the hairy LSTM math
        :param timestep_input:
        :param previous_state:
        :param previous_memory:
        :param dropout_mask:
        :return:
        """
        # Do the projections for all the gates all at once.
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        # Main LSTM equations using relevant chunks of the big linear
        # projections of the hidden state and inputs.
        input_gate = torch.sigmoid(projected_input[:, 0 * self.hidden_size:1 * self.hidden_size] +
                                   projected_state[:, 0 * self.hidden_size:1 * self.hidden_size])
        forget_gate = torch.sigmoid(projected_input[:, 1 * self.hidden_size:2 * self.hidden_size] +
                                    projected_state[:, 1 * self.hidden_size:2 * self.hidden_size])
        memory_init = torch.tanh(projected_input[:, 2 * self.hidden_size:3 * self.hidden_size] +
                                 projected_state[:, 2 * self.hidden_size:3 * self.hidden_size])
        output_gate = torch.sigmoid(projected_input[:, 3 * self.hidden_size:4 * self.hidden_size] +
                                    projected_state[:, 3 * self.hidden_size:4 * self.hidden_size])
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        highway_gate = torch.sigmoid(projected_input[:, 4 * self.hidden_size:5 * self.hidden_size] +
                                     projected_state[:, 4 * self.hidden_size:5 * self.hidden_size])
        highway_input_projection = projected_input[:, 5 * self.hidden_size:6 * self.hidden_size]
        timestep_output = highway_gate * timestep_output + (1 - highway_gate) * highway_input_projection

        # Only do dropout if the dropout prob is > 0.0 and we are in training mode.
        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(self, inputs, initial_state=None, labels=None, boxes_for_nms=None):
        if not isinstance(inputs, PackedSequence):
            raise ValueError('inputs must be PackedSequence but got %s' % (type(inputs)))

        assert isinstance(inputs, PackedSequence)
        sequence_tensor, batch_lengths, _, _ = inputs
        batch_size = batch_lengths[0]

        # We're just doing an LSTM decoder here so ignore states, etc
        if initial_state is None:
            previous_memory = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
            previous_state = sequence_tensor.new().resize_(batch_size, self.hidden_size).fill_(0)
        else:
            assert len(initial_state) == 2
            previous_memory = initial_state[1].squeeze(0)
            previous_state = initial_state[0].squeeze(0)

        previous_obj_embed = self.obj_embed.weight[0, None].expand(batch_size, self.embed_dim)

        if self.rnn_drop > 0.0:
            dropout_mask = get_dropout_mask(self.rnn_drop, previous_memory.size(), previous_memory.device)
        else:
            dropout_mask = None

        # Only accumulating label predictions here, discarding everything else
        out_dists = []
        out_commitments = []

        end_ind = 0
        for i, l_batch in enumerate(batch_lengths):
            start_ind = end_ind
            end_ind = end_ind + l_batch

            if previous_memory.size(0) != l_batch:
                previous_memory = previous_memory[:l_batch]
                previous_state = previous_state[:l_batch]
                previous_obj_embed = previous_obj_embed[:l_batch]
                if dropout_mask is not None:
                    dropout_mask = dropout_mask[:l_batch]

            timestep_input = torch.cat((sequence_tensor[start_ind:end_ind], previous_obj_embed), 1)

            previous_state, previous_memory = self.lstm_equations(timestep_input, previous_state,
                                                                  previous_memory, dropout_mask=dropout_mask)

            pred_dist = self.out_obj(previous_state)
            out_dists.append(pred_dist)

            if self.training:
                labels_to_embed = labels[start_ind:end_ind].clone()
                # Whenever labels are 0 set input to be our max prediction
                nonzero_pred = pred_dist[:, 1:].max(1)[1] + 1
                is_bg = (labels_to_embed == 0).nonzero()
                if is_bg.dim() > 0:
                    labels_to_embed[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
                out_commitments.append(labels_to_embed)
                previous_obj_embed = self.obj_embed(labels_to_embed + 1)
            else:
                # assert l_batch == 1
                out_dist_sample = F.softmax(pred_dist, dim=1)
                best_ind = out_dist_sample[:, 1:].max(1)[1] + 1
                out_commitments.append(best_ind)
                previous_obj_embed = self.obj_embed(best_ind + 1)

        out_commitments = torch.cat(out_commitments, 0)

        return torch.cat(out_dists, 0), out_commitments


def get_dropout_mask(dropout_probability, tensor_shape, device):
    """
    once get, it is fixed all the time
    """
    binary_mask = (torch.rand(tensor_shape) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().to(device).div(1.0 - dropout_probability)
    return dropout_mask


def block_orthogonal(tensor, split_sizes, gain=1.0):
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        # let's not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]