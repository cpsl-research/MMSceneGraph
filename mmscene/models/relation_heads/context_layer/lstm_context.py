import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from mmengine.model import BaseModule

from mmscene.registry import MODELS
from mmscene.utils import ConfigType
from ..utils import obj_edge_vectors


@MODELS.register_module()
class LSTMContext(BaseModule):
    """
    Modified from neural-motifs to encode contexts for each objects
    """

    def __init__(self,
                 num_classes: int,
                 num_predictes: int,
                 decoder: ConfigType,
                 embed_dim: int=256,
                 hidden_dim: int=512,
                 roi_dim: int=1024,
                 context_pooling_dim: int=4096,
                 dropout_rate: float=0.2,
                 context_object_layer: int=1,
                 context_edge_layer: int=1
                 obj_classes, rel_classes):
        
        super(LSTMContext, self).__init__()
        self.num_classes = num_classes
        self.num_predicates = num_predictes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.roi_dim = roi_dim
        self.context_pooling_dim = context_pooling_dim
        self.dropout_rate = dropout_rate
        self.context_object_layer = context_object_layer
        self.context_edge_layer = context_edge_layer

        # rename
        self.nl_obj = self.context_object_layer
        self.nl_edge = self.context_edge_layer
        assert self.nl_obj > 0 and self.nl_edge > 0

        self.use_gt_box = self.cfg.use_gt_box
        self.use_gt_label = self.cfg.use_gt_label

        # mode
        if self.cfg.use_gt_box:
            if self.cfg.use_gt_label:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # word embedding
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.glove_dir, wv_dim=self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # TODO
        # AlternatingHighwayLSTM is invalid for pytorch 1.0
        self.obj_ctx_rnn = torch.nn.LSTM(
            input_size=self.roi_dim + self.embed_dim + 128,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_obj,
            dropout=self.dropout_rate if self.nl_obj > 1 else 0,
            bidirectional=True)
        self.decoder_rnn = MODELS.build(decoder)
        
        
        DecoderRNN(self.cfg, self.obj_classes, embed_dim=self.embed_dim,
                                      inputs_dim=self.hidden_dim + self.roi_dim + self.embed_dim + 128,
                                      hidden_dim=self.hidden_dim,
                                      rnn_drop=self.dropout_rate)
        self.edge_ctx_rnn = torch.nn.LSTM(
            input_size=self.embed_dim + self.hidden_dim + self.roi_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.nl_edge,
            dropout=self.dropout_rate if self.nl_edge > 1 else 0,
            bidirectional=True)
        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.lin_obj_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.lin_edge_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = self.cfg.causal_effect_analysis

        if self.effect_analysis:
            self.register_buffer("untreated_dcd_feat",
                                 torch.zeros(self.hidden_dim + self.roi_dim + self.embed_dim + 128))
            self.register_buffer("untreated_obj_feat", torch.zeros(self.roi_dim + self.embed_dim + 128))
            self.register_buffer("untreated_edg_feat", torch.zeros(self.embed_dim + self.roi_dim))

    def init_weights(self):
        self.decoder_rnn.init_weights()
        for m in self.pos_embed:
            if isinstance(m, nn.Linear):
                kaiming_init(m, distribution='uniform', a=1)
        kaiming_init(self.lin_obj_h, distribution='uniform', a=1)
        kaiming_init(self.lin_edge_h, distribution='uniform', a=1)

    def sort_rois(self, det_result):
        c_x = center_x(det_result)
        # leftright order
        scores = c_x / (c_x.max() + 1)
        return sort_by_score(det_result, scores)

    def obj_ctx(self, obj_feats, det_result, obj_labels=None, ctx_average=False):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_labels: [num_obj] the GT labels of the image
        :param box_priors: [num_obj, 4] boxes. We'll use this for NMS
        :param boxes_per_cls
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        perm, inv_perm, ls_transposed = self.sort_rois(det_result)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        encoder_rep = self.lin_obj_h(encoder_rep)  # map to hidden_dim

        # untreated decoder input
        batch_size = encoder_rep.shape[0]

        if (not self.training) and self.effect_analysis and ctx_average:
            decoder_inp = self.untreated_dcd_feat.view(1, -1).expand(batch_size, -1)
        else:
            decoder_inp = torch.cat((obj_inp_rep, encoder_rep), 1)

        if self.training and self.effect_analysis:
            self.untreated_dcd_feat = self.moving_average(self.untreated_dcd_feat, decoder_inp)

        # Decode in order
        if self.mode != 'predcls':
            decoder_inp = PackedSequence(decoder_inp, ls_transposed)
            obj_dists, obj_preds = self.decoder_rnn(
                decoder_inp,  # obj_dists[perm],
                labels=obj_labels[perm] if obj_labels is not None else None)
            obj_preds = obj_preds[inv_perm]
            obj_dists = obj_dists[inv_perm]
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds, encoder_rep, perm, inv_perm, ls_transposed

    def edge_ctx(self, inp_feats, perm, inv_perm, ls_transposed):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :return: edge_ctx: [num_obj, #feats] For later!
        """
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]
        edge_reps = self.lin_edge_h(edge_reps)  # map to hidden_dim

        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, x, det_result, all_average=False, ctx_average=False):
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.use_gt_box:  # predcls or sgcls or training, just put obj_labels here
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.use_gt_label:  # predcls
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed1.weight

        pos_embed = self.pos_embed(encode_box_info(det_result))  # N x 128

        batch_size = x.shape[0]
        if all_average and self.effect_analysis and (not self.training):  # TDE: only in test mode
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
        else:
            obj_pre_rep = torch.cat((x, obj_embed, pos_embed), -1)  # N x (1024 + 200 + 128)

        # object level contextual feature
        obj_dists, obj_preds, obj_ctx, perm, inv_perm, ls_transposed = self.obj_ctx(obj_pre_rep, det_result, obj_labels,
                                                                                    ctx_average=ctx_average)
        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())

        if (all_average or ctx_average) and self.effect_analysis and (not self.training):  # TDE: Testing
            obj_rel_rep = torch.cat((self.untreated_edg_feat.view(1, -1).expand(batch_size, -1), obj_ctx), dim=-1)
        else:
            obj_rel_rep = torch.cat((obj_embed2, x, obj_ctx), -1)

        edge_ctx = self.edge_ctx(obj_rel_rep, perm=perm, inv_perm=inv_perm, ls_transposed=ls_transposed)

        # memorize average feature
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, torch.cat((obj_embed2, x), -1))

        return obj_dists, obj_preds, edge_ctx, None
