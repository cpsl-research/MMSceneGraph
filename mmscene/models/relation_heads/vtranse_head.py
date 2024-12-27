import torch
import torch.nn.functional as F
import torch.nn as nn
from mmengine.model import BaseModule, kaiming_init

from mmscene.registry import MODELS
from .relation_head import RelationHead
from .utils import obj_edge_vectors, encode_box_info, to_onehot


@MODELS.register_module()
class VTransEHead(RelationHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # word embedding
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.glove_dir, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])


        # # initialize the vtranse context
        # self.context_layer = VTransEContext(self.head_config, self.obj_classes, self.rel_classes)

    def init_weights(self):
        super().init_weights()
        for m in self.pos_embed:
            if isinstance(m, nn.Linear):
                kaiming_init(m, distribution='uniform', a=1)
        kaiming_init(self.pred_layer, a=1, distribution='uniform')
        kaiming_init(self.fc_layer, a=1, distribution='uniform')

    def forward(self,
                img,
                img_meta,
                det_result,
                gt_result=None,
                is_testing=False,
                ignore_classes=None):
        """
        Obtain the relation prediction results based on detection results.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            det_result: (Result): Result containing bbox, label, mask, point, rels,
                etc. According to different mode, all the contents have been
                set correctly. Feel free to  use it.
            gt_result : (Result): The ground truth information.
            is_testing:

        Returns:
            det_result with the following newly added keys:
                refine_scores (list[Tensor]): logits of object
                rel_scores (list[Tensor]): logits of relation
                rel_pair_idxes (list[Tensor]): (num_rel, 2) index of subject and object
                relmaps (list[Tensor]): (num_obj, num_obj):
                target_rel_labels (list[Tensor]): the target relation label.
        """
        roi_feats, union_feats, det_result = self.frontend_features(img, img_meta, det_result, gt_result)
        if roi_feats.shape[0] == 0:
            return det_result

        refine_obj_scores, obj_preds, edge_ctx, _ = self.backend_relations(roi_feats, det_result)

        if is_testing and ignore_classes is not None:
            refine_obj_scores = self.process_ignore_objects(refine_obj_scores, ignore_classes)
            obj_preds = refine_obj_scores[:, 1:].max(1)[1] + 1

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.context_pooling_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.context_pooling_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.context_pooling_dim)

        num_rels = [r.shape[0] for r in det_result.rel_pair_idxes]
        num_objs = [len(b) for b in det_result.bboxes]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(det_result.rel_pair_idxes, head_reps, tail_reps, obj_preds):
            prod_reps.append(head_rep[pair_idx[:, 0]] - tail_rep[pair_idx[:, 1]])
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = torch.cat(prod_reps, dim=0)
        pair_pred = torch.cat(pair_preds, dim=0)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_feats)
            else:
                prod_rep = prod_rep * union_feats

        rel_scores = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_scores = rel_scores + self.freq_bias.index_with_labels(pair_pred.long())

        # make some changes: list to tensor or tensor to tuple
        if self.training:
            det_result.target_labels = torch.cat(det_result.target_labels, dim=-1)
            det_result.target_rel_labels = torch.cat(det_result.target_rel_labels,
                                                     dim=-1) if det_result.target_rel_labels is not None else None
        else:
            refine_obj_scores = refine_obj_scores.split(num_objs, dim=0)
            rel_scores = rel_scores.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage

        det_result.refine_scores = refine_obj_scores
        det_result.rel_scores = rel_scores

        # ranking prediction:
        if self.with_relation_ranker:
            det_result = self.relation_ranking_forward(prod_rep, det_result, gt_result, num_rels, is_testing)
        return det_result
    
    def backend_relations(self, x, det_result, all_average=False, ctx_average=False):
        num_objs = [len(b) for b in det_result.bboxes]
        # labels will be used in DecoderRNN during training (for nms)
        if self.training or self.use_gt_box:
            obj_labels = torch.cat(det_result.labels)
        else:
            obj_labels = None

        if self.use_gt_label:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_dists = torch.cat(det_result.dists, dim=0).detach()
            obj_embed = obj_dists @ self.obj_embed1.weight

        pos_embed = self.pos_embed(encode_box_info(det_result))

        batch_size = x.shape[0]

        obj_pre_rep = torch.cat((x, obj_embed, pos_embed), -1)

        # object level contextual feature
        if self.mode != 'predcls':
            obj_scores = self.pred_layer(obj_pre_rep)
            obj_dists = F.softmax(obj_scores, dim=1)
            obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_scores = to_onehot(obj_preds, self.num_classes)

        # edge level contextual feature
        obj_embed2 = self.obj_embed2(obj_preds.long())
        obj_rel_rep = torch.cat((x, pos_embed, obj_embed2), -1)

        edge_ctx = F.relu(self.fc_layer(obj_rel_rep))

        return obj_scores, obj_preds, edge_ctx, None