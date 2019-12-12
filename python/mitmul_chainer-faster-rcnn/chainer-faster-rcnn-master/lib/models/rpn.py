#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Variable
from chainer.cuda import to_gpu
from lib.faster_rcnn.anchor_target_layer import AnchorTargetLayer
from lib.faster_rcnn.proposal_layer import ProposalLayer
from lib.faster_rcnn.smooth_l1_loss import smooth_l1_loss

import chainer
import chainer.functions as F
import chainer.links as L


class RPN(chainer.Chain):

    def __init__(
            self, in_ch=512, out_ch=512, n_anchors=9, feat_stride=16,
            anchor_scales=[8, 16, 32], num_classes=21, rpn_sigma=3.0):
        super(RPN, self).__init__(
            rpn_conv_3x3=L.Convolution2D(in_ch, out_ch, 3, 1, 1),
            rpn_cls_score=L.Convolution2D(out_ch, 2 * n_anchors, 1, 1, 0),
            rpn_bbox_pred=L.Convolution2D(out_ch, 4 * n_anchors, 1, 1, 0)
        )
        self.anchor_target_layer = AnchorTargetLayer(feat_stride)
        self.proposal_layer = ProposalLayer(feat_stride, anchor_scales)
        self.rpn_sigma = rpn_sigma

    def __call__(self, x, im_info, gpu=-1, gt_boxes=None):
        n = x.data.shape[0]
        h = F.relu(self.rpn_conv_3x3(x))
        rpn_cls_score = self.rpn_cls_score(h)
        rpn_bbox_pred = self.rpn_bbox_pred(h)
        rpn_cls_score_reshape = F.reshape(rpn_cls_score, (n, 2, -1))

        # RoI Proposal
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        c, hh, ww = rpn_cls_score.data.shape[1:]
        rpn_cls_prob_reshape = F.reshape(rpn_cls_prob, (n, c, hh, ww))
        rois = self.proposal_layer(
            rpn_cls_prob_reshape, rpn_bbox_pred, im_info, gt_boxes is not None)

        if gt_boxes is not None:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, \
                rpn_bbox_outside_weights = self.anchor_target_layer(
                    rpn_cls_score, gt_boxes, im_info)
            rpn_labels = rpn_labels.reshape((1, -1))

            # make it into Variable
            if gpu >= 0:
                tg = lambda x: to_gpu(x, device=gpu)
                rpn_labels = tg(rpn_labels)
                rpn_bbox_targets = tg(rpn_bbox_targets)
                rpn_bbox_inside_weights = tg(rpn_bbox_inside_weights)
                rpn_bbox_outside_weights = tg(rpn_bbox_outside_weights)
            rpn_labels = Variable(rpn_labels, volatile='auto')
            rpn_bbox_targets = Variable(rpn_bbox_targets, volatile='auto')
            # rpn_bbox_inside_weights = Variable(
            #     rpn_bbox_inside_weights, volatile=volatile)
            # rpn_bbox_outside_weights = Variable(
            #     rpn_bbox_outside_weights, volatile=volatile)

            rpn_cls_loss = F.softmax_cross_entropy(
                rpn_cls_score_reshape, rpn_labels)
            rpn_loss_bbox = smooth_l1_loss(
                rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights, self.rpn_sigma)

            return rpn_cls_loss, rpn_loss_bbox, rois
        else:
            return rois
