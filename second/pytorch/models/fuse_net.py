import numpy as np
import torch
import torch.nn as nn
import torchplus
from torchplus.tools import change_default_args
from torchplus.nn import Empty, GroupNorm, Sequential
from second.pytorch.core import box_torch_ops
from second.core.roi_pool.model.roi_layers import ROIPool

def get_direction_target(anchors,
                         reg_targets,
                         one_hot=True,
                         dir_offset=0,
                         num_bins=2):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = box_torch_ops.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(
            dir_cls_targets, num_bins, dtype=anchors.dtype)
    return dir_cls_targets

def rpn_nms(box_preds, cls_preds, example, box_coder, nms_score_threshold, nms_pre_max_size,
            nms_post_max_size, nms_iou_threshold, training, range_thresh=0):
    anchors = example["anchors"]
    batch_size = anchors.shape[0]
    batch_anchors = anchors.view(batch_size, -1, 7)
    batch_rect = example["calib"]["rect"]
    batch_Trv2c = example["calib"]["Trv2c"]
    batch_P2 = example["calib"]["P2"]
    if training:
        batch_labels = example["labels"]
        batch_reg_targets = example["reg_targets"]
        batch_dir_targets = get_direction_target(
            batch_anchors,
            batch_reg_targets,
            dir_offset=0.0,
            num_bins=2)
    else:
        batch_labels = [None] * batch_size
        batch_reg_targets = [None] * batch_size
        batch_dir_targets = [None] * batch_size

    if "anchors_mask" not in example:
        batch_anchors_mask = [None] * batch_size
    else:
        anchors_mask = example["anchors_mask"]
        batch_anchors_mask = anchors_mask.view(batch_size, -1)
    batch_box_props = box_preds.view(batch_size, -1, box_coder.code_size)
    batch_box_props = box_coder.decode_torch(batch_box_props, batch_anchors)
    batch_cls_props = cls_preds.view(batch_size, -1, 1)

    batch_far_proposals_bev = []
    batch_far_proposals_img = []
    batch_near_proposals_bev = []
    batch_near_proposals_img = []
    batch_rcnn_labels = []
    batch_rcnn_reg_target = []
    batch_rcnn_dir_target = []
    batch_rcnn_anchors = []
    for box_props, cls_props, labels, reg_target, dir_targets, rect, Trv2c, P2, a_mask, anchors in zip(
            batch_box_props, batch_cls_props, batch_labels, batch_reg_targets, batch_dir_targets,
            batch_rect, batch_Trv2c, batch_P2, batch_anchors_mask, batch_anchors):
        if a_mask is not None:
            box_props = box_props[a_mask]
            cls_props = cls_props[a_mask]
            anchors = anchors[a_mask]
            if training:
                labels = labels[a_mask]
                reg_target = reg_target[a_mask]
                dir_targets = dir_targets[a_mask]
        cls_scores = torch.sigmoid(cls_props)[..., 1:]
        top_scores = cls_props.squeeze(-1)
        nms_func = box_torch_ops.nms
        if nms_score_threshold > 0.0:
            thresh = torch.Tensor([nms_score_threshold],
                                  device=cls_scores.cpu().device).type_as(cls_scores)
            top_scores_keep = (top_scores >= thresh)
            top_scores = top_scores.masked_select(top_scores_keep)
        if top_scores.shape[0] != 0:
            # score threshold
            if nms_score_threshold > 0.0:
                box_props = box_props[top_scores_keep]
                anchors = anchors[top_scores_keep]
                if training:
                    labels = labels[top_scores_keep]
                    reg_target = reg_target[top_scores_keep]
                    dir_targets = dir_targets[top_scores_keep]
            # range
            range_thresh = torch.Tensor([range_thresh],
                                        device=box_props.cpu().device).type_as(box_props)
            # todo: uncertain, which is range
            far_boxes_idx = (box_props[:, 0] >= range_thresh)

            far_box_props = box_props[far_boxes_idx]
            far_top_socres = top_scores[far_boxes_idx]
            far_anchors = anchors[far_boxes_idx]
            if training:
                far_labels = labels[far_boxes_idx]
                far_reg_target = reg_target[far_boxes_idx]
                far_dir_target = dir_targets[far_boxes_idx]
            if far_box_props.shape[0] != 0:
                far_boxes_for_nms = far_box_props[:, [0, 1, 3, 4, 6]]
                far_box_props_corners = box_torch_ops.center_to_corner_box2d(
                    far_boxes_for_nms[:, :2], far_boxes_for_nms[:, 2:4],
                    far_boxes_for_nms[:, 4])
                far_boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    far_box_props_corners)

                far_selected = nms_func(
                    far_boxes_for_nms,
                    far_top_socres,
                    pre_max_size=nms_pre_max_size // 2,
                    post_max_size=nms_post_max_size // 2,
                    iou_threshold=nms_iou_threshold)
            else:
                far_selected = None

            if range_thresh > 0:
                near_boxes_idx = (box_props[:, 0] < range_thresh)
                near_box_props = box_props[near_boxes_idx]
                near_anchors = anchors[near_boxes_idx]
                near_top_socres = top_scores[near_boxes_idx]
                if training:
                    near_labels = labels[near_boxes_idx]
                    near_reg_target = reg_target[near_boxes_idx]
                    near_dir_target = dir_targets[near_boxes_idx]
                if near_box_props.shape[0] != 0:
                    near_boxes_for_nms = near_box_props[:, [0, 1, 3, 4, 6]]
                    near_box_props_corners = box_torch_ops.center_to_corner_box2d(
                        near_boxes_for_nms[:, :2], near_boxes_for_nms[:, 2:4],
                        near_boxes_for_nms[:, 4])
                    near_boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        near_box_props_corners)
                    near_selected = nms_func(
                        near_boxes_for_nms,
                        near_top_socres,
                        pre_max_size=nms_pre_max_size,
                        post_max_size=nms_post_max_size,
                        iou_threshold=nms_iou_threshold)
                else:
                    near_selected = None
            else:
                near_selected = None
        else:
            far_selected = None
            near_selected = None

        if far_selected is not None:
            far_proposals_3d = far_box_props[far_selected]
            num_far_selected = far_proposals_3d.shape[0]

            far_proposals_3d_fix = torch.zeros((nms_post_max_size // 2, 7)).cuda()
            far_anchors_fix = torch.zeros((nms_post_max_size // 2, 7)).cuda()
            far_proposals_3d_fix[:num_far_selected, :] = far_proposals_3d
            far_anchors_fix[:num_far_selected, :] = far_anchors[far_selected]
            far_anchors_fix = far_anchors_fix.unsqueeze(0)

            if training:
                far_labels_fix = torch.zeros((nms_post_max_size // 2)).cuda()
                far_reg_target_fix = torch.zeros((nms_post_max_size // 2, 7)).cuda()
                far_dir_target_fix = torch.zeros((nms_post_max_size // 2, 2)).cuda()

                far_labels_fix[:num_far_selected] = far_labels[far_selected]
                far_reg_target_fix[:num_far_selected, :] = far_reg_target[far_selected]
                far_dir_target_fix[:num_far_selected, :] = far_dir_target[far_selected]
                far_labels_fix = far_labels_fix.unsqueeze(0)
                far_reg_target_fix = far_reg_target_fix.unsqueeze(0)
                far_dir_target_fix = far_dir_target_fix.unsqueeze(0)

            far_proposals_bev_fix = far_proposals_3d_fix[:, [0, 1, 3, 4, 6]].unsqueeze(0)
            far_proposals_cam_fix = box_torch_ops.box_lidar_to_camera(far_proposals_3d_fix, rect, Trv2c)
            far_locs_cam = far_proposals_cam_fix[:, :3]
            far_dims_cam = far_proposals_cam_fix[:, 3:6]
            far_angles_cam = far_proposals_cam_fix[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            far_proposals_cam_corners = box_torch_ops.center_to_corner_box3d(
                far_locs_cam, far_dims_cam, far_angles_cam, camera_box_origin, axis=1)
            far_proposals_img_corners = box_torch_ops.project_to_image(
                far_proposals_cam_corners, P2)
            minxy = torch.min(far_proposals_img_corners, dim=1)[0]
            maxxy = torch.max(far_proposals_img_corners, dim=1)[0]
            far_proposals_img_fix = torch.cat([minxy, maxxy], dim=1).unsqueeze(0)
        else:
            far_proposals_bev_fix = torch.zeros((nms_post_max_size // 2, 5)).cuda().unsqueeze(0)
            far_proposals_img_fix = torch.zeros((nms_post_max_size // 2, 4)).cuda().unsqueeze(0)
            far_labels_fix = torch.zeros((nms_post_max_size // 2)).cuda().unsqueeze(0)
            far_reg_target_fix = torch.zeros((nms_post_max_size // 2, 7)).cuda().unsqueeze(0)
            far_dir_target_fix = torch.zeros((nms_post_max_size // 2, 2)).cuda().unsqueeze(0)
            far_anchors_fix = torch.zeros((nms_post_max_size // 2, 7)).cuda().unsqueeze(0)

        if near_selected is not None:
            near_proposals_3d = near_box_props[near_selected]
            num_near_selected = near_proposals_3d.shape[0]
            near_proposals_3d_fix = torch.zeros((nms_post_max_size, 7)).cuda()
            near_anchors_fix = torch.zeros((nms_post_max_size, 7)).cuda()

            near_proposals_3d_fix[:num_near_selected, :] = near_proposals_3d
            near_anchors_fix[:num_near_selected, :] = near_anchors[near_selected]
            near_anchors_fix = near_anchors_fix.unsqueeze(0)

            if training:
                near_labels_fix = torch.zeros((nms_post_max_size,)).cuda()
                near_reg_target_fix = torch.zeros((nms_post_max_size, 7)).cuda()
                near_dir_target_fix = torch.zeros((nms_post_max_size, 2)).cuda()

                near_labels_fix[:num_near_selected] = near_labels[near_selected]
                near_reg_target_fix[:num_near_selected, :] = near_reg_target[near_selected]
                near_dir_target_fix[:num_near_selected, :] = near_dir_target[near_selected]
                near_labels_fix = near_labels_fix.unsqueeze(0)
                near_reg_target_fix = near_reg_target_fix.unsqueeze(0)
                near_dir_target_fix = near_dir_target_fix.unsqueeze(0)

            near_proposals_bev_fix = near_proposals_3d_fix[:, [0, 1, 3, 4, 6]].unsqueeze(0)
            near_proposals_cam_fix = box_torch_ops.box_lidar_to_camera(near_proposals_3d_fix, rect, Trv2c)
            near_locs_cam = near_proposals_cam_fix[:, :3]
            near_dims_cam = near_proposals_cam_fix[:, 3:6]
            near_angles_cam = near_proposals_cam_fix[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            near_proposals_cam_corners = box_torch_ops.center_to_corner_box3d(
                near_locs_cam, near_dims_cam, near_angles_cam, camera_box_origin, axis=1)
            near_proposals_img_corners = box_torch_ops.project_to_image(
                near_proposals_cam_corners, P2)
            near_minxy = torch.min(near_proposals_img_corners, dim=1)[0]
            near_maxxy = torch.max(near_proposals_img_corners, dim=1)[0]
            near_proposals_img_fix = torch.cat([near_minxy, near_maxxy], dim=1).unsqueeze(0)
        else:
            near_proposals_bev_fix = torch.zeros((nms_post_max_size, 5)).cuda().unsqueeze(0)
            near_proposals_img_fix = torch.zeros((nms_post_max_size, 4)).cuda().unsqueeze(0)
            near_labels_fix = torch.zeros((nms_post_max_size)).cuda().unsqueeze(0)
            near_reg_target_fix = torch.zeros((nms_post_max_size, 7)).cuda().unsqueeze(0)
            near_dir_target_fix = torch.zeros((nms_post_max_size, 2)).cuda().unsqueeze(0)
            near_anchors_fix = torch.zeros((nms_post_max_size, 7)).cuda().unsqueeze(0)
        if training:
            rcnn_labels_fix = torch.cat([near_labels_fix, far_labels_fix], dim=1)
            rcnn_reg_target_fix = torch.cat([near_reg_target_fix, far_reg_target_fix], dim=1)
            rcnn_dir_target_fix = torch.cat([near_dir_target_fix, far_dir_target_fix], dim=1)
        else:
            rcnn_labels_fix = None
            rcnn_reg_target_fix = None
            rcnn_dir_target_fix = None
        if near_anchors_fix is not None:
            rcnn_anchors_fix = torch.cat([near_anchors_fix, far_anchors_fix], dim=1)
        batch_far_proposals_bev.append(far_proposals_bev_fix)
        batch_far_proposals_img.append(far_proposals_img_fix)
        batch_near_proposals_bev.append(near_proposals_bev_fix)
        batch_near_proposals_img.append(near_proposals_img_fix)
        batch_rcnn_labels.append(rcnn_labels_fix)
        batch_rcnn_reg_target.append(rcnn_reg_target_fix)
        batch_rcnn_dir_target.append(rcnn_dir_target_fix)
        batch_rcnn_anchors.append(rcnn_anchors_fix)
    batch_far_proposals_bev = torch.cat(batch_far_proposals_bev, dim=0)
    batch_far_proposals_img = torch.cat(batch_far_proposals_img, dim=0)
    if batch_near_proposals_bev[0] is not None:
        batch_near_proposals_bev = torch.cat(batch_near_proposals_bev, dim=0)
        batch_near_proposals_img = torch.cat(batch_near_proposals_img, dim=0)

    if training:
        batch_rcnn_labels = torch.cat(batch_rcnn_labels, dim=0)
        batch_rcnn_reg_target = torch.cat(batch_rcnn_reg_target, dim=0)
        batch_rcnn_dir_target = torch.cat(batch_rcnn_dir_target, dim=0)
    batch_rcnn_anchors = torch.cat(batch_rcnn_anchors, dim=0)
    rcnn_examples = {
        "far_props_bev": batch_far_proposals_bev,
        "far_props_img": batch_far_proposals_img,
        "near_props_bev": batch_near_proposals_bev,
        "near_props_img": batch_near_proposals_img,
        "rcnn_labels": batch_rcnn_labels,
        "rcnn_reg_targets": batch_rcnn_reg_target,
        "rcnn_dir_targets": batch_rcnn_dir_target,
        "rcnn_anchors": batch_rcnn_anchors
}
    return rcnn_examples

def get_crops(feat_map, proposals):
    ones_mat = np.ones(proposals.shape[:2], dtype=np.int32)
    idx = torch.unsqueeze(torch.arange(0, proposals.shape[0]), 1)
    rois_idx = ones_mat * idx
    rois_idx = np.reshape(rois_idx, [-1])  # [2000]
    # crop and resize
    crop_height = 7
    crop_width = 7

    # [2000, 192, 7, 7]
    crops_torch = ROIPool((crop_height, crop_width), 1.0/16.0)(feat_map.cuda(), proposals.view(-1, 5).cuda())
    return crops_torch

REGISTERED_RCNN_CLASSES = {}

def register_rcnn(cls, name=None):
    global REGISTERED_RCNN_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_RCNN_CLASSES, f"exist class: {REGISTERED_RCNN_CLASSES}"
    REGISTERED_RCNN_CLASSES[name] = cls
    return cls

def get_rcnn_class(name):
    global REGISTERED_RCNN_CLASSES
    assert name in REGISTERED_RCNN_CLASSES, f"available class: {REGISTERED_RCNN_CLASSES}"
    return REGISTERED_RCNN_CLASSES[name]

@register_rcnn
class RCNN(nn.Module):
    def __init__(self,
                 input_channels,
                 use_norm,
                 num_anchor_per_loc,
                 encode_background_as_zeros,
                 num_class,
                 box_code_size,
                 use_direction_classifier,
                 name="RCNN"):
        super(RCNN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_cls = num_class
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_direction_cls = use_direction_classifier
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size
        if encode_background_as_zeros:
            num_cls = num_class
        else:
            num_cls = (num_class + 1)

        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        else:
            BatchNorm2d = Empty
        self.input_channels = input_channels
        self.fc_layer1 = nn.Conv2d(input_channels * 7 * 7, 1024, 1)
        self.norm_layer1 = BatchNorm2d(1024)
        self.relu_layer1 = torch.nn.ReLU(inplace=False)
        self.fc_layer2 = nn.Conv2d(1024, 512, 1)
        self.norm_layer2 = BatchNorm2d(512)
        self.relu_layer2 = torch.nn.ReLU(inplace=False)

        self.conv_cls = nn.Conv2d(512, num_cls, 1)
        self.conv_box = nn.Conv2d(512, box_code_size, 1)

        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(512, 2, 1)

    def forward(self, fuse_props, near_props, training):
        if near_props is not None:
            props = torch.cat([near_props, fuse_props], dim=1)
        else:
            props = fuse_props.clone()
        # fully connected layer, replaced by conv11
        props_flatten = props.view(-1, 7 * 7 * self.input_channels, 1,
                                   1).cuda()  # [num_proposal, 7*7*feat_channel, 1, 1]
        props_fc1 = self.fc_layer1(props_flatten)
        props_norm1 = self.norm_layer1(props_fc1)
        props_relu1 = self.relu_layer1(props_norm1)

        props_fc2 = self.fc_layer2(props_relu1)
        props_norm2 = self.norm_layer2(props_fc2)
        props_relu2 = self.relu_layer2(props_norm2)

        box_preds = self.conv_box(props_relu2).squeeze(2)  # [2000, 7, 1]
        box_preds = box_preds.permute(2, 0, 1).contiguous()  # [1, 2000, 7]

        cls_preds = self.conv_cls(props_relu2).squeeze(2)  # [2000, 1, 1]
        cls_preds = cls_preds.permute(2, 0, 1).contiguous()  # [1, 2000, 1]
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(props_relu2).squeeze(2)  # [2000, 2, 1]
            dir_cls_preds = dir_cls_preds.permute(2, 0, 1).contiguous()  # [1, 2000, 2]
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict