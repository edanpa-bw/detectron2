# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, cat, interpolate
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

_TOTAL_SKIPPED = 0


__all__ = [
    "ROI_KEYPOINT_HEAD_REGISTRY",
    "build_keypoint_head",
    "BaseKeypointRCNNHead",
    "KRCNNConvDeconvUpsampleHead",
    "RangeKRCNNConvDeconvUpsampleHead"
]


ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")
ROI_KEYPOINT_HEAD_REGISTRY.__doc__ = """
Registry for keypoint heads, which make keypoint predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)


def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss


def range_keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer, range_imgs):
    """
    Compute the standard keypoint loss (cross entropy over heatmaps) and an additional range loss.
    This version vectorizes the operations for computing the range loss.

    Arguments:
        pred_keypoint_logits (Tensor): Shape (N, K, S, S) where:
            - N is the total number of instances in the batch,
            - K is the number of keypoints,
            - S is the side length of the keypoint heatmap.
        instances (list[Instances]): List (length = batch size) of Instances objects.
            Each Instances must have:
              - `proposal_boxes.tensor` of shape (n, 4)
              - `gt_keypoints.coordinates` of shape (n, K, 2)
        normalizer (float): Normalization factor. If None, use the number of visible keypoints.
        range_imgs (list[Tensor]): List (length = batch size) of range image tensors, each of shape (H, W).

    Returns:
        dict[str, Tensor]: {"loss_keypoint": ..., "loss_range": ...}
    """
    # --- Standard keypoint loss (remains mostly unchanged) ---
    heatmaps = []
    valid = []
    keypoint_side_len = pred_keypoint_logits.shape[2]

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        zero_loss = pred_keypoint_logits.sum() * 0
        return zero_loss, zero_loss, zero_loss

    N, K, S, _ = pred_keypoint_logits.shape
    pred_logits_orig = pred_keypoint_logits  # shape (N, K, S, S)
    pred_keypoint_logits_flat = pred_keypoint_logits.view(N * K, S * S)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits_flat[valid], keypoint_targets[valid], reduction="sum"
    )
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    # --- Vectorized range loss computation ---
    # We'll compute a differentiable soft-argmax for all keypoint heatmaps per image.
    # For each image, we need to:
    #   (a) Extract the predicted heatmaps for that image (shape: (n, K, S, S))
    #   (b) Compute a weighted average over the SxS grid to get predicted (x, y) in heatmap space,
    #       then transform to image coordinates using the proposal boxes.
    #   (c) Convert both predicted and ground-truth keypoints to normalized coordinates and
    #       sample the range image via F.grid_sample.

    device = pred_keypoint_logits.device
    range_loss_total = 0.0
    total_keypoints = 0
    offset = 0  # tracks instance offset into pred_logits_orig

    # Pre-compute coordinate grid for heatmaps: these are the “pixel indices” in the SxS grid.
    grid_vals = torch.arange(S * S, device=device, dtype=torch.float32)
    grid_x = (grid_vals % S).view(1, 1, -1)  # shape (1, 1, S*S)
    grid_y = (grid_vals // S).view(1, 1, -1)  # shape (1, 1, S*S)

    # Function to normalize image coordinates for grid_sample: output in [-1, 1]
    def normalize_coords(coords, H_img, W_img):
        # coords: (n, K, 2) with (x, y) in image coordinates
        norm_x = (coords[..., 0] / (W_img - 1)) * 2 - 1
        norm_y = (coords[..., 1] / (H_img - 1)) * 2 - 1
        return torch.stack([norm_x, norm_y], dim=2)  # (n, K, 2)

    # Process each image in the batch
    for i, instances_per_image in enumerate(instances):
        n_i = len(instances_per_image)
        if n_i == 0:
            continue

        # Retrieve proposal boxes and ground-truth keypoints for the image:
        # boxes: shape (n_i, 4); gt_keypoints: shape (n_i, K, 2)
        boxes = instances_per_image.proposal_boxes.tensor  # [x1, y1, x2, y2]
        gt_keypoints = instances_per_image.gt_keypoints.tensor

        # Get predicted heatmaps for these instances:
        # pred_heatmaps: shape (n_i, K, S, S)
        pred_heatmaps = pred_logits_orig[offset: offset + n_i]
        offset += n_i

        # Reshape heatmaps to (n_i, K, S*S) and compute softmax:
        pred_heatmaps_flat = pred_heatmaps.view(n_i, K, S * S)
        probs = F.softmax(pred_heatmaps_flat, dim=2)  # shape (n_i, K, S*S)

        # Compute expected coordinates in the heatmap coordinate system:
        exp_x = (probs * grid_x).sum(dim=2)  # (n_i, K)
        exp_y = (probs * grid_y).sum(dim=2)  # (n_i, K)

        # Transform expected heatmap coordinates to image coordinates using proposal boxes.
        # boxes: (n_i, 4) => extract x1, y1, x2, y2, and compute width and height.
        x1 = boxes[:, 0].unsqueeze(1)  # (n_i, 1)
        y1 = boxes[:, 1].unsqueeze(1)  # (n_i, 1)
        x2 = boxes[:, 2].unsqueeze(1)
        y2 = boxes[:, 3].unsqueeze(1)
        widths = (x2 - x1)
        heights = (y2 - y1)

        # The predicted coordinates in image space:
        pred_x = x1 + (exp_x + 0.5) * (widths / S)  # (n_i, K)
        pred_y = y1 + (exp_y + 0.5) * (heights / S)  # (n_i, K)
        pred_coords = torch.stack([pred_x, pred_y], dim=2)  # (n_i, K, 2)

        # Convert predicted and ground-truth keypoint coordinates to normalized coordinates.
        range_img = range_imgs[i]  # shape (H_img, W_img)
        H_img, W_img = range_img.shape
        norm_pred_coords = normalize_coords(pred_coords, H_img, W_img)  # (n_i, K, 2)
        norm_gt_coords = normalize_coords(gt_keypoints, H_img, W_img)  # (n_i, K, 2)

        # Reshape normalized coordinates to use with grid_sample.
        # grid_sample expects input of shape (N, H_out, W_out, 2); here H_out=W_out=1 for each keypoint.
        n_keypoints = n_i * K
        grid_pred = norm_pred_coords.view(n_keypoints, 1, 1, 2)
        grid_gt = norm_gt_coords.view(n_keypoints, 1, 1, 2)

        # Prepare the range image for grid_sample:
        # Expand the range image to shape (1, 1, H_img, W_img) and replicate for each keypoint.
        range_img_exp = range_img.unsqueeze(0).unsqueeze(0)  # (1, 1, H_img, W_img)
        range_img_tile = range_img_exp.expand(n_keypoints, -1, -1, -1)  # (n_keypoints, 1, H_img, W_img)

        # Sample range values at the predicted and gt coordinates:
        sampled_pred = F.grid_sample(range_img_tile, grid_pred, align_corners=True)
        sampled_gt = F.grid_sample(range_img_tile, grid_gt, align_corners=True)
        # Reshape to (n_i, K)
        sampled_pred = sampled_pred.view(n_i, K)
        sampled_gt = sampled_gt.view(n_i, K)

        # Compute the L1 loss for this image (mean over all keypoints)
        range_loss_img = F.l1_loss(sampled_pred, sampled_gt, reduction="mean")
        range_loss_total += range_loss_img * (n_i * K)
        total_keypoints += n_i * K

        heatmap_probs = F.softmax(pred_keypoint_logits.view(N * K, S * S), dim=1)
        heatmap_entropy = -(heatmap_probs * torch.log(heatmap_probs + 1e-6)).sum(dim=1)
        heatmap_reg_loss = heatmap_entropy.mean()

    if total_keypoints > 0:
        range_loss = range_loss_total / total_keypoints
    else:
        range_loss = torch.tensor(0.0, device=pred_keypoint_logits.device)

    return keypoint_loss, range_loss, heatmap_reg_loss


def keypoint_rcnn_inference(pred_keypoint_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

    Returns:
        None. Each element in pred_instances will contain extra "pred_keypoints" and
            "pred_keypoint_heatmaps" fields. "pred_keypoints" is a tensor of shape
            (#instance, K, 3) where the last dimension corresponds to (x, y, score).
            The scores are larger than 0. "pred_keypoint_heatmaps" contains the raw
            keypoint logits as passed to this function.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    pred_keypoint_logits = pred_keypoint_logits.detach()
    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits, bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)
    heatmap_results = pred_keypoint_logits.split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, heatmap_results_per_image, instances_per_image in zip(
        keypoint_results, heatmap_results, pred_instances
    ):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        # heatmap_results_per_image is (num instances)x(num keypoints)x(side)x(side)
        instances_per_image.pred_keypoints = keypoint_results_per_image
        instances_per_image.pred_keypoint_heatmaps = heatmap_results_per_image


class BaseKeypointRCNNHead(nn.Module):
    """
    Implement the basic Keypoint R-CNN losses and inference logic described in
    Sec. 5 of :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(self, *, num_keypoints, loss_weight=1.0, loss_normalizer=1.0):
        """
        NOTE: this interface is experimental.

        Args:
            num_keypoints (int): number of keypoints to predict
            loss_weight (float): weight to multiple on the keypoint loss
            loss_normalizer (float or str):
                If float, divide the loss by `loss_normalizer * #images`.
                If 'visible', the loss is normalized by the total number of
                visible keypoints across images.
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.loss_weight = loss_weight
        assert loss_normalizer == "visible" or isinstance(loss_normalizer, float), loss_normalizer
        self.loss_normalizer = loss_normalizer

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            "loss_weight": cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT,
            "num_keypoints": cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS,
        }
        normalize_by_visible = (
            cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS
        )  # noqa
        if not normalize_by_visible:
            batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
            positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
            ret["loss_normalizer"] = (
                ret["num_keypoints"] * batch_size_per_image * positive_sample_fraction
            )
        else:
            ret["loss_normalizer"] = "visible"
        return ret

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input 4D region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x)
        if self.training:
            num_images = len(instances)
            normalizer = (
                None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
            )
            return {
                "loss_keypoint": keypoint_rcnn_loss(x, instances, normalizer=normalizer)
                * self.loss_weight
            }
        else:
            keypoint_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from regional input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHead(BaseKeypointRCNNHead, nn.Sequential):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    It is described in Sec. 5 of :paper:`Mask R-CNN`.
    """

    @configurable
    def __init__(self, input_shape, *, num_keypoints, conv_dims, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
        """
        super().__init__(num_keypoints=num_keypoints, **kwargs)

        # default up_scale to 2.0 (this can be made an option)
        up_scale = 2.0
        in_channels = input_shape.channels

        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.add_module("conv_fcn_relu{}".format(idx), nn.ReLU())
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["input_shape"] = input_shape
        ret["conv_dims"] = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x



@ROI_KEYPOINT_HEAD_REGISTRY.register()
class RangeKRCNNConvDeconvUpsampleHead(KRCNNConvDeconvUpsampleHead, nn.Sequential):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    It is described in Sec. 5 of :paper:`Mask R-CNN`.

    Also has a range-based loss function.
    """

    def forward(self, x, instances: List[Instances], range_imgs):
        """
        Args:
            x: input 4D region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x)
        if self.training:
            num_images = len(instances)
            normalizer = (
                None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
            )
            keypoint_loss, range_loss, heatmap_loss = range_keypoint_rcnn_loss(x, instances, normalizer=normalizer, range_imgs=range_imgs)
            return {
                "loss_keypoint": keypoint_loss
                * self.loss_weight,
                "range_loss": range_loss * 10,
                "heatmap_loss": heatmap_loss * 1
            }
        else:
            keypoint_rcnn_inference(x, instances)
            return instances