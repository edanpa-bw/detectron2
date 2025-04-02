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
    Compute the keypoint loss (via cross entropy) as well as an additional range loss.

    Arguments:
        pred_keypoint_logits (Tensor): Tensor of shape (N, K, S, S) where:
            - N is the total number of instances in the batch.
            - K is the number of keypoints.
            - S is the side length of the keypoint heatmap.
            The values are spatial logits.
        instances (list[Instances]): A list (length = batch size) of Instances objects.
            Each Instances should contain:
              - `gt_keypoints`: a Keypoint object with a .coordinates tensor of shape (n, K, 2).
              - `proposal_boxes`: a Boxes object with a tensor attribute of shape (n, 4).
        normalizer (float): Normalization factor. If None, the loss is normalized by the number of visible keypoints.
        range_imgs (list[Tensor]): A list (length = batch size) of range image tensors, each of shape (H_img, W_img),
            corresponding to the input images.

    Returns:
        dict[str, Tensor]: A dictionary with keys "loss_keypoint" and "loss_range".
    """
    # --- Compute the standard keypoint loss ---
    heatmaps = []
    valid = []
    keypoint_side_len = pred_keypoint_logits.shape[2]

    # Compute target heatmaps from the ground-truth keypoints
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
        return zero_loss, zero_loss

    # Save the original shape and a copy of the logits for range loss computation.
    N, K, S, _ = pred_keypoint_logits.shape  # S is the keypoint heatmap size.
    pred_logits_orig = pred_keypoint_logits  # (N, K, S, S)
    pred_keypoint_logits_flat = pred_keypoint_logits.view(N * K, S * S)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits_flat[valid], keypoint_targets[valid], reduction="sum"
    )

    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    # --- Compute the range loss ---
    #
    # For each instance, we compute a differentiable estimate of its keypoint coordinates by applying
    # a soft-argmax over the heatmap (which is in the coordinate system of the proposal box). We then
    # sample the corresponding range image at both the predicted coordinate and the ground truth coordinate,
    # and compute an L1 loss between them.

    # Helper: Soft-argmax over one keypoint heatmap given the proposal box.
    def soft_argmax(heatmap, box):
        # heatmap: (S, S)
        S_local = heatmap.shape[0]
        heatmap_flat = heatmap.view(-1)
        prob = F.softmax(heatmap_flat, dim=0)
        prob = prob.view(S_local, S_local)
        device = heatmap.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S_local, device=device, dtype=torch.float32),
            torch.arange(S_local, device=device, dtype=torch.float32),
            indexing="ij",
        )
        exp_x = (prob * grid_x).sum()
        exp_y = (prob * grid_y).sum()
        # Transform heatmap coordinates to image coordinates based on the proposal box.
        # The proposal box (box) is given as [x1, y1, x2, y2]
        x1, y1, x2, y2 = box  # each a scalar
        box_width = x2 - x1
        box_height = y2 - y1
        pred_x = x1 + (exp_x + 0.5) * (box_width / S_local)
        pred_y = y1 + (exp_y + 0.5) * (box_height / S_local)
        return pred_x, pred_y

    # Helper: Sample a range value from a range image at a given (x, y) coordinate.
    def sample_range_value(range_img, coord):
        # range_img: (H_img, W_img)
        # coord: tuple (x, y) in image coordinates.
        H_img, W_img = range_img.shape
        x, y = coord
        norm_x = (x / (W_img - 1)) * 2 - 1  # Normalize to [-1, 1]
        norm_y = (y / (H_img - 1)) * 2 - 1  # Normalize to [-1, 1]
        # grid_sample expects a grid of shape (N, H_out, W_out, 2); here we sample a single point.
        grid = torch.tensor([[[[norm_x, norm_y]]]], device=range_img.device, dtype=torch.float32)
        range_img_unsq = range_img.unsqueeze(0).unsqueeze(0)  # (1, 1, H_img, W_img)
        sampled = F.grid_sample(range_img_unsq, grid, align_corners=True)
        return sampled.squeeze()  # scalar

    range_loss_total = 0.0
    count = 0
    offset = 0  # To index into pred_logits_orig (which is a concatenation over images)

    # Iterate over images in the batch
    for i, instances_per_image in enumerate(instances):
        if len(instances_per_image) == 0:
            continue
        range_img = range_imgs[i]  # Tensor of shape (H_img, W_img)
        boxes = instances_per_image.proposal_boxes.tensor  # (num_instances, 4)
        # Assume gt_keypoints.coordinates is available and of shape (num_instances, K, 2)
        gt_keypoints = instances_per_image.gt_keypoints.tensor[:, :, :2]  # (num_instances, K, 2)
        num_instances = boxes.shape[0]
        for j in range(num_instances):
            box = boxes[j]  # (4,)
            instance_heatmaps = pred_logits_orig[offset + j]  # (K, S, S)
            for k in range(K):
                heatmap = instance_heatmaps[k]  # (S, S)
                pred_x, pred_y = soft_argmax(heatmap, box)
                # Ground-truth coordinate for keypoint k for this instance:
                gt_coord = gt_keypoints[j, k]  # (2,)
                pred_range = sample_range_value(range_img, (pred_x, pred_y))
                gt_range = sample_range_value(range_img, (gt_coord[0], gt_coord[1]))
                range_loss_total += F.l1_loss(pred_range, gt_range, reduction="mean")
                count += 1
        offset += num_instances

    if count > 0:
        range_loss = range_loss_total / count
    else:
        range_loss = torch.tensor(0.0, device=pred_keypoint_logits.device)

    return keypoint_loss, range_loss


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


def sample_range_channel(range_img, keypoints):
    """
    Samples the range channel from a 2D tensor image at given keypoint locations.

    Args:
        range_img (Tensor): A tensor of shape (H, W) representing the range image.
        keypoints (Tensor): A tensor of shape (N, 2) with (x, y) coordinates.

    Returns:
        Tensor: The sampled range values for each keypoint.
    """
    H, W = range_img.shape[-2:]
    norm_keypoints = keypoints.clone().float()
    norm_keypoints[:, 0] = (norm_keypoints[:, 0] / (W - 1)) * 2 - 1  # Normalize x
    norm_keypoints[:, 1] = (norm_keypoints[:, 1] / (H - 1)) * 2 - 1  # Normalize y
    grid = norm_keypoints.view(1, 1, -1, 2)  # (1, 1, N, 2)
    range_img = range_img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    sampled = F.grid_sample(range_img, grid, align_corners=True)
    return sampled.view(-1)


def sample_range_channel(range_img, keypoints):
    """
    Samples the range channel from a 2D tensor at given keypoint locations.

    Args:
        range_img (Tensor): Tensor of shape (H, W) representing the range image.
        keypoints (Tensor): Tensor of shape (N, 2) with (x, y) coordinates.

    Returns:
        Tensor: The sampled range values for each keypoint.
    """
    H, W = range_img.shape[-2:]
    norm_keypoints = keypoints.clone().float()
    norm_keypoints[:, 0] = (norm_keypoints[:, 0] / (W - 1)) * 2 - 1  # Normalize x to [-1, 1]
    norm_keypoints[:, 1] = (norm_keypoints[:, 1] / (H - 1)) * 2 - 1  # Normalize y to [-1, 1]
    grid = norm_keypoints.view(1, 1, -1, 2)  # (1, 1, N, 2)
    range_img = range_img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    sampled = F.grid_sample(range_img, grid, align_corners=True)
    return sampled.view(-1)


def soft_argmax_2d(heatmaps):
    """
    Computes differentiable keypoint coordinates from heatmaps using a soft-argmax.

    Args:
        heatmaps (Tensor): Tensor of shape (B, K, H, W) representing the predicted keypoint heatmaps.

    Returns:
        Tensor: Predicted coordinates of shape (B, K, 2) where the last dimension is (x, y).
    """
    B, K, H, W = heatmaps.shape
    # Reshape and apply softmax over spatial dimensions
    heatmaps_reshaped = heatmaps.view(B, K, -1)
    softmax_heatmaps = F.softmax(heatmaps_reshaped, dim=-1)
    softmax_heatmaps = softmax_heatmaps.view(B, K, H, W)

    # Create coordinate grids
    device = heatmaps.device
    x_coords = torch.linspace(0, W - 1, W, device=device)
    y_coords = torch.linspace(0, H - 1, H, device=device)
    # Use indexing='ij' so that grid_y corresponds to rows and grid_x to columns
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)

    # Compute expected coordinates by summing over spatial dimensions weighted by the softmax probabilities
    expected_x = (softmax_heatmaps * grid_x).sum(dim=(-2, -1))
    expected_y = (softmax_heatmaps * grid_y).sum(dim=(-2, -1))
    coords = torch.stack([expected_x, expected_y], dim=-1)  # (B, K, 2)
    return coords


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
            keypoint_loss, range_loss = range_keypoint_rcnn_loss(x, instances, normalizer=normalizer, range_imgs=range_imgs)
            return {
                "loss_keypoint": keypoint_loss
                * self.loss_weight,
                "range_loss": range_loss * self.loss_weight
            }
        else:
            keypoint_rcnn_inference(x, instances)
            return instances