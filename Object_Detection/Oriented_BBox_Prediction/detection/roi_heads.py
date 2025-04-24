from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align

from . import _utils as det_utils


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, theta_logits=None, regression_theta_logits=None):
    """
    Computes the loss for Faster R-CNN with oriented bounding boxes.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[Tensor])
        regression_targets (list[Tensor])
        targets (list[dict]): Each dictionary contains 'thetas' (Tensor of angles)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    if theta_logits is not None:
        box_regression = box_regression.reshape(N, -1, 4)
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        theta_logits = theta_logits.reshape(N, -1, len(theta_logits[0]))
        regression_theta_logits = torch.cat(regression_theta_logits, dim=0)
        theta_loss = F.cross_entropy(theta_logits[sampled_pos_inds_subset, labels_pos], regression_theta_logits[sampled_pos_inds_subset])
        box_loss = box_loss / labels.numel()
        theta_loss = theta_loss / labels.numel()
        return classification_loss, box_loss, theta_loss
    box_regression = box_regression.reshape(N, -1, 5)
    
    
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        thresholded_thetas = None,
        num_thetas = 1,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0, 1.0) #use only 1.0 for theta if thresholded thetas
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        self.oriented_box_coder = det_utils.OrientedBoxCoder(weights=bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.num_thetas = num_thetas
        self.thresholded_thetas = thresholded_thetas

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_thetas = [t["thetas"].to(dtype) for t in targets]  # Extract thetas

        # Append ground-truth bboxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # Get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # Sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        matched_gt_thetas = []  # Store matched thetas
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            gt_thetas_in_image = gt_thetas[img_id]

            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
                gt_thetas_in_image = torch.zeros((1, 1), dtype=dtype, device=device)

            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            matched_gt_thetas.append(gt_thetas_in_image[matched_idxs[img_id]])
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        # Convert matched_gt_thetas from tuple to list and concatenate
        if self.thresholded_thetas:
            matched_gt_thetas = [theta.view(-1, 1) for theta in matched_gt_thetas]
            # assign correct bin to each theta
            theta_target_logits = []
            for theta in matched_gt_thetas:
                theta_target_logits.append(torch.zeros((theta.shape[0], self.num_thetas), dtype=dtype, device=device))
                for i in range(self.num_thetas):
                    theta_target_logits[-1][:, i] = (theta[:, 0] == i).float()
            return proposals, matched_idxs, labels, regression_targets, theta_target_logits
                
        else:
            matched_gt_thetas = [theta.view(-1, 1) for theta in matched_gt_thetas]
            regression_targets = [torch.cat([reg, theta], dim=1) for reg, theta in zip(regression_targets, matched_gt_thetas)]

            return proposals, matched_idxs, labels, regression_targets




    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        #print(f"Class logits shape: {class_logits.shape}, Box regression shape: {box_regression.shape}, Proposals shape: {len(proposals)}, Image shapes: {len(image_shapes)}")
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes_un, pred_thetas_un = self.oriented_box_coder.decode_unchange(box_regression, proposals)
        pred_boxes, pred_thetas = self.oriented_box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_thetas_list = pred_thetas.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_boxes_un_list = pred_boxes_un.split(boxes_per_image, 0)
        pred_thetas_un_list = pred_thetas_un.split(boxes_per_image, 0)
        all_boxes = []
        all_scores = []
        all_labels = []
        all_thetas = []
        all_boxes_un = []
        all_thetas_un = []
        for boxes, scores, image_shape, thetas, boxes_un, thetas_un in zip(pred_boxes_list, pred_scores_list, image_shapes, pred_thetas_list, pred_boxes_un_list, pred_thetas_un_list):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            
            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            #print(f" initial Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Labels shape: {labels.shape}, Thetas shape: {thetas.shape}")
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            thetas = thetas[:, 1:]
            boxes_un = boxes_un[:, 1:]
            thetas_un = thetas_un[:, 1:]
            

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            thetas = thetas.reshape(-1)
            boxes_un = boxes_un.reshape(-1, 4)
            thetas_un = thetas_un.reshape(-1)
            
            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            thetas = thetas[inds]
            boxes_un, thetas_un = boxes_un[inds], thetas_un[inds]
            
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            thetas = thetas[keep]
            boxes_un, thetas_un = boxes_un[keep], thetas_un[keep]
        

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            thetas = thetas[keep]
            boxes_un, thetas_un = boxes_un[keep], thetas_un[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_thetas.append(thetas)
            all_boxes_un.append(boxes_un)
            all_thetas_un.append(thetas_un)
            #print(f"Boxes shape: {boxes.shape}, Scores shape: {scores.shape}, Labels shape: {labels.shape}, Thetas shape: {thetas.shape}")
        return all_boxes, all_scores, all_labels, all_thetas, all_boxes_un, all_thetas_un


    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if not t["thetas"].dtype in floating_point_types:
                    raise TypeError(f"fuck me ig")
        if self.training:
            if self.thresholded_thetas:
                proposals, matched_idxs, labels, regression_targets, regression_theta_logits = self.select_training_samples(proposals, targets)
            else:
                proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None


        box_features = self.box_roi_pool(features, proposals, image_shapes)

        box_features = self.box_head(box_features)
        if self.thresholded_thetas:
            class_logits, box_regression, theta_logits, box_regression_full = self.box_predictor(box_features)
        else:
            class_logits, box_regression = self.box_predictor(box_features)
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            
            if self.thresholded_thetas:
                loss_classifier, loss_box_reg, loss_theta_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets, theta_logits, regression_theta_logits)
                losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_theta_reg": loss_theta_reg}
            else:
                loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
                losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            if self.thresholded_thetas:
                boxes, scores, labels, thetas, boxes_un, thetas_un  = self.postprocess_detections(class_logits, box_regression_full, proposals, image_shapes)
            else:
                boxes, scores, labels, thetas, boxes_un, thetas_un = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "thetas": thetas[i],
                        "boxes_un": boxes_un[i],
                        "thetas_un": thetas_un[i],
                    }
                )

        return result, losses