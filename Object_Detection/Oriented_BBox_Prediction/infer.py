import torch
import numpy as np
import cv2
import torchvision
import argparse
import random
import os
import yaml
from tqdm import tqdm
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score], ...],
    #       'car' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2], ...],
    #       'car' : [[x1, y1, x2, y2], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    # average precisions for ALL classes
    aps = []
    pre_all = []
    rec_all = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)
    
        pre_all.append(precisions)
        rec_all.append(recalls)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps, pre_all, rec_all


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    st = SceneTextDataset('test', root_dir=dataset_config['root_dir'])
    test_dataset = DataLoader(st, batch_size=1, shuffle=False)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                            min_size=600,
                                                            max_size=1000,
                                                            box_score_thresh=0.7,
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNFastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'])

    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                                'tv_frcnn_r50fpn_' + train_config['ckpt_name']),
                                                    map_location=device))

    return faster_rcnn_model, st, test_dataset


def infer(args):
    output_dir = 'samples_tv_r50fpn'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)

    for sample_count in tqdm(range(10)):
        random_idx = random.randint(0, len(voc))
        im, target, fname = voc[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = voc.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=voc.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite('{}/output_frcnn_gt_{}.png'.format(output_dir, sample_count), gt_im)

        # Getting predictions from trained model
        frcnn_output = faster_rcnn_model(im, None)[0]
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        thetas = frcnn_output['thetas']
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Saving images with predicted boxes
        for idx, (box,theta) in enumerate(boxes, thetas):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # rotate the box
            theta = thetas[idx].detach().cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            rect = ((center_x, center_y), (width, height), np.degrees(theta))

            # Convert rotated rect to bounding box
            box = cv2.boxPoints(rect)
            # convert to int
            box = np.int64(box)
            x1, y1 = np.min(box, axis=0)
            x2, y2 = np.max(box, axis=0)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(voc.idx2label[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite('{}/output_frcnn_{}.jpg'.format(output_dir, sample_count), im)


def evaluate_map(args):
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)
    gts = []
    preds = []
    for im, target, fname in tqdm(test_dataset):
        im_name = fname
        im = im.float().to(device)
        target_boxes = target['bboxes'].float().to(device)[0]
        target_thetas = target['thetas'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        frcnn_output = faster_rcnn_model(im, None)[0]

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        thetas = frcnn_output['thetas'] 

        pred_boxes = {}
        gt_boxes = {}
        for label_name in voc.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []

        for idx, box, theta in enumerate(zip(boxes, thetas)):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            theta = thetas[idx].detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            rect = ((center_x, center_y), (width, height), np.degrees(theta))

            # Convert rotated rect to bounding box
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            x1, y1 = np.min(box, axis=0)
            x2, y2 = np.max(box, axis=0)
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box, theta in enumerate(zip(target_boxes, target_thetas)):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            theta = thetas[idx].detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            rect = ((center_x, center_y), (width, height), np.degrees(theta))

            # Convert rotated rect to bounding box
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            x1, y1 = np.min(box, axis=0)
            x2, y2 = np.max(box, axis=0)
            gt_boxes[label_name].append([x1, y1, x2, y2])

        gts.append(gt_boxes)
        preds.append(pred_boxes)

    thresholds = [np.linspace(0.5, 0.95, 10)]
    for threshold in thresholds:
        mean_ap, all_ap = compute_map(preds, gts, iou_threshold=threshold, method='interp')
        print('Mean Average Precision at IoU threshold {} : {:.4f}'.format(threshold, mean_ap))
    
        
def evaluate_map_v2(model, dataset, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    faster_rcnn_model = model
    test_dataset = dataset
    gts = []
    preds = []
    for ims, targets, fnames in tqdm(test_dataset):
        for im, target, fname in zip(ims, targets, fnames):
            im = im.float().to(device)
            im_name = fname
            target_boxes = target['bboxes'].float().to(device)
            target_thetas = target['thetas'].float().to(device)
            target_labels = target['labels'].long().to(device)
            frcnn_output = model(im.unsqueeze(0))[0]

            boxes = frcnn_output['boxes']
            labels = frcnn_output['labels']
            scores = frcnn_output['scores']
            thetas = frcnn_output['thetas'] 

            pred_boxes = {}
            gt_boxes = {}

            lnames = set()

            for idx, (box, theta) in enumerate(zip(boxes, thetas)):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                theta = theta.detach().cpu().numpy()
                label = labels[idx].detach().cpu().item()
                score = scores[idx].detach().cpu().item()
                label_name = str(label)
                lnames.add(label_name)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                rect = ((center_x, center_y), (width, height), theta.item())
                # Convert rotated rect to bounding box
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                x1, y1 = np.min(box, axis=0)
                x2, y2 = np.max(box, axis=0)
                if label_name not in pred_boxes:
                    pred_boxes[label_name] = []
                pred_boxes[label_name].append([x1, y1, x2, y2, score])
            for idx, (box, theta) in enumerate(zip(target_boxes, target_thetas)):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                theta = theta.detach().cpu().numpy()
                # convert to float
                
                label = target_labels[idx].detach().cpu().item()
                label_name = str(label)
                lnames.add(label_name)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                rect = ((center_x, center_y), (width, height), theta.item())
                # Convert rotated rect to bounding box
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                x1, y1 = np.min(box, axis=0)
                x2, y2 = np.max(box, axis=0)
                if label_name not in gt_boxes:
                    gt_boxes[label_name] = []
                gt_boxes[label_name].append([x1, y1, x2, y2])

            gts.append(gt_boxes)
            preds.append(pred_boxes)
    
    thresholds = np.linspace(0.05, 0.95, 10) 
    retmean = []
    pre_all = []
    rec_all = []
    for threshold in thresholds:
        mean_ap, all_aps, precision, recall = compute_map(preds, gts, iou_threshold=threshold, method='interp')
        print('Mean Average Precision at IoU threshold {} : {:.4f}'.format(threshold, mean_ap))
        retmean.append(mean_ap)
        rec_all.append(recall)
        pre_all.append(precision)
    return retmean, pre_all, rec_all

def infer_from_model(model, dataset, output_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    sample_count = 0
    for ims, targetss, fnames in dataset:
        if sample_count >= 10:
            break  # Limit to 10 samples

        iml = [im.float().to(device) for im in ims]

        for targets in targetss:
            targets['boxes'] = targets['bboxes'].float().to(device)
            del targets['bboxes']
            targets['labels'] = targets['labels'].long().to(device)
            targets['thetas'] = targets['thetas'].float().to(device)

        for idy, target in enumerate(targetss):   
            # Load image
            gt_im = cv2.imread(fnames[idy])
            gt_im_copy = gt_im.copy()
            gt_im_copy_unrotated = gt_im.copy()
            # Draw ground truth boxes
            for idx, (box,theta) in enumerate(zip(target['boxes'], target['thetas'])):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                rect = ((center_x, center_y), (width, height), theta.item())
                rotated_box = cv2.boxPoints(rect).astype(int)
                cv2.polylines(gt_im_copy, [rotated_box], isClosed=True, color=[0, 255, 0], thickness=2)
            cv2.imwrite(f'{output_dir}/output_frcnn_gt_{sample_count}.png', gt_im_copy) 

            # Get predictions
            im = iml[idy]
            frcnn_output = model(im.unsqueeze(0), verbose=True)[0]  # Ensure input is batched
            boxes, labels, scores, thetas = frcnn_output['boxes'], frcnn_output['labels'], frcnn_output['scores'], frcnn_output['thetas']

            im_pred = cv2.imread(fnames[idy])
            im_copy = im_pred.copy()

            # Draw predicted boxes
            for idx, (box, theta) in enumerate(zip(boxes, thetas)):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Convert bounding box to rotated rectangle
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                rect = ((center_x, center_y), (width, height), theta.item())
                rotated_box = cv2.boxPoints(rect).astype(int)
                cv2.polylines(im_copy, [rotated_box], isClosed=True, color=[0, 0, 255], thickness=2)
            cv2.imwrite(f'{output_dir}/output_frcnn_{sample_count}.jpg', im_copy)
            
            # draw unrotated predictions
            im_pred = cv2.imread(fnames[idy])
            im_copy_2 = im_pred.copy()
            boxes, thetas = frcnn_output['boxes_un'], frcnn_output['thetas_un']
            for idx, (box, theta) in enumerate(zip(boxes, thetas)):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Convert bounding box to rotated rectangle
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                rect = ((center_x, center_y), (width, height), theta.item())
                rotated_box = cv2.boxPoints(rect).astype(int)
                cv2.polylines(im_copy_2, [rotated_box], isClosed=True, color=[0, 0, 255], thickness=2)
            cv2.imwrite(f'{output_dir}/output_frcnn_unrotated_{sample_count}.jpg', im_copy_2)
            
            print(f'Processed sample {sample_count}, saved to {output_dir}')
            sample_count += 1
            if sample_count >= 10:
                return  # Stop inference after 10 samples

            
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inference using torchvision code faster rcnn')
    parser.add_argument('--config', dest='config_path',
                        default='config/st.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    args = parser.parse_args()
    
    if args.infer_samples:
        infer(args)
    else:
        print('Not Inferring for samples as `infer_samples` argument is False')

    if args.evaluate:
        evaluate_map(args)
    else:
        print('Not Evaluating as `evaluate` argument is False')