import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from infer import evaluate_map_v2, infer_from_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader
from detection.transform import GeneralizedRCNNTransform, resize_boxes
import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator
import cv2
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Collate function
def collate_function(data):
    return tuple(zip(*data))


import cv2
import numpy as np

def save_proposals(proposals, ims, epoch_dir, filename):
    # Convert PyTorch tensor to NumPy
    if isinstance(ims, torch.Tensor):
        ims = ims.cpu().numpy()  # Move to CPU and convert to NumPy
        ims = ims.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
        ims = (ims * 255).astype(np.uint8)  # Convert to uint8 (0-255)

    # Ensure it's in BGR format for OpenCV
    if ims.shape[-1] == 3:  # If RGB, convert to BGR
        ims = cv2.cvtColor(ims, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes
    for proposal in proposals:
        x1, y1, x2, y2 = map(int, proposal.cpu().numpy())  # Ensure integer coordinates
        print(f"Drawing bounding box: ({x1}, {y1}) -> ({x2}, {y2})")  # Debugging
        cv2.rectangle(ims, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box

    # Save image
    cv2.imwrite(f"{epoch_dir}/{filename}.png", ims)




def save_objectness_heatmap(objectness_scores, base_dir=""):
    num_levels = len(objectness_scores[0])  
    num_imgs = len(objectness_scores)

    os.makedirs(base_dir, exist_ok=True)

    for level in range(num_levels):
        level_dir = os.path.join(base_dir, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)

        for img_idx in range(num_imgs):
            heatmap = objectness_scores[img_idx][level][0]  # Select first anchor channel

            plt.figure(figsize=(5, 5))
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            plt.title(f"Level {level} - Image {img_idx}")
            plt.axis("off")

            img_path = os.path.join(level_dir, f"img_{img_idx}_heatmap.png")
            print(f"Saving heatmap to {img_path}")
            plt.savefig(img_path)
            plt.close()

def train(args, output_dir='output'):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
    
    print(config)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    # Load dataset
    st = SceneTextDataset('train', root_dir=dataset_config['root_dir'])
    train_indices = np.random.choice(len(st), len(st) - 4, replace=False)
    test_indices = np.setdiff1d(np.arange(len(st)), train_indices)
    
    train_data = torch.utils.data.Subset(st, train_indices)
    test_data = torch.utils.data.Subset(st, test_indices)
    
    train_dataset = DataLoader(train_data, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_function)
    test_dataset = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_function)
    
    # Model setup
    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained_backbone=True, min_size=600, max_size=1000, thresholded_thetas=True, num_thetas=10)
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'],
        thresholded_thetas=True,
        num_thetas=10
    )
    
    faster_rcnn_model.to(device)
    os.makedirs(train_config['task_name'], exist_ok=True)
    
    optimizer = torch.optim.SGD(
        lr=1E-4, params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        weight_decay=5E-5, momentum=0.9
    )
    
    num_epochs = train_config['num_epochs']
    num_epochs = 50
    faster_rcnn_model.train()
    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        

        epoch_dir = os.path.join(output_dir, 'heatmap_frames', f'epoch_{i}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        os.makedirs(os.path.join(output_dir, f'epoch_{i}'), exist_ok=True)
        os.makedirs(os.path.join(epoch_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(epoch_dir, 'train'), exist_ok=True)      
        #get objectness heatmap
        faster_rcnn_model.eval()
        epoch_dir = os.path.join(output_dir, 'heatmap_frames', f'epoch_{i}')
        for ims, targets, _ in test_dataset:
            images = [im.float().to(device) for im in ims]
            with torch.no_grad():
                transform = GeneralizedRCNNTransform(min_size=800, max_size=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
                image_list, _ = transform(images)

                backbone_features = faster_rcnn_model.backbone(image_list.tensors)
                targets = None
                # Extract RPN objectness scores
                features_list = list(backbone_features.values())  # Convert OrderedDict to list of tensors
                proposals, proposal_losses, _ = faster_rcnn_model.rpn(image_list, backbone_features, targets)
                
                for idx, proposal in enumerate(proposals):
                    proposals[idx] = resize_boxes(proposal, image_list.image_sizes[idx], ims[idx].shape[-2:])
                    save_proposals(proposals[idx], ims[idx], epoch_dir, f'test_{idx}')
                
                rpn_logits, _ = faster_rcnn_model.rpn.head(features_list)
                # Convert logits to probabilities
                objectness_scores = [logit.sigmoid().cpu().numpy() for logit in rpn_logits]
                save_objectness_heatmap(objectness_scores, epoch_dir)
                
    
        faster_rcnn_model.train()
        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            images = [im.float().to(device) for im in ims]
            
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                target['thetas'] = target['thetas'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = sum(batch_losses.values())
            
            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())
            
            loss.backward()
            optimizer.step()
        
        print(f'Finished epoch {i}')
        torch.save(
            faster_rcnn_model.state_dict(),
            os.path.join(output_dir, f'tv_frcnn_r50fpn_{train_config["ckpt_name"]}')
        )
        
        loss_output = (
            f"RPN Classification Loss: {np.mean(rpn_classification_losses):.4f} | "
            f"RPN Localization Loss: {np.mean(rpn_localization_losses):.4f} | "
            f"FRCNN Classification Loss: {np.mean(frcnn_classification_losses):.4f} | "
            f"FRCNN Localization Loss: {np.mean(frcnn_localization_losses):.4f}"
        )
        print(loss_output)
    
    print('Done Training...')

class Args:
    config_path = 'config/st.yaml'

args = Args()
train(args, output_dir='output')