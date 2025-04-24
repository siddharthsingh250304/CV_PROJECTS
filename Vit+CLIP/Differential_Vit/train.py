# -*- coding: utf-8 -*-
"""
tune_diff_vit.py

Sequentially tune hyperparameters for a Vision Transformer (ViT)
modified with Differential Attention (Diff-ViT) on CIFAR-10.

Tuning Stages:
1. Tune Patch Size
2. Tune Model Parameters (using best patch size)
3. Tune Data Augmentation (using best model config)
4. Tune Positional Embedding (using best config + aug)

Saves the best model state_dict and config FOUND AT EACH STAGE,
plus the overall best Diff-ViT model state_dict and config across all runs.

Based on the paper: https://arxiv.org/abs/2410.05258
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
import os
import json

# Global Settings and Constants
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
RESULTS_DIR_BASE = "results_diff_vit_tuned" # Changed results dir name
PLOTS_DIR = os.path.join(RESULTS_DIR_BASE, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR_BASE, "models")
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(MODELS_DIR, exist_ok=True)

NUM_EPOCHS_TUNE = 20         # Epochs for ALL tuning stages
BEST_MODEL_FILENAME = os.path.join(MODELS_DIR, "best_diff_vit_overall_tuned.pth") # Overall best Diff-ViT
BEST_CONFIG_FILENAME = os.path.join(RESULTS_DIR_BASE, "best_config_diff_vit_overall_tuned.json") # Overall best config
BEST_PLOT_FILENAME = os.path.join(PLOTS_DIR, "best_train_curves_diff_vit_overall_tuned.png") # Overall best curves
TUNING_LOG_FILENAME = os.path.join(RESULTS_DIR_BASE, "tuning_log_diff_vit.txt")
DATA_DIR = './data_cifar'
BASE_BATCH_SIZE = 128


# --- Model Definitions (Copied from previous diff_vit script) ---
# Includes PatchEmbed, MLP, Positional Embeddings, Standard MHA (unused but kept for potential comparison later),
# MultiHeadDifferentialAttention, TransformerEncoderLayer (with selection flag), VisionTransformer (with selection flag)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(p):
        return [p / np.power(10000, 2 * (h // 2) / d_hid) for h in range(d_hid)]
    t = np.array([get_position_angle_vec(p) for p in range(n_position)])
    t[:, 0::2] = np.sin(t[:, 0::2])
    t[:, 1::2] = np.cos(t[:, 1::2])
    return torch.FloatTensor(t).unsqueeze(0)

class MultiHeadAttention(nn.Module): # Standard MHA - unused here but kept for completeness
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., layer_idx=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., layer_idx=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.wk = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.wv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.lambda_param = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(self.head_dim * 2)
        l = layer_idx
        self.lambda_init = 0.8 - 0.6 * np.exp(-0.3 * (l - 1))
        # print(f"  Layer {l}: DiffAttn lambda_init={self.lambda_init:.4f}") # Optional: verbose init
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        q = self.wq(x).reshape(B, N, H, 2 * D).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, H, 2 * D).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, H, 2 * D).permute(0, 2, 1, 3)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        attn1_scores = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn2_scores = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1_scores.softmax(dim=-1)
        attn2 = attn2_scores.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)
        lambda_val = self.lambda_param.view(1, 1, 1, 1)
        diff_attn_output = (attn1 - lambda_val * attn2) @ v
        normalized_output = self.norm(diff_attn_output.reshape(-1, 2*D)).reshape(B, H, N, 2*D)
        scaled_output = normalized_output * (1.0 - self.lambda_init)
        concatenated_output = scaled_output.transpose(1, 2).reshape(B, N, C * 2)
        x = self.proj(concatenated_output)
        x = self.proj_drop(x)
        return x, None

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_differential_attn=False, layer_idx=1): # use_differential_attn flag remains
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.use_differential_attn = use_differential_attn
        AttentionModule = MultiHeadDifferentialAttention if use_differential_attn else MultiHeadAttention
        self.attn = AttentionModule(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, layer_idx=layer_idx)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        identity1 = x
        x_norm1 = self.norm1(x)
        attn_output, _ = self.attn(x_norm1)
        x = identity1 + attn_output
        identity2 = x
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = identity2 + mlp_output
        return x

class VisionTransformer(nn.Module): # use_differential_attn flag remains
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, pos_embed_type='1d_learned', use_differential_attn=False): # Flag passed here
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pos_embed_type = pos_embed_type
        self.img_size = img_size
        self.use_differential_attn = use_differential_attn
        #print(f"Initializing ViT ({'DiffAttn' if use_differential_attn else 'StdAttn'})...") # Less verbose
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_patches = num_patches
        self.seq_len = num_patches + 1

        if pos_embed_type == 'no_pos':
            self.pos_embed = None
        elif pos_embed_type == '1d_learned':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        elif pos_embed_type == 'sinusoidal':
            self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.seq_len, embed_dim), requires_grad=False)
        else:
            raise ValueError(f"Unknown PE type: {pos_embed_type}")
        nn.init.trunc_normal_(self.cls_token, std=.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                use_differential_attn=use_differential_attn, layer_idx=i+1) # Flag passed down
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                 nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def prepare_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# --- Training Components (Reused) ---
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, num_total_epochs, scheduler=None):
    model.train()
    total_loss=0.0
    correct_predictions=0
    total_samples=0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_total_epochs} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
             if isinstance(scheduler, SequentialLR):
                 scheduler.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        try:
            current_lr = scheduler.get_last_lr()[0]
        except:
            current_lr = optimizer.param_groups[0]['lr']
        batch_acc = (predicted == labels).sum().item()/labels.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}", lr=f"{current_lr:.1e}")
    return total_loss / total_samples, correct_predictions / total_samples

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, desc="[Eval]"):
    model.eval()
    total_loss=0.0
    correct_predictions=0
    total_samples=0
    progress_bar = tqdm(data_loader, desc=desc, leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        batch_acc = (predicted == labels).sum().item()/labels.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
    return total_loss / total_samples, correct_predictions / total_samples

def plot_curves(train_losses, val_losses, train_accs, val_accs, title_suffix='', save_path=None):
    if not train_losses or not val_losses or not train_accs or not val_accs:
        print(f"Warn: No history to plot for {title_suffix}")
        return
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title(f'Loss {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Train Acc')
    plt.plot(epochs, val_accs, 'ro-', label='Val Acc')
    plt.title(f'Accuracy {title_suffix}')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot: {os.path.basename(save_path)}")
    else:
        plt.show()
    plt.close()

# --- Data Loading (Reused) ---
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_NORMALIZE = transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
def get_cifar10_dataloaders(config, data_augmentation_names=None, data_dir=DATA_DIR, val_ratio=0.1, seed=42):
    image_size = config.get('img_size', 32)
    batch_size = config.get('batch_size', BASE_BATCH_SIZE)
    num_workers = config.get('n_workers', os.cpu_count() // 2 or 1)
    baseline_transform = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor(), CIFAR_NORMALIZE])
    try:
        base_train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=baseline_transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=baseline_transform)
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        return None, None, None
    base_train_size = len(base_train_dataset)
    val_size = int(base_train_size * val_ratio)
    train_indices_size = base_train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(base_train_size), [train_indices_size, val_size], generator=generator)
    val_dataset = Subset(base_train_dataset, val_indices)
    train_dataset_orig = Subset(base_train_dataset, train_indices)
    if data_augmentation_names and len(data_augmentation_names) > 0:
        aug_transforms_list = [transforms.Resize([image_size, image_size])]
        if 'random_crop' in data_augmentation_names:
            aug_transforms_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)))
        if 'horizontal_flip' in data_augmentation_names:
            aug_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if 'color_jitter' in data_augmentation_names:
            aug_transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        aug_transforms_list.extend([transforms.ToTensor(), CIFAR_NORMALIZE])
        if 'cutout' in data_augmentation_names or 'random_erasing' in data_augmentation_names:
            aug_transforms_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
        augmented_transform = transforms.Compose(aug_transforms_list)
        try:
            aug_train_instance = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=augmented_transform)
            train_dataset_aug = Subset(aug_train_instance, train_indices)
            final_train_dataset = ConcatDataset([train_dataset_orig, train_dataset_aug])
        except Exception as e:
            print(f"Warn: Aug error: {e}. Using non-aug.")
            final_train_dataset = train_dataset_orig
    else:
        final_train_dataset = train_dataset_orig
    train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# --- Training Runner Function (for Tuning) ---
def run_training_for_tuning(config, num_epochs, data_augmentation_names=None, experiment_name="DiffViT_Tune_Run"):
    """Runs training FOR DIFF-VIT ONLY and returns best state_dict, history, test_acc, val_acc."""
    print(f"\n--- Running Tuning Config: {experiment_name} ---")
    print(f"Parameters: {json.dumps(config, indent=2)}")
    print(f"Data Augmentations: {data_augmentation_names}")
    print(f"Positional Embedding: {config.get('pos_embed_type', '1d_learned')}")
    print(f"Number of Epochs: {num_epochs}")

    trainloader, valloader, testloader = get_cifar10_dataloaders(config, data_augmentation_names=data_augmentation_names)
    if trainloader is None:
        print("!!! Data loading failed. Skipping. !!!")
        return None, {}, -1.0, -1.0

    try:
        # ALWAYS use differential attention for tuning runs in this script
        model = VisionTransformer(
            img_size=config.get('img_size', 32),
            patch_size=config['patch_size'],
            embed_dim=config.get('embed_dim', 192),
            depth=config.get('depth', 8),
            num_heads=config.get('num_heads', 6),
            mlp_ratio=config.get('mlp_ratio', 4.),
            qkv_bias=config.get('qkv_bias', True),
            drop_rate=config.get('drop_rate', 0.1),
            attn_drop_rate=config.get('attn_drop_rate', 0.1),
            num_classes=10,
            pos_embed_type=config.get('pos_embed_type', '1d_learned'),
            use_differential_attn=True # HARDCODED for this tuning script
        ).to(DEVICE)
        print(f"Diff-ViT Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"!!! Model Creation Error: {e}. Skipping. !!!")
        return None, {}, -1.0, -1.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.get('learning_rate', 5e-4), weight_decay=config.get('weight_decay', 0.05))
    steps_per_epoch = len(trainloader)
    warmup_epochs = config.get('warmup_epochs', 5)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    decay_steps = max(total_steps - warmup_steps, 1)
    if warmup_steps >= total_steps:
        warmup_steps = max(0, total_steps - 1)
        decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps) if warmup_steps > 0 else None
    else:
        scheduler1 = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    best_val_acc = 0.0
    best_model_wts = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, trainloader, DEVICE, epoch, num_epochs, scheduler)
        val_loss, val_acc = evaluate(model, criterion, valloader, DEVICE, desc="[Val]")
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        try:
            current_lr = scheduler.get_last_lr()[0]
        except:
            current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.1e}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"✨ New best val acc: {best_val_acc:.4f}")

    total_time_min = (time.time() - start_time) / 60
    print(f"Training finished in {total_time_min:.2f} min. Best Val Acc: {best_val_acc:.4f}")

    test_acc_final = -1.0
    test_loss_final = -1.0
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    else:
        print("Warning: No best validation weights found, evaluating final weights.")
    try:
        test_loss_final, test_acc_final = evaluate(model, criterion, testloader, DEVICE, desc="[Test]")
        print(f"Final Test Accuracy (using best val wts): {test_acc_final:.4f}")
    except Exception as e:
        print(f"Error during final test evaluation: {e}")

    # Don't save individual plots/models here, handled by main loop saving stage/overall best
    return best_model_wts, history, test_acc_final, best_val_acc


# --- Helper Function to Save Stage Results ---
def save_stage_results(stage_name, best_info, models_dir, results_dir, plots_dir, tuning_log_data):
    """Saves the best Diff-ViT model state_dict, config, and plot for a tuning stage."""
    if not best_info or not best_info.get('config') or not best_info.get('state_dict'):
        log_entry = f"[{stage_name}] No best results found or data missing, skipping saving."
        print(log_entry)
        tuning_log_data.append(log_entry)
        return
    stage_id = best_info.get('id', stage_name.lower().replace(" ", "_"))
    val_acc = best_info.get('val_acc', -1.0)
    config = best_info.get('config')
    state_dict = best_info.get('state_dict')
    history = best_info.get('history')
    log_entry = f"\n--- Saving Best Results for {stage_name} (ID: {stage_id}, Val Acc: {val_acc:.4f}) ---"
    print(log_entry)
    tuning_log_data.append(log_entry)
    model_filename = os.path.join(models_dir, f"best_diff_vit_{stage_id}.pth")
    config_filename = os.path.join(results_dir, f"best_config_diff_vit_{stage_id}.json")
    plot_filename = os.path.join(plots_dir, f"best_curves_diff_vit_{stage_id}.png")
    try:
        config_to_save = config.copy()
        config_to_save['achieved_val_accuracy'] = val_acc
        with open(config_filename, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        log_entry = f"[{stage_name}] Saved best config to {config_filename}"
        print(log_entry)
        tuning_log_data.append(log_entry)
    except Exception as e:
        log_entry = f"[{stage_name}] Error saving config: {e}"
        print(log_entry)
        tuning_log_data.append(log_entry)
    try:
        torch.save(state_dict, model_filename)
        log_entry = f"[{stage_name}] Saved best model state_dict to {model_filename}"
        print(log_entry)
        tuning_log_data.append(log_entry)
    except Exception as e:
        log_entry = f"[{stage_name}] Error saving model state_dict: {e}"
        print(log_entry)
        tuning_log_data.append(log_entry)
    if history:
        try:
            plot_curves(history.get('train_loss'), history.get('val_loss'), history.get('train_acc'), history.get('val_acc'), title_suffix=f'(Best {stage_name}: {stage_id})', save_path=plot_filename)
            tuning_log_data.append(f"[{stage_name}] Saved best training plots to {plot_filename}")
        except Exception as e:
             log_entry = f"[{stage_name}] Error saving plot: {e}"
             print(log_entry)
             tuning_log_data.append(log_entry)
    else:
        log_entry = f"[{stage_name}] Warning: History missing for plotting."
        print(log_entry)
        tuning_log_data.append(log_entry)


# --- Main Sequential Tuning Function (for Diff-ViT Only) ---
def main_sequential_tuning():
    """Performs sequential tuning for Diff-ViT: Patch -> Params -> Aug -> PosEmbed."""
    tuning_log_data = [] # Store log messages
    start_overall_time = time.time()

    # --- Overall Best Diff-ViT Tracking ---
    overall_best_info = {'val_acc': 0.0, 'test_acc': -1.0, 'config': None, 'state_dict': None, 'history': None, 'stage_achieved': "None", 'run_id': "None"}

    log_entry = f"=== Starting Diff-ViT Sequential Tuning ({time.strftime('%Y-%m-%d %H:%M:%S')}) ==="
    print("\n" + log_entry)
    tuning_log_data.append(log_entry)
    tuning_log_data.append(f"Device: {DEVICE}")
    tuning_log_data.append(f"Results Dir: {RESULTS_DIR_BASE}")
    tuning_log_data.append(f"Epochs per Tuning Run: {NUM_EPOCHS_TUNE}")

    # --- Stage 1: Patch Size Tuning ---
    stage1_start_time = time.time()
    log_entry = "\n" + "="*15 + f" Stage 1: Patch Size Tuning ({NUM_EPOCHS_TUNE} Epochs) " + "="*15
    print(log_entry)
    tuning_log_data.append(log_entry)
    tuning_log_data.append(f"Data Augmentation: None")

    baseline_config_p = { # Baseline config for patch size tuning (small model)
        'img_size': 32, 'patch_size': 4, 'embed_dim': 192, 'depth': 6, 'num_heads': 6, 'mlp_ratio': 4.,
        'batch_size': BASE_BATCH_SIZE, 'learning_rate': 5e-4, 'weight_decay': 0.05, 'drop_rate': 0.1,
        'attn_drop_rate': 0.1, 'warmup_epochs': 5, 'pos_embed_type': '1d_learned',
        'n_workers': os.cpu_count() // 2 or 1
    }
    patch_sizes_to_try = [4, 8]
    best_patch_info = {'val_acc': 0.0, 'ps': -1, 'config': None, 'state_dict': None, 'history': None, 'id': None}

    for ps in patch_sizes_to_try:
        run_id = f"S1_Patch_{ps}"
        print(f"\n--- Testing {run_id} (Diff-ViT) ---")
        config_ps = baseline_config_p.copy()
        config_ps['patch_size'] = ps
        if ps == 2: config_ps['batch_size'] = BASE_BATCH_SIZE // 2
        elif ps == 4: config_ps['batch_size'] = BASE_BATCH_SIZE
        elif ps == 8: config_ps['batch_size'] = BASE_BATCH_SIZE * 2
        if config_ps['img_size'] % ps != 0:
            print(f"Skip {ps}: img not divisible.")
            continue
        if config_ps['embed_dim'] % config_ps['num_heads'] != 0:
            print(f"Skip {ps}: embed/heads mismatch.")
            continue

        # Run ONLY Diff-ViT
        state_dict_run, history_run, test_acc_run, best_val_acc_run = run_training_for_tuning(
            config_ps, num_epochs=NUM_EPOCHS_TUNE, data_augmentation_names=None, experiment_name=run_id
        )
        log_msg = f"{run_id}: Best Val Acc = {best_val_acc_run:.4f}, Final Test Acc = {test_acc_run:.4f}"
        print(log_msg)
        tuning_log_data.append(log_msg)

        # Update best for this stage (based only on Diff-ViT run)
        if state_dict_run is not None and best_val_acc_run > best_patch_info['val_acc']:
            best_patch_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'ps': ps,
                               'config': config_ps, 'state_dict': state_dict_run, 'history': history_run, 'id': run_id}
            print(f"*** New best for Stage 1: {run_id} (Val Acc: {best_patch_info['val_acc']:.4f}) ***")

        # Update overall best Diff-ViT tracker
        if state_dict_run is not None and best_val_acc_run > overall_best_info['val_acc']:
            overall_best_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'config': config_ps,
                                 'state_dict': state_dict_run, 'history': history_run,
                                 'stage_achieved': "Stage 1 (Patch)", 'run_id': run_id}
            print(f"*** New OVERALL Best Diff-ViT Found! (From {run_id}, Val Acc: {overall_best_info['val_acc']:.4f}) ***")

    stage1_time = time.time() - stage1_start_time
    if best_patch_info['ps'] == -1:
        log_entry = "!!! Stage 1 (Patch) failed. Exiting. !!!"
        print(log_entry)
        tuning_log_data.append(log_entry)
        with open(TUNING_LOG_FILENAME, 'w') as f:
            f.write("\n".join(tuning_log_data))
        return
    log_entry = f"\n--- Stage 1 Result (Duration: {stage1_time/60:.2f} min): Best Patch Size = {best_patch_info['ps']} (Val Acc: {best_patch_info['val_acc']:.4f}) ---"
    print(log_entry)
    tuning_log_data.append(log_entry)
    save_stage_results("Stage1_Patch", best_patch_info, MODELS_DIR, RESULTS_DIR_BASE, PLOTS_DIR, tuning_log_data)
    best_config_stage1 = best_patch_info['config'] # Config to carry forward

    # --- Stage 2: Model Parameter Tuning ---
    stage2_start_time = time.time()
    log_entry = "\n" + "="*15 + f" Stage 2: Model Parameter Tuning ({NUM_EPOCHS_TUNE} Epochs) " + "="*15
    print(log_entry)
    tuning_log_data.append(log_entry)
    tuning_log_data.append(f"Using Best Patch Size: {best_config_stage1['patch_size']}")
    param_tune_augmentation_names = ['random_crop', 'horizontal_flip'] # Use mild augmentation
    tuning_log_data.append(f"Using Augmentations: {param_tune_augmentation_names}")

    base_config_m = best_config_stage1.copy() # Inherits best patch size and adjusted batch size
    base_config_m.update({'embed_dim': 256, 'depth': 8, 'num_heads': 8, 'learning_rate': 3e-4, 'warmup_epochs': 5})
    param_configs_to_try = { # Define param variations relative to base_config_m
        "baseline_m": base_config_m.copy(),
        "mlp_ratio_2": {**base_config_m, 'mlp_ratio': 2.},
        "deeper": {**base_config_m, 'depth': 12, 'num_heads': 8},
        "wider": {**base_config_m, 'embed_dim': 384, 'num_heads': 12, 'batch_size': max(32, base_config_m['batch_size']//2)},
        "smaller": {**base_config_m, 'embed_dim': 192, 'depth': 6, 'num_heads': 6}
    }
    best_param_info = {'val_acc': 0.0, 'name': None, 'config': None, 'state_dict': None, 'history': None, 'id': None}

    for name, config_m in param_configs_to_try.items():
        run_id = f"S2_Param_{name}"
        print(f"\n--- Testing {run_id} (Diff-ViT) ---")
        if config_m['embed_dim'] % config_m['num_heads'] != 0:
            print(f"Skipping '{name}': embed/heads mismatch.")
            continue

        state_dict_run, history_run, test_acc_run, best_val_acc_run = run_training_for_tuning(
            config_m, num_epochs=NUM_EPOCHS_TUNE, data_augmentation_names=param_tune_augmentation_names, experiment_name=run_id
        )
        log_msg = f"{run_id}: Best Val Acc = {best_val_acc_run:.4f}, Final Test Acc = {test_acc_run:.4f}"
        print(log_msg)
        tuning_log_data.append(log_msg)

        # Update best for this stage
        if state_dict_run is not None and best_val_acc_run > best_param_info['val_acc']:
            best_param_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'name': name,
                               'config': config_m, 'state_dict': state_dict_run, 'history': history_run, 'id': run_id}
            print(f"*** New best for Stage 2: {run_id} (Val Acc: {best_param_info['val_acc']:.4f}) ***")

        # Update overall best Diff-ViT tracker
        if state_dict_run is not None and best_val_acc_run > overall_best_info['val_acc']:
            overall_best_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'config': config_m,
                                 'state_dict': state_dict_run, 'history': history_run,
                                 'stage_achieved': "Stage 2 (Model Param)", 'run_id': run_id}
            print(f"*** New OVERALL Best Diff-ViT Found! (From {run_id}, Val Acc: {overall_best_info['val_acc']:.4f}) ***")

    stage2_time = time.time() - stage2_start_time
    if best_param_info['name'] is None:
        log_entry = "!!! Stage 2 (Model Param) failed. Exiting. !!!"
        print(log_entry)
        tuning_log_data.append(log_entry)
        with open(TUNING_LOG_FILENAME, 'w') as f:
            f.write("\n".join(tuning_log_data))
        return
    log_entry = f"\n--- Stage 2 Result (Duration: {stage2_time/60:.2f} min): Best Model Config = '{best_param_info['name']}' (Val Acc: {best_param_info['val_acc']:.4f}) ---"
    print(log_entry)
    tuning_log_data.append(log_entry)
    tuning_log_data.append(f"Best Stage 2 Config Params:\n{json.dumps(best_param_info['config'], indent=2)}")
    save_stage_results("Stage2_Model", best_param_info, MODELS_DIR, RESULTS_DIR_BASE, PLOTS_DIR, tuning_log_data)
    best_config_stage2 = best_param_info['config'] # Config to carry forward

    # --- Stage 3: Data Augmentation Tuning ---
    stage3_start_time = time.time()
    log_entry = "\n" + "="*15 + f" Stage 3: Data Augmentation Tuning ({NUM_EPOCHS_TUNE} Epochs) " + "="*15
    print(log_entry)
    tuning_log_data.append(log_entry)
    tuning_log_data.append(f"Using Best Model Config: '{best_param_info['name']}' (Patch: {best_config_stage2['patch_size']})")

    augmentation_strategies = { "None": None, "Mild": ['random_crop', 'horizontal_flip'],}
    config_aug_base = best_config_stage2.copy() # Best config from stage 2
    best_aug_info = {'val_acc': 0.0, 'name': None, 'config': None, 'state_dict': None, 'history': None, 'id': None, 'augmentation_names': None}

    for aug_name, aug_names_list in augmentation_strategies.items():
        run_id = f"S3_Aug_{aug_name}"
        print(f"\n--- Testing {run_id} (Diff-ViT) ---")
        print(f"Augmentations: {aug_names_list}")

        state_dict_run, history_run, test_acc_run, best_val_acc_run = run_training_for_tuning(
            config_aug_base, num_epochs=NUM_EPOCHS_TUNE, data_augmentation_names=aug_names_list, experiment_name=run_id
        )
        log_msg = f"{run_id}: Best Val Acc = {best_val_acc_run:.4f}, Final Test Acc = {test_acc_run:.4f}"
        print(log_msg)
        tuning_log_data.append(log_msg)

        # Update best for this stage
        if state_dict_run is not None and best_val_acc_run > best_aug_info['val_acc']:
            best_aug_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'name': aug_name, 'config': config_aug_base,
                             'state_dict': state_dict_run, 'history': history_run, 'id': run_id, 'augmentation_names': aug_names_list}
            print(f"*** New best for Stage 3: {run_id} (Val Acc: {best_aug_info['val_acc']:.4f}) ***")

        # Update overall best Diff-ViT tracker
        if state_dict_run is not None and best_val_acc_run > overall_best_info['val_acc']:
            current_best_config = config_aug_base.copy()
            current_best_config['data_augmentation_names'] = aug_names_list
            overall_best_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'config': current_best_config,
                                 'state_dict': state_dict_run, 'history': history_run,
                                 'stage_achieved': "Stage 3 (Augmentation)", 'run_id': run_id}
            print(f"*** New OVERALL Best Diff-ViT Found! (From {run_id}, Val Acc: {overall_best_info['val_acc']:.4f}) ***")

    stage3_time = time.time() - stage3_start_time
    if best_aug_info['name'] is None:
         log_entry = "!!! Stage 3 (Augmentation) failed. Exiting. !!!"
         print(log_entry)
         tuning_log_data.append(log_entry)
         with open(TUNING_LOG_FILENAME, 'w') as f:
             f.write("\n".join(tuning_log_data))
         return
    log_entry = f"\n--- Stage 3 Result (Duration: {stage3_time/60:.2f} min): Best Aug Strategy = '{best_aug_info['name']}' (Val Acc: {best_aug_info['val_acc']:.4f}) ---"
    print(log_entry)
    tuning_log_data.append(log_entry)
    save_stage_results("Stage3_Aug", best_aug_info, MODELS_DIR, RESULTS_DIR_BASE, PLOTS_DIR, tuning_log_data)
    best_config_stage3 = best_aug_info['config'] # Base config used
    best_aug_names_stage3 = best_aug_info['augmentation_names'] # Best augmentations found

    # --- Stage 4: Positional Embedding Tuning ---
    stage4_start_time = time.time()
    log_entry = "\n" + "="*15 + f" Stage 4: Positional Embedding Tuning ({NUM_EPOCHS_TUNE} Epochs) " + "="*15
    print(log_entry)
    tuning_log_data.append(log_entry)
    tuning_log_data.append(f"Using Best Model Config: '{best_param_info['name']}' (Patch: {best_config_stage3['patch_size']})")
    tuning_log_data.append(f"Using Best Augmentation: '{best_aug_info['name']}'")

    pe_types_to_try = ['1d_learned','sinusoidal', 'no_pos'] # Test all options including default
    config_pos_base = best_config_stage3.copy() # Best config from stage 3 run
    aug_names_pos_base = best_aug_names_stage3 # Best augs from stage 3
    best_pos_info = {'val_acc': 0.0, 'name': None, 'config': None, 'state_dict': None, 'history': None, 'id': None}

    for pe_type in pe_types_to_try:
        run_id = f"S4_PE_{pe_type}"
        print(f"\n--- Testing {run_id} (Diff-ViT) ---")
        config_pe = config_pos_base.copy()
        config_pe['pos_embed_type'] = pe_type

        state_dict_run, history_run, test_acc_run, best_val_acc_run = run_training_for_tuning(
            config_pe, num_epochs=NUM_EPOCHS_TUNE, data_augmentation_names=aug_names_pos_base, experiment_name=run_id
        )
        log_msg = f"{run_id}: Best Val Acc = {best_val_acc_run:.4f}, Final Test Acc = {test_acc_run:.4f}"
        print(log_msg)
        tuning_log_data.append(log_msg)

        # Update best for this stage
        if state_dict_run is not None and best_val_acc_run > best_pos_info['val_acc']:
             best_pos_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'name': pe_type, 'config': config_pe,
                              'state_dict': state_dict_run, 'history': history_run, 'id': run_id}
             print(f"*** New best for Stage 4: {run_id} (Val Acc: {best_pos_info['val_acc']:.4f}) ***")

        # Update overall best Diff-ViT tracker
        if state_dict_run is not None and best_val_acc_run > overall_best_info['val_acc']:
            current_best_config = config_pe.copy()
            current_best_config['data_augmentation_names'] = aug_names_pos_base
            overall_best_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'config': current_best_config,
                                 'state_dict': state_dict_run, 'history': history_run,
                                 'stage_achieved': "Stage 4 (Pos Embed)", 'run_id': run_id}
            print(f"*** New OVERALL Best Diff-ViT Found! (From {run_id}, Val Acc: {overall_best_info['val_acc']:.4f}) ***")

    stage4_time = time.time() - stage4_start_time
    if best_pos_info['name'] is None:
        log_entry = "!!! Stage 4 (Pos Embed) failed to find any viable model. !!!"
        print(log_entry)
        tuning_log_data.append(log_entry)
    else:
        log_entry = f"\n--- Stage 4 Result (Duration: {stage4_time/60:.2f} min): Best Pos Embed Type = '{best_pos_info['name']}' (Val Acc: {best_pos_info['val_acc']:.4f}) ---"
        print(log_entry)
        tuning_log_data.append(log_entry)
        save_stage_results("Stage4_Pos", best_pos_info, MODELS_DIR, RESULTS_DIR_BASE, PLOTS_DIR, tuning_log_data)

    # --- Final Overall Saving ---
    log_entry = "\n" + "="*15 + " Final Overall Diff-ViT Results & Saving " + "="*15
    print(log_entry)
    tuning_log_data.append(log_entry)

    if overall_best_info['config'] and overall_best_info['state_dict']:
        final_best_config = overall_best_info['config'].copy()
        final_best_config['overall_best_val_accuracy'] = overall_best_info['val_acc']
        final_best_config['overall_best_test_accuracy'] = overall_best_info['test_acc']
        final_best_config['overall_best_stage_achieved'] = overall_best_info['stage_achieved']
        final_best_config['overall_best_run_id'] = overall_best_info['run_id']

        log_entry = f"Overall Best Diff-ViT Val Acc: {overall_best_info['val_acc']:.4f} (Achieved in {overall_best_info['stage_achieved']}, Run: {overall_best_info['run_id']})"
        print(log_entry)
        tuning_log_data.append(log_entry)
        log_entry = f"Overall Best Diff-ViT Test Acc (for that run): {overall_best_info['test_acc']:.4f}"
        print(log_entry)
        tuning_log_data.append(log_entry)

        print("Final Overall Best Diff-ViT Config:", json.dumps(final_best_config, indent=2))
        tuning_log_data.append("Final Overall Best Diff-ViT Configuration:\n" + json.dumps(final_best_config, indent=2))

        # Save Overall Best Config
        try:
            with open(BEST_CONFIG_FILENAME, 'w') as f:
                json.dump(final_best_config, f, indent=4)
            log_entry = f"Saved final overall best config to {BEST_CONFIG_FILENAME}"
            print(log_entry)
            tuning_log_data.append(log_entry)
        except Exception as e:
            log_entry = f"Error saving final overall config: {e}"
            print(log_entry)
            tuning_log_data.append(log_entry)

        # Save Overall Best Model State Dict
        try:
            torch.save(overall_best_info['state_dict'], BEST_MODEL_FILENAME)
            log_entry = f"Saved final overall best model state_dict to {BEST_MODEL_FILENAME}"
            print(log_entry)
            tuning_log_data.append(log_entry)
        except Exception as e:
            log_entry = f"Error saving final overall model state_dict: {e}"
            print(log_entry)
            tuning_log_data.append(log_entry)

        # Save Overall Best Training Curves Plot
        if overall_best_info['history']:
            try:
                plot_curves(overall_best_info['history'].get('train_loss'), overall_best_info['history'].get('val_loss'),
                            overall_best_info['history'].get('train_acc'), overall_best_info['history'].get('val_acc'),
                            title_suffix=f'(Overall Best Diff-ViT: {overall_best_info["run_id"]})', save_path=BEST_PLOT_FILENAME)
                tuning_log_data.append(f"Saved overall best training plots to {BEST_PLOT_FILENAME}")
            except Exception as e:
                 log_entry = f"Error saving final overall plot: {e}"
                 print(log_entry)
                 tuning_log_data.append(log_entry)
        else:
            log_entry = "Warning: History for the overall best run not available for plotting."
            print(log_entry)
            tuning_log_data.append(log_entry)

        # Optional: Final evaluation using the saved overall best model
        print("\n--- Re-evaluating Final Overall Best Diff-ViT Model on Test Set ---")
        try:
            model_config_keys = VisionTransformer.__init__.__code__.co_varnames[1:VisionTransformer.__init__.__code__.co_argcount]
            final_model_config = {k: v for k, v in final_best_config.items() if k in model_config_keys}
            # Ensure use_differential_attn is correctly set for model loading
            final_model_config['use_differential_attn'] = True

            final_model = VisionTransformer(**final_model_config).to(DEVICE)
            final_model.load_state_dict(overall_best_info['state_dict'])
            _, _, final_test_loader = get_cifar10_dataloaders(final_best_config, data_augmentation_names=None)
            if final_test_loader:
                 final_test_loss, final_test_acc = evaluate(final_model, nn.CrossEntropyLoss(), final_test_loader, DEVICE)
                 log_entry = f"Final Test Accuracy of saved overall best Diff-ViT model: {final_test_acc:.4f}"
                 print(log_entry)
                 tuning_log_data.append(log_entry)
            else:
                 log_entry = "Could not create test loader for final evaluation."
                 print(log_entry)
                 tuning_log_data.append(log_entry)
        except Exception as e:
            log_entry = f"Error during final re-evaluation: {e}"
            print(log_entry)
            tuning_log_data.append(log_entry)

    else:
        log_entry = "!!! Tuning process failed to produce any viable Diff-ViT model configuration. !!!"
        print(log_entry)
        tuning_log_data.append(log_entry)

    # Save the complete tuning log
    overall_end_time = time.time()
    total_tuning_time = overall_end_time - start_overall_time
    tuning_log_data.append(f"\nTotal Tuning Duration: {total_tuning_time/60:.2f} minutes ({total_tuning_time:.2f} seconds)")
    try:
        with open(TUNING_LOG_FILENAME, 'w') as f:
            f.write("\n".join(tuning_log_data))
        print(f"Saved tuning log to {TUNING_LOG_FILENAME}")
    except Exception as e:
        print(f"Error writing final tuning log: {e}")

    print("\n=== Diff-ViT Sequential Tuning Script Finished ===")


if __name__ == "__main__":
    main_sequential_tuning()
