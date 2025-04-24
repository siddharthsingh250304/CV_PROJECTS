# -*- coding: utf-8 -*-
"""
train_tune_vit.py - MODIFIED FOR SHORT TUNE + LONG FINAL TRAIN

Script 3: Train and Tune Standard Vision Transformer on CIFAR-10 sequentially.
1. Tune Patch Size (20 epochs)
2. Tune Model Parameters (20 epochs, using best patch size)
3. Tune Data Augmentation (20 epochs, using best model config)
4. Tune Positional Embedding (20 epochs, using best model config + best augmentation)
5. Final Training (60 epochs, using overall best configuration found)

Saves the best model state_dict and config FOUND AT EACH TUNING STAGE.
Saves the final model state_dict, config, and plot from the 60-epoch training run.

**Uses ConcatDataset augmentation for TRAINING ONLY.**
**Uses LinearWarmup+CosineAnnealing LR Schedule.**
**Corrected model indentation and evaluation datasets.**
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
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
RESULTS_DIR = "results_vit_tuned" # Changed results dir name
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(MODELS_DIR, exist_ok=True)

# --- MODIFIED EPOCH SETTINGS ---
NUM_EPOCHS_TUNE = 20         # Epochs for ALL tuning stages (Patch, Param, Aug, PosEmbed)
NUM_EPOCHS_FINAL_TRAIN = 60  # Epochs for the final training run with the best config
# --- END MODIFIED EPOCH SETTINGS ---

BEST_MODEL_FILENAME = os.path.join(MODELS_DIR, "best_vit_final_trained.pth") # Final trained model
BEST_CONFIG_FILENAME = os.path.join(RESULTS_DIR, "best_config_final_trained.json") # Config used for final train
BEST_PLOT_FILENAME = os.path.join(PLOTS_DIR, "best_train_curves_final_trained.png") # Curves from final train
TUNING_LOG_FILENAME = os.path.join(RESULTS_DIR, "tuning_log_concat_aug_final_train.txt")
DATA_DIR = './data_cifar' # Specify data directory

# --------------------------------------------------------------------------
# Model Definition (Standard ViT - SAME AS BEFORE)
# --------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            print(f"Warning: Input image size ({H}*{W}) doesn't match model expected size ({self.img_size}*{self.img_size}). Resizing might be needed.")
        x = self.proj(x).flatten(2).transpose(1, 2)  # B x num_patches x embed_dim
        return x

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, dim, attn_dropout=0.):
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = (q @ k.transpose(-2, -1)) / self.sqrt_dim
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9) # Use large negative value
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        output = attn @ v
        return output, attn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn = ScaledDotProductAttention(head_dim, attn_dropout=attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x, attn = self.attn(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class MLP(nn.Module):
    """ MLP as used in Vision Transformer """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """ Standard Transformer Encoder Layer """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        identity1 = x; x_norm1 = self.norm1(x)
        attn_output, attn_weights = self.attn(x_norm1)
        x = identity1 + attn_output
        identity2 = x; x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = identity2 + mlp_output
        return x, attn_weights

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]); sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, pos_embed_type='1d_learned'):
        super().__init__()
        self.num_classes = num_classes; self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size; self.pos_embed_type = pos_embed_type; self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches; self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)); self.pos_drop = nn.Dropout(p=drop_rate)
        self.seq_len = num_patches + 1

        # Positional Embedding Logic
        if pos_embed_type == 'no_pos': self.pos_embed = None; print("Using no positional embedding.")
        elif pos_embed_type == '1d_learned':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim)); nn.init.trunc_normal_(self.pos_embed, std=.02)
            print("Using 1D learned positional embedding.")
        elif pos_embed_type == '2d_learned':
            h_patches, w_patches = img_size // patch_size, img_size // patch_size
            if embed_dim % 2 != 0: raise ValueError("Embed dim must be even for 2D learned PE split.")
            embed_dim_spatial = embed_dim // 2
            self.pos_embed_row = nn.Parameter(torch.zeros(1, h_patches, embed_dim_spatial)); self.pos_embed_col = nn.Parameter(torch.zeros(1, w_patches, embed_dim_spatial))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed_row, std=.02); nn.init.trunc_normal_(self.pos_embed_col, std=.02); nn.init.trunc_normal_(self.pos_embed_cls, std=.02)
            print("Using 2D learned positional embedding.")
        elif pos_embed_type == 'sinusoidal':
            print("Using sinusoidal positional embedding.")
            self.pos_embed = nn.Parameter(get_sinusoid_encoding_table(self.seq_len, embed_dim), requires_grad=False)
        else: raise ValueError(f"Unknown PE type: {pos_embed_type}")
        nn.init.trunc_normal_(self.cls_token, std=.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim); self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)

    def prepare_tokens(self, x):
        B = x.shape[0]; x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1); x = torch.cat((cls_tokens, x), dim=1)

        # --- REVISED POSITIONAL EMBEDDING LOGIC ---
        if self.pos_embed_type == '1d_learned' or self.pos_embed_type == 'sinusoidal':
            # These types are guaranteed to have self.pos_embed defined
            # self.pos_embed will be None only if type was 'no_pos', which is handled below
            if self.pos_embed is not None:
                 x = x + self.pos_embed
        elif self.pos_embed_type == '2d_learned':
            # Construct and add 2D PE using self.pos_embed_row, self.pos_embed_col, self.pos_embed_cls
            h_patches, w_patches = self.img_size // self.patch_size, self.img_size // self.patch_size
            # Need to ensure embed_dim is compatible if not checked in __init__
            if self.embed_dim % 2 != 0:
                 raise ValueError("Embed dim must be even for 2D learned PE split.")
            embed_dim_spatial = self.embed_dim // 2

            pos_embed_row = self.pos_embed_row.unsqueeze(2).expand(-1, -1, w_patches, -1)
            pos_embed_col = self.pos_embed_col.unsqueeze(1).expand(-1, h_patches, -1, -1)
            pos_embed_patches = torch.cat((pos_embed_row, pos_embed_col), dim=-1).flatten(1, 2)
            full_pos_embed = torch.cat((self.pos_embed_cls, pos_embed_patches), dim=1)
            x = x + full_pos_embed
        #elif self.pos_embed_type == 'no_pos':
            # Explicitly do nothing if type is 'no_pos', as self.pos_embed is None
        #    pass
        # else: # Optional: handle unknown type if __init__ didn't catch it
        #    raise ValueError(f"Unknown pos_embed_type encountered in prepare_tokens: {self.pos_embed_type}")
        # --- END REVISED LOGIC ---

        x = self.pos_drop(x)
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks: x, _ = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# --------------------------------------------------------------------------
# Training Components (SAME AS BEFORE)
# --------------------------------------------------------------------------
def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, num_total_epochs, scheduler=None): # Added num_total_epochs
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_total_epochs} [Train]", leave=False) # Show total epochs
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
             if isinstance(scheduler, SequentialLR): scheduler.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        try: current_lr = scheduler.get_last_lr()[0]
        except: current_lr = optimizer.param_groups[0]['lr']
        batch_acc = (predicted == labels).sum().item() / labels.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}", lr=f"{current_lr:.1e}")
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, desc_prefix="[Eval]"): # Added desc_prefix
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(data_loader, desc=desc_prefix, leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        batch_acc = (predicted == labels).sum().item() / labels.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")
    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def plot_curves(train_losses, val_losses, train_accs, val_accs, title_suffix='', save_path=None):
    if not train_losses or not val_losses or not train_accs or not val_accs:
        print(f"Warning: Cannot plot curves for '{title_suffix}'. Missing history data.")
        return
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss'); plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'Loss {title_suffix}'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training Accuracy'); plt.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    plt.title(f'Accuracy {title_suffix}'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.ylim(0, 1.0); plt.legend(); plt.grid(True)
    plt.tight_layout()
    if save_path:
        try: plt.savefig(save_path); print(f"Saved plot to {save_path}")
        except Exception as e: print(f"Error saving plot to {save_path}: {e}")
    else: plt.show()
    plt.close()

# --------------------------------------------------------------------------
# Data Loading Helper (SAME AS BEFORE)
# --------------------------------------------------------------------------
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]; CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_NORMALIZE = transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)

def get_cifar10_dataloaders(config, data_augmentation_names=None, data_dir=DATA_DIR, val_ratio=0.1, seed=42):
    image_size = config.get('img_size', 32); batch_size = config.get('batch_size', 128)
    default_workers = min(os.cpu_count() // 2, 8) if os.cpu_count() else 1
    num_workers = config.get('n_workers', default_workers)
    baseline_transform = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor(), CIFAR_NORMALIZE])
    try:
        base_train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=baseline_transform)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=baseline_transform)
    except Exception as e: print(f"Error loading base CIFAR-10: {e}"); return None, None, None
    base_train_size = len(base_train_dataset); val_size = int(base_train_size * val_ratio); train_indices_size = base_train_size - val_size
    if train_indices_size <= 0 or val_size <= 0: print(f"Error: Invalid train/val split"); return None, None, None
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(base_train_size), [train_indices_size, val_size], generator=generator)
    val_dataset = Subset(base_train_dataset, val_indices)
    train_dataset_orig = Subset(base_train_dataset, train_indices)
    if data_augmentation_names and len(data_augmentation_names) > 0:
        aug_transforms_list = [transforms.Resize([image_size, image_size])]
        if 'random_crop' in data_augmentation_names: aug_transforms_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)))
        if 'horizontal_flip' in data_augmentation_names: aug_transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if 'color_jitter' in data_augmentation_names: aug_transforms_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        aug_transforms_list.extend([transforms.ToTensor(), CIFAR_NORMALIZE])
        if 'cutout' in data_augmentation_names or 'random_erasing' in data_augmentation_names: aug_transforms_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))
        augmented_transform = transforms.Compose(aug_transforms_list)
        try:
            aug_train_instance = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=augmented_transform)
            train_dataset_aug = Subset(aug_train_instance, train_indices)
            final_train_dataset = ConcatDataset([train_dataset_orig, train_dataset_aug])
        except Exception as e: print(f"Error creating augmented dataset: {e}. Falling back."); final_train_dataset = train_dataset_orig
    else: final_train_dataset = train_dataset_orig
    train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    eval_batch_size = batch_size * 2
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# --------------------------------------------------------------------------
# Training Runner Function (SAME AS BEFORE)
# --------------------------------------------------------------------------
def run_training_for_tuning(config, num_epochs, data_augmentation_names=None,
                           experiment_name=""):
    """Runs training and returns best model state_dict, history, final test acc, best val acc."""
    print(f"\n--- Running Config: {experiment_name} ---")
    print(f"Parameters: {json.dumps(config, indent=2)}")
    print(f"Data Augmentations: {data_augmentation_names}")
    print(f"Positional Embedding: {config.get('pos_embed_type', '1d_learned')}")
    print(f"Number of Epochs: {num_epochs}")

    trainloader, valloader, testloader = get_cifar10_dataloaders(
        config, data_augmentation_names=data_augmentation_names
    )
    if trainloader is None: return None, {}, -1.0, -1.0

    try:
        model = VisionTransformer(
            img_size=config.get('img_size', 32), patch_size=config['patch_size'],
            embed_dim=config['embed_dim'], depth=config['depth'], num_heads=config['num_heads'],
            mlp_ratio=config.get('mlp_ratio', 4.), qkv_bias=config.get('qkv_bias', True),
            drop_rate=config.get('drop_rate', 0.1), attn_drop_rate=config.get('attn_drop_rate', 0.1),
            num_classes=10, pos_embed_type=config.get('pos_embed_type', '1d_learned')
        ).to(DEVICE)
        print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} params.")
    except Exception as e: print(f"!!! Error creating model: {e}. Skipping. !!!"); return None, {}, -1.0, -1.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.get('learning_rate', 1e-3), weight_decay=config.get('weight_decay', 0.05))

    steps_per_epoch = len(trainloader)
    warmup_epochs = config.get('warmup_epochs', 5)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    decay_steps = max(total_steps - warmup_steps, 1)

    print(f"Scheduler: Warmup Epochs={warmup_epochs}, Total Epochs={num_epochs}")
    print(f"Scheduler Steps: Warmup={warmup_steps}, Decay={decay_steps}, Total={total_steps}")

    if warmup_steps >= total_steps: warmup_steps = max(0, total_steps - 1); decay_steps = total_steps - warmup_steps
    if decay_steps <= 0: scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps) if warmup_steps > 0 else None
    else:
        scheduler1 = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    best_val_acc = 0.0; best_model_wts = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, trainloader, DEVICE, epoch, num_epochs, scheduler) # Pass num_epochs
        val_loss, val_acc = evaluate(model, criterion, valloader, DEVICE)
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        try: current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        except: current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.1e}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc; best_model_wts = copy.deepcopy(model.state_dict())
            print(f"✨ New best val acc: {best_val_acc:.4f} at epoch {epoch+1}")

    end_time = time.time(); total_time = end_time - start_time
    print(f"Training finished in {total_time/60:.2f} min. Best Val Acc: {best_val_acc:.4f}")

    test_acc_final = -1.0
    if best_model_wts:
        print("Loading best val weights for final test eval...")
        try:
            model.load_state_dict(best_model_wts)
            test_loss_final, test_acc_final = evaluate(model, criterion, testloader, DEVICE, desc_prefix="[Test Best]")
            print(f"Final Test Accuracy (best val wts): {test_acc_final:.4f}")
        except Exception as e: print(f"Error loading best weights/eval: {e}"); test_acc_final = -2.0
    else:
        print("No improvement observed. Evaluating final weights on test set...")
        try:
            test_loss_final, test_acc_final = evaluate(model, criterion, testloader, DEVICE, desc_prefix="[Test Final]")
            print(f"Final Test Accuracy (final wts): {test_acc_final:.4f}")
        except Exception as e: print(f"Error eval final weights: {e}"); test_acc_final = -2.0

    return best_model_wts, history, test_acc_final, best_val_acc

# --------------------------------------------------------------------------
# Helper Function to Save Stage Results (SAME AS BEFORE)
# --------------------------------------------------------------------------
def save_stage_results(stage_name, best_info, models_dir, results_dir, plots_dir, tuning_log_data):
    if not best_info or not best_info.get('config') or not best_info.get('state_dict'):
        log_entry = f"[{stage_name}] No best results found/data missing, skipping saving."; print(log_entry); tuning_log_data.append(log_entry); return
    stage_id = best_info.get('id', stage_name.lower().replace(" ", "_")); val_acc = best_info.get('val_acc', -1.0)
    config = best_info.get('config'); state_dict = best_info.get('state_dict'); history = best_info.get('history')
    log_entry = f"\n--- Saving Best Results for {stage_name} (ID: {stage_id}, Val Acc: {val_acc:.4f}) ---"; print(log_entry); tuning_log_data.append(log_entry)
    model_filename = os.path.join(models_dir, f"best_vit_{stage_id}.pth")
    config_filename = os.path.join(results_dir, f"best_config_{stage_id}.json")
    plot_filename = os.path.join(plots_dir, f"best_curves_{stage_id}.png")
    try:
        config_to_save = config.copy(); config_to_save['achieved_val_accuracy'] = val_acc
        with open(config_filename, 'w') as f: json.dump(config_to_save, f, indent=4)
        log_entry = f"[{stage_name}] Saved best config to {config_filename}"; print(log_entry); tuning_log_data.append(log_entry)
    except Exception as e: log_entry = f"[{stage_name}] Error saving config: {e}"; print(log_entry); tuning_log_data.append(log_entry)
    try:
        torch.save(state_dict, model_filename)
        log_entry = f"[{stage_name}] Saved best model state_dict to {model_filename}"; print(log_entry); tuning_log_data.append(log_entry)
    except Exception as e: log_entry = f"[{stage_name}] Error saving model state_dict: {e}"; print(log_entry); tuning_log_data.append(log_entry)
    if history:
        try:
            plot_curves(history.get('train_loss'), history.get('val_loss'), history.get('train_acc'), history.get('val_acc'),
                        title_suffix=f'(Best {stage_name}: {stage_id})', save_path=plot_filename)
            tuning_log_data.append(f"[{stage_name}] Saved best training plots to {plot_filename}")
        except Exception as e: log_entry = f"[{stage_name}] Error saving plot: {e}"; print(log_entry); tuning_log_data.append(log_entry)
    else: log_entry = f"[{stage_name}] Warning: History missing for plotting."; print(log_entry); tuning_log_data.append(log_entry)


# -*- coding: utf-8 -*-
"""
train_tune_vit.py - MODIFIED FOR RESUME + COMPLETE TUNE + FINAL TRAIN

Script 4: Resumes tuning, completes Stage 4, identifies overall best,
          then runs final 60-epoch training.
"""

# ... (Keep all imports, constants, model def, helpers, etc.) ...
# --- Use the CORRECTED VisionTransformer class ---

# Global Settings
# ...
RESULTS_DIR = "results_vit_tuned" # Point to the directory with existing partial results
FINAL_RESULTS_DIR = "results_vit_concat_aug_final_train" # Separate dir for final run output
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
FINAL_PLOTS_DIR = os.path.join(FINAL_RESULTS_DIR, "plots")
FINAL_MODELS_DIR = os.path.join(FINAL_RESULTS_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FINAL_RESULTS_DIR, exist_ok=True) # Ensure final dir exists
os.makedirs(FINAL_PLOTS_DIR, exist_ok=True); os.makedirs(FINAL_MODELS_DIR, exist_ok=True)

NUM_EPOCHS_TUNE = 20
NUM_EPOCHS_FINAL_TRAIN = 60

# --- Filenames for Stage 5 / Final Output ---
BEST_MODEL_FILENAME = os.path.join(FINAL_MODELS_DIR, "best_vit_final_trained.pth")
BEST_CONFIG_FILENAME = os.path.join(FINAL_RESULTS_DIR, "best_config_final_trained.json")
BEST_PLOT_FILENAME = os.path.join(FINAL_PLOTS_DIR, "best_train_curves_final_trained.png")
TUNING_LOG_FILENAME = os.path.join(RESULTS_DIR, "tuning_log_resumed.txt") # New log file


def main_sequential_tuning():
    """Resumes tuning, completes Stage 4, finds overall best, runs final training."""
    tuning_log_data = []
    start_overall_time = time.time()

    # --- Overall Best Config Tracking ---
    overall_best_config_info = {
        'val_acc': 0.0, 'config': None, 'stage_achieved': "None", 'run_id': "None"
    }

    log_entry = f"=== RESUMING ViT Sequential Tuning ({time.strftime('%Y-%m-%d %H:%M:%S')}) ==="
    # ... (logging setup) ...
    tuning_log_data.append(f"Tuning Results Dir: {RESULTS_DIR}")
    tuning_log_data.append(f"Final Output Dir: {FINAL_RESULTS_DIR}")

    # --- Attempt to Load Previous State ---
    best_config_stage1 = None
    best_config_stage2 = None
    best_config_stage3 = None # This is the base config for Stage 4
    best_aug_names_stage3 = None
    best_param_info_name_s2 = None # Keep track of the best model name from S2

    # Check Stage 1 best (Assuming Patch 4 was best)
    s1_config_path = os.path.join(RESULTS_DIR, "best_config_Stage1_Patch_4.json")
    s1_run_id = "Stage1_Patch_4"
    if os.path.exists(s1_config_path):
        try:
            with open(s1_config_path, 'r') as f: best_config_stage1 = json.load(f)
            log_entry = f"Loaded existing Stage 1 best config from {s1_run_id}"; print(log_entry); tuning_log_data.append(log_entry)
            # Update overall best based on this loaded stage if needed
            s1_val_acc = best_config_stage1.get('achieved_val_accuracy', 0.0)
            if s1_val_acc > overall_best_config_info['val_acc']:
                 overall_best_config_info.update({'val_acc': s1_val_acc, 'config': best_config_stage1, 'stage_achieved': "Stage 1", 'run_id': s1_run_id})
                 log_entry = f"Updated overall best config from loaded Stage 1: {s1_run_id} (Val Acc: {s1_val_acc:.4f})"; print(log_entry); tuning_log_data.append(log_entry)

        except Exception as e:
            log_entry = f"Warning: Found Stage 1 config file but failed to load: {e}"; print(log_entry); tuning_log_data.append(log_entry)
            best_config_stage1 = None # Reset on failure

    # Check Stage 2 best (Assuming Wider was best)
    s2_config_path = os.path.join(RESULTS_DIR, "best_config_Stage2_Model_wider.json")
    s2_run_id = "Stage2_Model_wider"
    if os.path.exists(s2_config_path):
        try:
            with open(s2_config_path, 'r') as f: best_config_stage2 = json.load(f)
            log_entry = f"Loaded existing Stage 2 best config from {s2_run_id}"; print(log_entry); tuning_log_data.append(log_entry)
            best_param_info_name_s2 = "wider" # Hardcoded based on filename/log
            # Update overall best
            s2_val_acc = best_config_stage2.get('achieved_val_accuracy', 0.0)
            if s2_val_acc > overall_best_config_info['val_acc']:
                 overall_best_config_info.update({'val_acc': s2_val_acc, 'config': best_config_stage2, 'stage_achieved': "Stage 2", 'run_id': s2_run_id})
                 log_entry = f"Updated overall best config from loaded Stage 2: {s2_run_id} (Val Acc: {s2_val_acc:.4f})"; print(log_entry); tuning_log_data.append(log_entry)

        except Exception as e:
            log_entry = f"Warning: Found Stage 2 config file but failed to load: {e}"; print(log_entry); tuning_log_data.append(log_entry)
            best_config_stage2 = None # Reset on failure
            best_param_info_name_s2 = None

    # Check Stage 3 best (Assuming Mild was best)
    s3_config_path = os.path.join(RESULTS_DIR, "best_config_Stage3_Aug_Mild.json")
    s3_run_id = "Stage3_Aug_Mild"
    if os.path.exists(s3_config_path):
        try:
            with open(s3_config_path, 'r') as f: temp_config_s3 = json.load(f)
            # The config saved in S3 might be the base config (from S2)
            # We need the base config (S2 best) and the aug names
            best_config_stage3 = best_config_stage2 # Base config for S4 is the best from S2
            best_aug_names_stage3 = temp_config_s3.get('data_augmentation_names', ['random_crop', 'horizontal_flip']) # Default if missing
            log_entry = f"Loaded existing Stage 3 best result info from {s3_run_id} (Augs: {best_aug_names_stage3})"; print(log_entry); tuning_log_data.append(log_entry)
            # Update overall best
            s3_val_acc = temp_config_s3.get('achieved_val_accuracy', 0.0)
            # Construct the config corresponding to this run (S2 best + S3 augs)
            config_s3_run = best_config_stage2.copy()
            config_s3_run['data_augmentation_names'] = best_aug_names_stage3
            if s3_val_acc > overall_best_config_info['val_acc']:
                 overall_best_config_info.update({'val_acc': s3_val_acc, 'config': config_s3_run, 'stage_achieved': "Stage 3", 'run_id': s3_run_id})
                 log_entry = f"Updated overall best config from loaded Stage 3: {s3_run_id} (Val Acc: {s3_val_acc:.4f})"; print(log_entry); tuning_log_data.append(log_entry)

        except Exception as e:
            log_entry = f"Warning: Found Stage 3 config file but failed to load: {e}"; print(log_entry); tuning_log_data.append(log_entry)
            best_config_stage3 = None # Reset on failure
            best_aug_names_stage3 = None

    # Check if we have the necessary configs to proceed to Stage 4
    if best_config_stage3 is None or best_aug_names_stage3 is None:
        log_entry = "!!! Cannot proceed to Stage 4: Missing best config/aug info from previous stages. Rerun needed. !!!"
        print(log_entry); tuning_log_data.append(log_entry)
        # Save log and exit
        with open(TUNING_LOG_FILENAME, 'w') as f: f.write("\n".join(tuning_log_data)); return

    # --- Stage 4: Positional Embedding Tuning (Resume/Complete) ---
    stage4_start_time = time.time()
    log_entry = "\n" + "="*15 + f" Stage 4: Positional Embedding Tuning (Resume - {NUM_EPOCHS_TUNE} Epochs) " + "="*15
    print(log_entry); tuning_log_data.append(log_entry)
    tuning_log_data.append(f"Using Best Model Config from Stage 2 ('{best_param_info_name_s2 or 'Unknown'}')")
    tuning_log_data.append(f"Using Best Augmentation from Stage 3 ('{best_aug_names_stage3}')")

    # Check which runs exist, only run missing ones
    pos_embed_types_to_run = []
    s4_run_ids_to_check = {
        '1d_learned': "Stage4_Pos_1d_learned",
        '2d_learned': "Stage4_Pos_2d_learned",
        'sinusoidal': "Stage4_Pos_sinusoidal",
        'no_pos': "Stage4_Pos_no_pos"
    }
    for pe_type, run_id in s4_run_ids_to_check.items():
         # Check if a *model file* exists for this run, indicating it completed
         model_path = os.path.join(MODELS_DIR, f"best_vit_{run_id}.pth")
         config_path = os.path.join(RESULTS_DIR, f"best_config_{run_id}.json")

         if os.path.exists(model_path) and os.path.exists(config_path):
             log_entry = f"Found existing results for {run_id}. Loading accuracy."; print(log_entry); tuning_log_data.append(log_entry)
             # Try loading the config to get accuracy and update overall best if needed
             try:
                 with open(config_path, 'r') as f: temp_config_s4 = json.load(f)
                 s4_val_acc = temp_config_s4.get('achieved_val_accuracy', 0.0)
                 log_entry = f"  Loaded Val Acc: {s4_val_acc:.4f}"; print(log_entry); tuning_log_data.append(log_entry)
                 # Update overall best config tracker if this loaded result is better
                 if s4_val_acc > overall_best_config_info['val_acc']:
                     # Config for this run is best_config_stage3 + this pe_type + best_aug_names_stage3
                     config_s4_run = best_config_stage3.copy()
                     config_s4_run['pos_embed_type'] = pe_type
                     config_s4_run['data_augmentation_names'] = best_aug_names_stage3
                     overall_best_config_info.update({'val_acc': s4_val_acc, 'config': config_s4_run, 'stage_achieved': "Stage 4", 'run_id': run_id})
                     log_entry = f"Updated overall best config from loaded Stage 4: {run_id} (Val Acc: {s4_val_acc:.4f})"; print(log_entry); tuning_log_data.append(log_entry)
             except Exception as e:
                 log_entry = f"  Warning: Failed to load config {config_path}: {e}"; print(log_entry); tuning_log_data.append(log_entry)
         else:
             # Only add to run list if results don't exist
             if pe_type in ['2d_learned', 'no_pos']: # Only run the ones missing based on logs
                 log_entry = f"Results for {run_id} not found. Adding to run queue."; print(log_entry); tuning_log_data.append(log_entry)
                 pos_embed_types_to_run.append(pe_type)
             else:
                  log_entry = f"Skipping {run_id} as results exist."; print(log_entry); tuning_log_data.append(log_entry)


    config_pos_base = best_config_stage3.copy() # Best arch params from S2
    aug_names_pos_base = best_aug_names_stage3 # Best augs from S3

    best_pos_info = {'val_acc': 0.0, 'name': None, 'config': None, 'state_dict': None, 'history': None, 'id': None, 'augmentation_names': None}
    # Initialize best_pos_info with the best loaded S4 result if any
    # (This assumes you only save the *best* of stage 4, not necessarily the last run)
    loaded_best_s4_acc = 0
    for pe_type, run_id in s4_run_ids_to_check.items():
        config_path = os.path.join(RESULTS_DIR, f"best_config_{run_id}.json")
        if os.path.exists(config_path):
            try:
                 with open(config_path, 'r') as f: temp_config_s4 = json.load(f)
                 s4_val_acc = temp_config_s4.get('achieved_val_accuracy', 0.0)
                 if s4_val_acc > loaded_best_s4_acc:
                     loaded_best_s4_acc = s4_val_acc
                     # We might need to load the state_dict too if we want to save only the single best
                     # For simplicity now, just track the accuracy
                     best_pos_info['val_acc'] = s4_val_acc
                     best_pos_info['name'] = pe_type
                     # ... potentially load other fields if needed for save_stage_results
            except Exception: pass # Ignore errors loading existing configs here


    # Now run only for sinusoidal 
    
    pe_type = 'sinusoidal'
    run_id = f"Stage4_Pos_{pe_type}" # s4_run_ids_to_check[pe_type]
    print(f"\n--- Testing {run_id} ---")
    config_pe = config_pos_base.copy(); config_pe['pos_embed_type'] = pe_type

    # *** IMPORTANT : Ensure the VisionTransformer class has the fix before this call ***
    state_dict_run, history_run, test_acc_run, best_val_acc_run = run_training_for_tuning(
        config_pe, num_epochs=NUM_EPOCHS_TUNE, data_augmentation_names=aug_names_pos_base, experiment_name=run_id
    )
    log_msg = f"{run_id}: Best Val Acc = {best_val_acc_run:.4f}, Final Test Acc = {test_acc_run:.4f}"
    print(log_msg); tuning_log_data.append(log_msg)

    # Update best for this stage (compare against potentially loaded best)
    if state_dict_run is not None and best_val_acc_run > best_pos_info['val_acc']:
        best_pos_info = {'val_acc': best_val_acc_run, 'test_acc': test_acc_run, 'name': pe_type, 'config': config_pe,
                            'state_dict': state_dict_run, 'history': history_run, 'id': run_id, 'augmentation_names': aug_names_pos_base}
        print(f"*** New best for Stage 4: {run_id} (Val Acc: {best_pos_info['val_acc']:.4f}) ***")
        # Save results for this specific run immediately
        save_stage_results("Stage4_Pos", best_pos_info, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, tuning_log_data)


    # Update overall best *configuration* tracker
    if state_dict_run is not None and best_val_acc_run > overall_best_config_info['val_acc']:
        current_best_config = config_pe.copy()
        current_best_config['data_augmentation_names'] = aug_names_pos_base
        overall_best_config_info = {'val_acc': best_val_acc_run, 'config': current_best_config,
                                    'stage_achieved': "Stage 4 (Pos Embed)", 'run_id': run_id}
        print(f"*** New Overall Best CONFIG Found! (From {run_id}, Val Acc: {overall_best_config_info['val_acc']:.4f}) ***")

    stage4_time = time.time() - stage4_start_time
    log_entry = f"\n--- Stage 4 Result (Duration for new runs: {stage4_time/60:.2f} min): Best Pos Embed overall in S4 = '{best_pos_info.get('name', 'None')}' (Val Acc: {best_pos_info.get('val_acc', 0.0):.4f}) ---"
    print(log_entry); tuning_log_data.append(log_entry)
    # Note: save_stage_results might have been called multiple times if a new best was found during the loop



if __name__ == "__main__":
     # --- IMPORTANT: Apply the fix to the VisionTransformer class ---
     # --- Ensure RESULTS_DIR points to 'results_vit_tuned' ---
     # --- Ensure FINAL_RESULTS_DIR points to 'results_vit_concat_aug_final_train' ---
    main_sequential_tuning()