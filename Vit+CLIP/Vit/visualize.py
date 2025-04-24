# -*- coding: utf-8 -*-
"""
visualize_standard_vit.py

Visualize attention maps and positional embeddings for a trained Standard ViT on CIFAR-10,
based on the tuning results from the sequential tuning script (train.py).

Performs Tasks:
- Visualize last layer attention (Avg + Heads) using the final trained model.
- Visualize average attention per layer using the final trained model.
- Visualize Attention Rollout using the final trained model.
- Visualize Positional Embedding Similarity for models saved during tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from PIL import Image
import math
import copy

# --- Constants and Helper Functions ---
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# <<< --- Standard ViT Model Definition (Copied from train.py) --- >>>
# Ensure the VisionTransformer class includes the fix for prepare_tokens!
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches=(img_size//patch_size)**2; self.img_size=img_size; self.patch_size=patch_size
        self.num_patches=num_patches; self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        B,C,H,W=x.shape
        if H!=self.img_size or W!=self.img_size: print(f"Warning: Input image size ({H}*{W}) doesn't match model expected size ({self.img_size}*{self.img_size}).")
        return self.proj(x).flatten(2).transpose(1,2)

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self,dim,attn_dropout=0.): super().__init__(); self.sqrt_dim=np.sqrt(dim); self.attn_dropout=nn.Dropout(attn_dropout)
    def forward(self,q,k,v,mask=None):
        attn=(q@k.transpose(-2,-1))/self.sqrt_dim
        if mask is not None: attn=attn.masked_fill(mask==0,-1e9)
        attn=F.softmax(attn,dim=-1); attn=self.attn_dropout(attn); output=attn@v; return output,attn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(); self.num_heads=num_heads; head_dim=embed_dim//num_heads
        if head_dim*num_heads!=embed_dim: raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.scale=head_dim**-0.5
        self.qkv=nn.Linear(embed_dim,embed_dim*3,bias=qkv_bias)
        self.attn=ScaledDotProductAttention(head_dim,attn_dropout=attn_drop)
        self.proj=nn.Linear(embed_dim,embed_dim); self.proj_drop=nn.Dropout(proj_drop)
    def forward(self,x):
        B,N,C=x.shape; qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]; x_attn,attn=self.attn(q,k,v)
        x_attn=x_attn.transpose(1,2).reshape(B,N,C); x_out=self.proj(x_attn); x_out=self.proj_drop(x_out)
        return x_out,attn

class MLP(nn.Module):
    """ MLP as used in Vision Transformer """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(); out_features=out_features or in_features; hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features); self.act=act_layer(); self.fc2=nn.Linear(hidden_features,out_features); self.drop=nn.Dropout(drop)
    def forward(self,x): x=self.fc1(x);x=self.act(x);x=self.drop(x);x=self.fc2(x);x=self.drop(x); return x

class TransformerEncoderLayer(nn.Module):
    """ Standard Transformer Encoder Layer """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(); self.norm1=norm_layer(embed_dim)
        self.attn=MultiHeadAttention(embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop)
        self.norm2=norm_layer(embed_dim); mlp_hidden_dim=int(embed_dim*mlp_ratio)
        self.mlp=MLP(in_features=embed_dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)
    def forward(self,x):
        identity1=x; x_norm1=self.norm1(x); attn_output,attn_weights=self.attn(x_norm1)
        x=identity1+attn_output; identity2=x; x_norm2=self.norm2(x); mlp_output=self.mlp(x_norm2); x=identity2+mlp_output
        return x,attn_weights

def get_sinusoid_encoding_table(n_position, d_hid):
    """ Sinusoid position encoding table """
    def get_position_angle_vec(p): return[p/np.power(10000,2*(h//2)/d_hid) for h in range(d_hid)]
    t=np.array([get_position_angle_vec(p) for p in range(n_position)]); t[:,0::2]=np.sin(t[:,0::2]); t[:,1::2]=np.cos(t[:,1::2])
    return torch.FloatTensor(t).unsqueeze(0)

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, pos_embed_type='1d_learned'):
        super().__init__(); self.num_classes=num_classes; self.num_features=self.embed_dim=embed_dim
        self.patch_size=patch_size; self.pos_embed_type=pos_embed_type; self.img_size=img_size
        self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches; self.num_patches=num_patches
        self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim)); self.pos_drop=nn.Dropout(p=drop_rate); self.seq_len=num_patches+1
        # Positional Embedding Logic
        if pos_embed_type=='no_pos': self.pos_embed=None; print("Using no positional embedding.")
        elif pos_embed_type=='1d_learned': self.pos_embed=nn.Parameter(torch.zeros(1,self.seq_len,embed_dim)); nn.init.trunc_normal_(self.pos_embed,std=.02); print("Using 1D learned positional embedding.")
        elif pos_embed_type=='2d_learned':
            h_patches,w_patches=img_size//patch_size,img_size//patch_size
            if embed_dim%2!=0: raise ValueError("Embed dim must be even for 2D learned PE split.")
            embed_dim_spatial=embed_dim//2
            self.pos_embed_row=nn.Parameter(torch.zeros(1,h_patches,embed_dim_spatial)); self.pos_embed_col=nn.Parameter(torch.zeros(1,w_patches,embed_dim_spatial))
            self.pos_embed_cls=nn.Parameter(torch.zeros(1,1,embed_dim)); nn.init.trunc_normal_(self.pos_embed_row,std=.02); nn.init.trunc_normal_(self.pos_embed_col,std=.02); nn.init.trunc_normal_(self.pos_embed_cls,std=.02); print("Using 2D learned positional embedding.")
        elif pos_embed_type=='sinusoidal': self.pos_embed=nn.Parameter(get_sinusoid_encoding_table(self.seq_len,embed_dim),requires_grad=False); print("Using sinusoidal positional embedding.")
        else: raise ValueError(f"Unknown PE type: {pos_embed_type}")
        nn.init.trunc_normal_(self.cls_token,std=.02)
        self.blocks=nn.ModuleList([TransformerEncoderLayer(embed_dim,num_heads,mlp_ratio,qkv_bias,drop_rate,attn_drop_rate,norm_layer=norm_layer) for _ in range(depth)])
        self.norm=norm_layer(embed_dim); self.head=nn.Linear(embed_dim,num_classes); self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(m,nn.Linear): nn.init.trunc_normal_(m.weight,std=.02); nn.init.constant_(m.bias,0) if hasattr(m,'bias')and m.bias is not None else None
        elif isinstance(m,nn.LayerNorm): nn.init.constant_(m.bias,0); nn.init.constant_(m.weight,1.0)
        elif isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu'); nn.init.constant_(m.bias,0) if hasattr(m,'bias')and m.bias is not None else None
    def prepare_tokens(self,x): # Use the fixed version
        B=x.shape[0];x=self.patch_embed(x);cls_tokens=self.cls_token.expand(B,-1,-1);x=torch.cat((cls_tokens,x),dim=1)
        if self.pos_embed_type=='1d_learned' or self.pos_embed_type=='sinusoidal':
            if self.pos_embed is not None: x=x+self.pos_embed
        elif self.pos_embed_type=='2d_learned':
            h_patches,w_patches=self.img_size//self.patch_size,self.img_size//self.patch_size
            if self.embed_dim%2!=0: raise ValueError("Embed dim must be even for 2D learned PE split.")
            embed_dim_spatial=self.embed_dim//2
            pos_embed_row=self.pos_embed_row.unsqueeze(2).expand(-1,-1,w_patches,-1); pos_embed_col=self.pos_embed_col.unsqueeze(1).expand(-1,h_patches,-1,-1)
            pos_embed_patches=torch.cat((pos_embed_row,pos_embed_col),dim=-1).flatten(1,2); full_pos_embed=torch.cat((self.pos_embed_cls,pos_embed_patches),dim=1)
            x=x+full_pos_embed
        x=self.pos_drop(x); return x
    def forward(self,x):
        x=self.prepare_tokens(x)
        for blk in self.blocks: x,_=blk(x)
        x=self.norm(x); return self.head(x[:,0])
    @torch.no_grad()
    def get_last_selfattention(self, x):
        x=self.prepare_tokens(x); attn_weights=None
        for i,blk in enumerate(self.blocks):
            x,attn_map=blk(x)
            if i==len(self.blocks)-1: attn_weights=attn_map
        return attn_weights
    @torch.no_grad()
    def get_intermediate_layers_attention(self, x):
        x=self.prepare_tokens(x); attentions=[]
        for blk in self.blocks: x,attn_map=blk(x); attentions.append(attn_map)
        return attentions
# --- End of VisionTransformer Definition ---


def load_model_from_config(config_path, model_path, device):
    """Loads a Standard ViT model using config and state dict."""
    print(f"Loading config from: {config_path}")
    print(f"Loading model weights from: {model_path}")
    if not os.path.exists(config_path): print(f"Error: Config file not found: {config_path}"); return None, None
    if not os.path.exists(model_path): print(f"Error: Model file not found: {model_path}"); return None, None
    try:
        with open(config_path, 'r') as f: config = json.load(f)
        # Extract standard ViT arguments
        model_kwargs = {
            'img_size': config.get('img_size', 32),
            'patch_size': config['patch_size'],
            'in_chans': config.get('in_chans', 3),
            'num_classes': config.get('num_classes', 10),
            'embed_dim': config['embed_dim'],
            'depth': config['depth'],
            'num_heads': config['num_heads'],
            'mlp_ratio': config.get('mlp_ratio', 4.0),
            'qkv_bias': config.get('qkv_bias', True),
            'drop_rate': 0.0, # Set dropouts to 0 for eval
            'attn_drop_rate': 0.0,
            'norm_layer': nn.LayerNorm,
            'pos_embed_type': config.get('pos_embed_type', '1d_learned')
        }
        print("Model Kwargs loaded:", {k:v for k,v in model_kwargs.items() if k!='norm_layer'})
        model = VisionTransformer(**model_kwargs)
        # Load state dict using weights_only=True for security
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.to(device); model.eval()
        print("Model loaded successfully.")
        # Clean config dict loaded from tuning stages if necessary
        config_cleaned = config.copy()
        config_cleaned.pop('achieved_val_accuracy', None)
        config_cleaned.pop('final_run_best_val_accuracy', None)
        config_cleaned.pop('final_run_test_accuracy', None)
        config_cleaned.pop('tuning_best_config_from_run_id', None)
        config_cleaned.pop('tuning_best_config_val_acc', None)
        return model, config_cleaned # Return the cleaned config
    except KeyError as e: print(f"Error: Missing required key {e} in config file {config_path}"); return None, None
    except Exception as e: print(f"Error loading model: {e}"); return None, None

def get_cifar10_test_data(config, data_dir='./data_cifar', indices=None):
    """Loads CIFAR-10 test data with appropriate transforms."""
    if config is None: print("Error: Cannot load data without config."); return None
    img_size = config.get('img_size', 32)
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])
    try:
        full_testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        if indices:
            valid_indices = [i for i in indices if 0 <= i < len(full_testset)]
            if len(valid_indices) < len(indices): print(f"Warning: Some indices out of bounds (0-{len(full_testset)-1}). Using valid: {valid_indices}")
            if not valid_indices: print("Error: No valid indices provided."); return None
            testset = Subset(full_testset, valid_indices)
        else: testset = full_testset
        print(f"Loaded CIFAR-10 test set. Size: {len(testset)}")
        return testset
    except Exception as e: print(f"Error loading CIFAR-10 data: {e}"); return None

def tensor_to_pil(tensor_img):
    """Converts a normalized PyTorch tensor back to a PIL image."""
    inv_normalize = transforms.Normalize(mean=[-m/s for m,s in zip(CIFAR_MEAN,CIFAR_STD)], std=[1/s for s in CIFAR_STD])
    img_deprocessed = inv_normalize(tensor_img.cpu()).clamp(0,1); return transforms.ToPILImage()(img_deprocessed)

# --- Visualization Functions ---

def visualize_last_layer_attention(model, dataset, config, output_dir):
    """Visualizes attention from CLS to patches for the last layer."""
    print("\n--- Visualizing Last Layer Attention ---")
    os.makedirs(output_dir, exist_ok=True)
    if dataset is None: print("Dataset not loaded. Skipping."); return
    try:
        img_size = config.get('img_size', 32); patch_size = config['patch_size']; num_heads = config['num_heads']; grid_size = img_size // patch_size
    except KeyError as e: print(f"Missing key {e} in config. Skipping last layer viz."); return

    num_images = len(dataset)
    print(f"Visualizing last layer attention for {num_images} images...")
    for idx in range(num_images):
        try:
            img_tensor, label_idx = dataset[idx]; label_name = CIFAR_CLASSES[label_idx]
            img_pil = tensor_to_pil(img_tensor); img_tensor_batch = img_tensor.unsqueeze(0).to(DEVICE)
            att_mat = model.get_last_selfattention(img_tensor_batch)
            if att_mat is None: print(f"Skipping img {idx}: No attention map."); continue

            att_mat = att_mat.squeeze(0).cpu().numpy(); cls_to_patch_att = att_mat[:,0,1:] #[H, N]
            avg_cls_to_patch_att = cls_to_patch_att.mean(axis=0); avg_att_map = avg_cls_to_patch_att.reshape(grid_size, grid_size)
            head_att_maps = cls_to_patch_att.reshape(num_heads, grid_size, grid_size)

            # Plot average
            fig_avg, ax_avg = plt.subplots(1,2,figsize=(8,4)); fig_avg.suptitle(f'Avg CLS Attn (Last Layer) - Img {idx} ({label_name})', fontsize=12)
            ax_avg[0].imshow(img_pil); ax_avg[0].set_title("Original"); ax_avg[0].axis('off')
            im=ax_avg[1].imshow(avg_att_map,cmap='viridis',interpolation='nearest'); ax_avg[1].set_title("Avg Attn (CLS->Patches)"); ax_avg[1].axis('off'); fig_avg.colorbar(im,ax=ax_avg[1])
            plt.tight_layout(rect=[0,0.03,1,0.95]); save_path_avg=os.path.join(output_dir,f"last_layer_avg_att_img_{idx}_{label_name}.png"); plt.savefig(save_path_avg); plt.close(fig_avg)

            # Plot heads
            num_cols=4; num_rows=math.ceil((num_heads+1)/num_cols)
            fig_heads,axes_heads=plt.subplots(num_rows,num_cols,figsize=(num_cols*3,num_rows*3)); fig_heads.suptitle(f'Head-wise CLS Attn (Last Layer) - Img {idx} ({label_name})',fontsize=12)
            axes_flat=axes_heads.flatten(); axes_flat[0].imshow(img_pil); axes_flat[0].set_title("Original"); axes_flat[0].axis('off')
            for h in range(num_heads):
                if h+1 < len(axes_flat): ax=axes_flat[h+1]; im=ax.imshow(head_att_maps[h],cmap='viridis',interpolation='nearest'); ax.set_title(f"Head {h+1}"); ax.axis('off')
            for i in range(num_heads+1,len(axes_flat)): axes_flat[i].axis('off')
            plt.tight_layout(rect=[0,0.03,1,0.95]); save_path_heads=os.path.join(output_dir,f"last_layer_heads_att_img_{idx}_{label_name}.png"); plt.savefig(save_path_heads); plt.close(fig_heads)

            # Print progress minimally
            if (idx + 1) % 50 == 0 or idx == num_images - 1:
                 print(f"  Processed last layer attention for image {idx+1}/{num_images}")

        except Exception as e: print(f"Error processing img {idx} (last layer): {e}"); continue
    print("Finished visualizing last layer attention.")

def visualize_all_layers_attention(model, dataset, config, output_dir):
    """Visualizes average CLS->patch attention for every layer."""
    print("\n--- Visualizing All Layers Attention (Avg Heads) ---")
    os.makedirs(output_dir, exist_ok=True)
    if dataset is None: print("Dataset not loaded. Skipping."); return
    try:
        img_size = config.get('img_size', 32); patch_size = config['patch_size']; depth = config['depth']; grid_size = img_size//patch_size
    except KeyError as e: print(f"Missing key {e} in config. Skipping all layers viz."); return

    num_images = len(dataset)
    print(f"Visualizing all layers attention for {num_images} images...")
    for idx in range(num_images):
        try:
            img_tensor, label_idx = dataset[idx]; label_name = CIFAR_CLASSES[label_idx]
            img_pil = tensor_to_pil(img_tensor); img_tensor_batch = img_tensor.unsqueeze(0).to(DEVICE)
            intermediate_attentions = model.get_intermediate_layers_attention(img_tensor_batch)
            if not intermediate_attentions or any(a is None for a in intermediate_attentions): print(f"Skipping img {idx}: No intermediate attn maps."); continue
            if len(intermediate_attentions)!=depth: print(f"Warning: Retrieved {len(intermediate_attentions)} attn maps, expected {depth}.")

            num_cols=4; num_rows=math.ceil((depth+1)/num_cols); fig,axes=plt.subplots(num_rows,num_cols,figsize=(num_cols*3,num_rows*3)); fig.suptitle(f'Avg CLS Attn per Layer - Img {idx} ({label_name})',fontsize=12)
            axes_flat=axes.flatten(); axes_flat[0].imshow(img_pil); axes_flat[0].set_title("Original"); axes_flat[0].axis('off')
            valid_att_count = 0
            for layer_idx, att_mat in enumerate(intermediate_attentions):
                if att_mat is None: continue
                if layer_idx+1 >= len(axes_flat): break
                att_mat_np = att_mat.squeeze(0).cpu().numpy(); cls_to_patch_att = att_mat_np[:,0,1:]
                avg_cls_to_patch_att = cls_to_patch_att.mean(axis=0); avg_att_map = avg_cls_to_patch_att.reshape(grid_size, grid_size)
                ax = axes_flat[layer_idx+1]; im = ax.imshow(avg_att_map,cmap='viridis',interpolation='nearest'); ax.set_title(f"Layer {layer_idx+1}"); ax.axis('off'); valid_att_count+=1
            for i in range(valid_att_count+1,len(axes_flat)): axes_flat[i].axis('off')
            plt.tight_layout(rect=[0,0.03,1,0.95]); save_path=os.path.join(output_dir,f"all_layers_avg_att_img_{idx}_{label_name}.png"); plt.savefig(save_path); plt.close(fig)

            if (idx + 1) % 50 == 0 or idx == num_images - 1:
                 print(f"  Processed all layers attention for image {idx+1}/{num_images}")

        except Exception as e: print(f"Error processing img {idx} (all layers): {e}"); continue
    print("Finished visualizing all layers attention.")

class AttentionRollout:
    """ Calculates Attention Rollout as described in Abnar & Zuidema, 2020."""
    def __init__(self,model,head_fusion="mean"): self.model=model; self.head_fusion=head_fusion; self.model.eval()
    def __call__(self,input_tensor):
        attentions = self.model.get_intermediate_layers_attention(input_tensor)
        if not attentions or any(a is None for a in attentions): print("Error: Rollout failed, missing attn maps."); return None
        device=input_tensor.device; attentions=[a.to(device) for a in attentions if a is not None]
        if not attentions: return None
        B,H,N_plus_1,_=attentions[0].shape; result=torch.eye(N_plus_1,device=device).unsqueeze(0).expand(B,-1,-1) # [B, N+1, N+1]
        # Iterate through layers, multiplying attention matrices
        for attn_layer in attentions: # attn_layer is [B, H, N+1, N+1]
            # Fuse heads
            if self.head_fusion=="mean": attn_fused=attn_layer.mean(axis=1) # [B, N+1, N+1]
            elif self.head_fusion=="max": attn_fused=attn_layer.max(axis=1)[0]
            elif self.head_fusion=="min": attn_fused=attn_layer.min(axis=1)[0]
            else: raise ValueError("Invalid head_fusion")
            # Add residual connection (identity matrix) as per rollout paper eq. 5
            I=torch.eye(N_plus_1,device=device).unsqueeze(0).expand(B,-1,-1)
            a = attn_fused + I
            # Normalize (row-wise stochastic matrix) - paper eq. 6
            a = a / (a.sum(axis=-1, keepdim=True) + 1e-9) # Add epsilon for stability
            # Matrix multiply with previous result - paper eq. 7
            result = a @ result
        # result matrix shape: [B, N+1, N+1]
        # result[b, i, j] = attention flow from token j to token i in batch b
        return result

def visualize_attention_rollout(model, dataset, config, output_dir):
    """Visualizes Attention Rollout maps."""
    print("\n--- Visualizing Attention Rollout ---")
    os.makedirs(output_dir, exist_ok=True)
    if dataset is None: print("Dataset not loaded. Skipping."); return
    try:
        img_size = config.get('img_size', 32); patch_size = config['patch_size']; grid_size = img_size // patch_size
    except KeyError as e: print(f"Missing key {e} in config. Skipping rollout viz."); return
    rollout_calculator = AttentionRollout(model, head_fusion="mean")

    num_images = len(dataset)
    print(f"Visualizing attention rollout for {num_images} images...")
    for idx in range(num_images):
        try:
            img_tensor, label_idx = dataset[idx]; label_name = CIFAR_CLASSES[label_idx]
            img_pil = tensor_to_pil(img_tensor); img_tensor_batch = img_tensor.unsqueeze(0).to(DEVICE)
            rollout_matrix = rollout_calculator(img_tensor_batch)
            if rollout_matrix is None: print(f"Skipping img {idx}: Rollout failed."); continue

            rollout_matrix_np = rollout_matrix.squeeze(0).cpu().numpy() # [N+1, N+1]
            # Visualize flow INTO CLS token FROM patch tokens (row 0, columns 1 to N)
            rollout_cls_in = rollout_matrix_np[0, 1:] # Shape [N]
            # Check for NaN/inf values before reshaping
            if np.isnan(rollout_cls_in).any() or np.isinf(rollout_cls_in).any():
                print(f"Skipping img {idx}: NaN/Inf detected in rollout matrix.")
                continue

            vis_map_reshaped = rollout_cls_in.reshape(grid_size, grid_size)

            fig,ax=plt.subplots(1,2,figsize=(8,4)); fig.suptitle(f'Attention Rollout - Img {idx} ({label_name})',fontsize=12)
            ax[0].imshow(img_pil);ax[0].set_title("Original");ax[0].axis('off'); im=ax[1].imshow(vis_map_reshaped,cmap='viridis',interpolation='nearest')
            ax[1].set_title("Rollout (Patches -> CLS)");ax[1].axis('off'); fig.colorbar(im,ax=ax[1])
            plt.tight_layout(rect=[0,0.03,1,0.95]); save_path=os.path.join(output_dir,f"rollout_att_img_{idx}_{label_name}.png"); plt.savefig(save_path); plt.close(fig)

            if (idx + 1) % 50 == 0 or idx == num_images - 1:
                 print(f"  Processed rollout for image {idx+1}/{num_images}")

        except Exception as e: print(f"Error processing img {idx} (rollout): {e}"); continue
    print("Finished visualizing attention rollout.")


def visualize_pos_embed_similarity(model, config, output_dir):
    """Visualizes the similarity matrix of positional embeddings."""
    print("\n--- Visualizing Positional Embedding Similarity ---")
    os.makedirs(output_dir, exist_ok=True)
    if model is None: print("Model not loaded. Skipping PE viz."); return
    pe_type = config.get('pos_embed_type', 'unknown'); print(f"Attempting PE visualization for type: {pe_type}")
    pos_embed = None
    # --- Logic to get or reconstruct PE based on type ---
    if pe_type == '1d_learned' or pe_type == 'sinusoidal':
        if hasattr(model,'pos_embed') and model.pos_embed is not None: pos_embed=model.pos_embed.squeeze(0).detach().cpu()
        else: print(f"Skipping: Type {pe_type} but 'pos_embed' not found/None."); return
    elif pe_type == '2d_learned':
        if hasattr(model,'pos_embed_row') and hasattr(model,'pos_embed_col') and hasattr(model,'pos_embed_cls'):
            try:
                # Ensure necessary attributes are present for reconstruction
                if not all(hasattr(model, attr) for attr in ['img_size', 'patch_size', 'embed_dim']):
                    print("Skipping: Missing model attributes (img_size/patch_size/embed_dim) needed for 2D PE reconstruction."); return
                if model.embed_dim % 2 != 0: print("Skipping: embed_dim must be even for 2D PE."); return

                h_p=model.img_size//model.patch_size; w_p=model.img_size//model.patch_size; emb_s=model.embed_dim//2
                p_r=model.pos_embed_row.detach().cpu().unsqueeze(2).expand(-1,-1,w_p,-1); p_c=model.pos_embed_col.detach().cpu().unsqueeze(1).expand(-1,h_p,-1,-1)
                p_p=torch.cat((p_r,p_c),dim=-1).flatten(1,2); pos_embed=torch.cat((model.pos_embed_cls.detach().cpu(),p_p),dim=1).squeeze(0)
            except Exception as e: print(f"Skipping: Error reconstructing 2D PE: {e}"); return
        else: print("Skipping: Type 2d_learned but missing PE attributes (row/col/cls)."); return
    elif pe_type=='no_pos': print("Skipping: Type is 'no_pos'."); return
    else: print(f"Skipping: Unknown PE type '{pe_type}'."); return

    # --- Calculate and Plot Similarity ---
    # Ensure pos_embed is not None before proceeding
    if pos_embed is None: print(f"Skipping: Positional embedding could not be obtained for type {pe_type}."); return

    similarity_matrix = torch.matmul(pos_embed, pos_embed.T).numpy() # [N+1, N+1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 7)); im = ax.imshow(similarity_matrix, cmap='viridis')
    ax.set_title(f"Positional Embedding Similarity (Dot Product)\nType: {pe_type}"); ax.set_xlabel("Token Index (0=CLS, 1..N=Patches)"); ax.set_ylabel("Token Index")
    fig.colorbar(im, ax=ax, label="Dot Product Similarity"); plt.tight_layout()
    save_path = os.path.join(output_dir, f"pos_embed_similarity_{pe_type}.png"); plt.savefig(save_path); plt.close(fig)
    print(f"Saved PE similarity map: {os.path.basename(save_path)}")

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standard ViT Visualization Script')
    # --- Default Paths ---
    # DEFAULT_FINAL_RESULTS_DIR = "results_vit_concat_aug_final_train" # No longer primary source for viz model
    DEFAULT_TUNING_RESULTS_DIR = "results_vit_tuned" # Location of tuning stage models

    # --- Arguments for the Model Used for Attention Viz ---
    # POINT THESE TO THE BEST TUNING MODEL (SINUSOIDAL)
    parser.add_argument('--attn_config_path', type=str,
                        default=os.path.join(DEFAULT_TUNING_RESULTS_DIR, "best_config_Stage4_Pos_sinusoidal.json"),
                        help='Path to the ViT JSON config file for attention visualization (best tuning model)')
    parser.add_argument('--attn_model_path', type=str,
                        default=os.path.join(DEFAULT_TUNING_RESULTS_DIR, "models", "best_vit_Stage4_Pos_sinusoidal.pth"),
                        help='Path to the ViT .pth state_dict file for attention visualization (best tuning model)')

    # --- Arguments for Specific PE Type Models (PE Viz Comparison) ---
    # These still point to the models saved during Stage 4 tuning
    parser.add_argument('--learned1d_config_path', type=str,
                        default=os.path.join(DEFAULT_TUNING_RESULTS_DIR, "best_config_Stage4_Pos_1d_learned.json"), # Assumes this exists from first attempt
                        help=f'Path to 1D learned PE ViT JSON config file (expected in {DEFAULT_TUNING_RESULTS_DIR})')
    parser.add_argument('--learned1d_model_path', type=str,
                        default=os.path.join(DEFAULT_TUNING_RESULTS_DIR, "models", "best_vit_Stage4_Pos_1d_learned.pth"), # Assumes this exists
                        help=f'Path to 1D learned PE ViT .pth state_dict file (expected in {DEFAULT_TUNING_RESULTS_DIR}/models)')
    parser.add_argument('--learned2d_config_path', type=str,
                        default=os.path.join(DEFAULT_TUNING_RESULTS_DIR, "best_config_Stage4_Pos_2d_learned.json"),
                        help=f'Path to 2D learned PE ViT JSON config file (expected in {DEFAULT_TUNING_RESULTS_DIR})')
    parser.add_argument('--learned2d_model_path', type=str,
                        default=os.path.join(DEFAULT_TUNING_RESULTS_DIR, "models", "best_vit_Stage4_Pos_2d_learned.pth"),
                        help=f'Path to 2D learned PE ViT .pth state_dict file (expected in {DEFAULT_TUNING_RESULTS_DIR}/models)')
    parser.add_argument('--sinusoidal_config_path', type=str,
                         default=os.path.join(DEFAULT_TUNING_RESULTS_DIR, "best_config_Stage4_Pos_sinusoidal.json"),
                         help=f'Path to Sinusoidal PE ViT JSON config file (expected in {DEFAULT_TUNING_RESULTS_DIR})')
    parser.add_argument('--sinusoidal_model_path', type=str,
                         default=os.path.join(DEFAULT_TUNING_RESULTS_DIR, "models", "best_vit_Stage4_Pos_sinusoidal.pth"),
                         help=f'Path to Sinusoidal PE ViT .pth state_dict file (expected in {DEFAULT_TUNING_RESULTS_DIR}/models)')

    # --- Other Arguments ---
    parser.add_argument('--data_dir', type=str, default='./data_cifar', help='Directory for CIFAR-10 data')
    parser.add_argument('--output_dir', type=str, default='./standard_vit_visualizations', help='Directory to save output plots')
    parser.add_argument('--img_indices', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5], help='Indices of CIFAR-10 test images to visualize for attention maps')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model for Attention Viz (Using Best Tuning Model) ---
    print("\n" + "="*10 + " Loading Best Standard ViT Model from Tuning (for Attention Viz) " + "="*10)
    # Use the arguments specifically for attention visualization
    attn_model, attn_config = load_model_from_config(args.attn_config_path, args.attn_model_path, DEVICE)

    if attn_model is None or attn_config is None:
         print("!!! Failed to load model/config for attention visualization. Cannot perform attention visualizations. !!!")
         print(f" Checked paths:\n  Config: {args.attn_config_path}\n  Weights: {args.attn_model_path}")
    else:
        print(f"--- Using model with PE Type: {attn_config.get('pos_embed_type', 'N/A')} for attention maps ---")
        # Load only the subset of data needed for attention visualization
        test_dataset_attn = get_cifar10_test_data(attn_config, args.data_dir, indices=args.img_indices)
        if test_dataset_attn:
            # --- Run Attention Visualizations ---
            attention_output_dir = os.path.join(args.output_dir, "attention_maps_best_tuning_model") # Changed subdir name
            print(f"\nSaving attention visualizations to: {attention_output_dir}")
            visualize_last_layer_attention(attn_model, test_dataset_attn, attn_config, attention_output_dir)
            visualize_all_layers_attention(attn_model, test_dataset_attn, attn_config, attention_output_dir)
            visualize_attention_rollout(attn_model, test_dataset_attn, attn_config, attention_output_dir)
        else:
             print("Failed to load dataset for attention visualization.")
        del attn_model, attn_config, test_dataset_attn # Free memory
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Load Models for PE Visualization ---
    # (This part remains the same, loading models from Stage 4 tuning)
    pe_output_dir = os.path.join(args.output_dir, "pos_embedding_comparison")
    models_for_pe = {
        "1d_learned": (args.learned1d_config_path, args.learned1d_model_path),
        "2d_learned": (args.learned2d_config_path, args.learned2d_model_path),
        "sinusoidal": (args.sinusoidal_config_path, args.sinusoidal_model_path),
    }

    print("\n" + "="*10 + " Loading Models for Positional Embedding Visualization " + "="*10)
    for pe_type, (config_path, model_path) in models_for_pe.items():
        print(f"\n--- Processing PE Type: {pe_type} ---")
        model, config = load_model_from_config(config_path, model_path, DEVICE)
        if model and config:
            loaded_pe_type = config.get('pos_embed_type')
            if loaded_pe_type != pe_type:
                 print(f"  Warning: Loaded config PE type '{loaded_pe_type}' != expected '{pe_type}'. Visualizing based on loaded type.")
            visualize_pos_embed_similarity(model, config, pe_output_dir)
            del model, config
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        else:
            print(f"  Skipping PE visualization for type '{pe_type}' as model/config failed to load.")

    print(f"\nVisualizations saved in: {args.output_dir}")
    print("Script finished.")