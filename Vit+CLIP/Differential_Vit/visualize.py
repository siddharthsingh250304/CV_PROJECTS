# -*- coding: utf-8 -*-
"""
visualize_diff_vit.py

Visualize attention maps and positional embeddings for a trained Diff-ViT on CIFAR-10.

Performs Tasks 1.4.2, 1.4.3, 1.4.4 from the description.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F # Import F
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
import copy # Import copy for potential deep copies if needed

# --- Constants and Helper Functions ---
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# <<< --- PASTE YOUR VisionTransformer AND DEPENDENCIES HERE --- >>>
# Make sure this includes:
# - PatchEmbed
# - MLP
# - get_sinusoid_encoding_table
# - MultiHeadAttention (even if unused in final model, needed for vis)
# - MultiHeadDifferentialAttention
# - TransformerEncoderLayer
# - VisionTransformer (the version from tune_diff_vit.py)

# --- Placeholder Definitions (REPLACE WITH YOUR ACTUAL CODE) ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size=img_size; self.patch_size=patch_size; self.num_patches=(img_size//patch_size)**2
        self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)
    def forward(self, x):
        B,C,H,W=x.shape
        if H!=self.img_size or W!=self.img_size: x=F.interpolate(x,size=(self.img_size,self.img_size),mode='bilinear',align_corners=False)
        return self.proj(x).flatten(2).transpose(1,2)
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__();out_features=out_features or in_features;hidden_features=hidden_features or in_features
        self.fc1=nn.Linear(in_features,hidden_features);self.act=act_layer();self.fc2=nn.Linear(hidden_features,out_features);self.drop=nn.Dropout(drop)
    def forward(self,x):x=self.fc1(x);x=self.act(x);x=self.drop(x);x=self.fc2(x);x=self.drop(x);return x
def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(p): return [p/np.power(10000,2*(h//2)/d_hid) for h in range(d_hid)]
    t=np.array([get_position_angle_vec(p) for p in range(n_position)]);t[:,0::2]=np.sin(t[:,0::2]);t[:,1::2]=np.cos(t[:,1::2])
    return torch.FloatTensor(t).unsqueeze(0)
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., layer_idx=None): # Added layer_idx for consistency
        super().__init__();self.num_heads=num_heads;head_dim=embed_dim//num_heads;assert head_dim*num_heads==embed_dim
        self.scale=head_dim**-0.5;self.qkv=nn.Linear(embed_dim,embed_dim*3,bias=qkv_bias);self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(embed_dim,embed_dim);self.proj_drop=nn.Dropout(proj_drop)
    def forward(self,x):
        B,N,C=x.shape;qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0);attn=(q@k.transpose(-2,-1))*self.scale;attn=attn.softmax(dim=-1);attn=self.attn_drop(attn)
        x=(attn@v).transpose(1,2).reshape(B,N,C);x=self.proj(x);x=self.proj_drop(x);return x,attn # Standard MHA returns attn
class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., layer_idx=1):
        super().__init__();self.num_heads=num_heads;self.head_dim=embed_dim//num_heads;assert self.head_dim*num_heads==embed_dim
        self.scale=self.head_dim**-0.5;self.wq=nn.Linear(embed_dim,embed_dim*2,bias=qkv_bias);self.wk=nn.Linear(embed_dim,embed_dim*2,bias=qkv_bias)
        self.wv=nn.Linear(embed_dim,embed_dim*2,bias=qkv_bias);self.lambda_param=nn.Parameter(torch.tensor(1.0))
        self.norm=nn.LayerNorm(self.head_dim*2);l=layer_idx;self.lambda_init=0.8-0.6*np.exp(-0.3*(l-1));self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(embed_dim*2,embed_dim);self.proj_drop=nn.Dropout(proj_drop)
        # Store attn1/attn2 for potential visualization if needed, but requires modifying forward
        self.attn1_map = None
        self.attn2_map = None
        self.compute_std_attn = False # Flag to compute standard-like attention

    def forward(self,x):
        B,N,C=x.shape;H=self.num_heads;D=self.head_dim
        q=self.wq(x).reshape(B,N,H,2*D).permute(0,2,1,3);k=self.wk(x).reshape(B,N,H,2*D).permute(0,2,1,3);v=self.wv(x).reshape(B,N,H,2*D).permute(0,2,1,3)
        q1,q2=q.chunk(2,dim=-1);k1,k2=k.chunk(2,dim=-1)
        attn1_scores=(q1@k1.transpose(-2,-1))*self.scale;attn2_scores=(q2@k2.transpose(-2,-1))*self.scale
        attn1=attn1_scores.softmax(dim=-1);attn2=attn2_scores.softmax(dim=-1)
        self.attn1_map = attn1 # Store raw softmax scores
        self.attn2_map = attn2
        attn1_dropped=self.attn_drop(attn1);attn2_dropped=self.attn_drop(attn2) # Apply dropout *before* subtraction
        lambda_val=self.lambda_param.view(1,1,1,1)
        diff_attn_output=(attn1_dropped-lambda_val*attn2_dropped)@v # Use dropped attention for output calculation
        normalized_output=self.norm(diff_attn_output.reshape(-1,2*D)).reshape(B,H,N,2*D)
        scaled_output=normalized_output*(1.0-self.lambda_init);concatenated_output=scaled_output.transpose(1,2).reshape(B,N,C*2)
        x=self.proj(concatenated_output);x=self.proj_drop(x)
        # Return the un-dropped attn1 map if requested for visualization
        return x, attn1.detach() if self.compute_std_attn else None

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_differential_attn=False, layer_idx=1):
        super().__init__(); self.norm1=norm_layer(embed_dim); self.use_differential_attn=use_differential_attn
        AttentionModule=MultiHeadDifferentialAttention if use_differential_attn else MultiHeadAttention
        self.attn=AttentionModule(embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop,layer_idx=layer_idx)
        self.norm2=norm_layer(embed_dim); mlp_hidden_dim=int(embed_dim*mlp_ratio)
        self.mlp=MLP(in_features=embed_dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

    def forward(self, x, compute_std_attn=False): # Pass flag down
        identity1=x; x_norm1=self.norm1(x)
        # Set flag for DiffAttn module if needed
        if isinstance(self.attn, MultiHeadDifferentialAttention):
            self.attn.compute_std_attn = compute_std_attn
        attn_output, attn_map = self.attn(x_norm1) # DiffAttn forward now returns attn1 if flag is set
        x=identity1+attn_output; identity2=x; x_norm2=self.norm2(x); mlp_output=self.mlp(x_norm2); x=identity2+mlp_output
        return x, attn_map # Return attn_map (which is None if not DiffAttn or flag=False)

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, pos_embed_type='1d_learned', use_differential_attn=False):
        super().__init__(); self.num_classes=num_classes; self.num_features=self.embed_dim=embed_dim
        self.patch_size=patch_size; self.pos_embed_type=pos_embed_type; self.img_size=img_size; self.use_differential_attn=use_differential_attn
        self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches; self.cls_token=nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_drop=nn.Dropout(p=drop_rate); self.num_patches=num_patches; self.seq_len=num_patches+1
        if pos_embed_type=='no_pos': self.pos_embed=None
        elif pos_embed_type=='1d_learned': self.pos_embed=nn.Parameter(torch.zeros(1,self.seq_len,embed_dim)); nn.init.trunc_normal_(self.pos_embed,std=.02)
        elif pos_embed_type=='sinusoidal': self.pos_embed=nn.Parameter(get_sinusoid_encoding_table(self.seq_len,embed_dim),requires_grad=False)
        else: raise ValueError(f"Unknown PE type: {pos_embed_type}")
        nn.init.trunc_normal_(self.cls_token,std=.02)
        self.blocks=nn.ModuleList([TransformerEncoderLayer(embed_dim,num_heads,mlp_ratio,qkv_bias,drop_rate,attn_drop_rate,norm_layer=norm_layer,use_differential_attn=use_differential_attn,layer_idx=i+1) for i in range(depth)])
        self.norm=norm_layer(embed_dim); self.head=nn.Linear(embed_dim,num_classes); self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(m,nn.Linear): nn.init.trunc_normal_(m.weight,std=.02); nn.init.constant_(m.bias,0) if hasattr(m,'bias')and m.bias is not None else None
        elif isinstance(m,nn.LayerNorm): nn.init.constant_(m.bias,0); nn.init.constant_(m.weight,1.0)
        elif isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='relu'); nn.init.constant_(m.bias,0) if hasattr(m,'bias')and m.bias is not None else None
    def prepare_tokens(self,x): B=x.shape[0];x=self.patch_embed(x);cls_tokens=self.cls_token.expand(B,-1,-1);x=torch.cat((cls_tokens,x),dim=1);x=x+self.pos_embed if self.pos_embed is not None else x;x=self.pos_drop(x);return x
        # --- REPLACE the existing single-line forward method with this ---
    def forward(self, x):
        x = self.prepare_tokens(x)
        # Iterate through the transformer blocks
        for blk in self.blocks:
            x, _ = blk(x) # Correctly unpack the tuple returned by the block's forward pass
        x = self.norm(x)
        return self.head(x[:, 0]) # Return the prediction for the [CLS] token
    # --- End of corrected forward method ---
    # --- Methods to get standard-like attention maps for visualization ---
    @torch.no_grad()
    def get_last_selfattention(self, x):
        """Gets the 'attn1' map (like standard attention) from the last layer."""
        x = self.prepare_tokens(x)
        attn_weights = None
        for i, blk in enumerate(self.blocks):
            # Pass compute_std_attn=True only for the last block
            is_last = (i == len(self.blocks) - 1)
            x, attn_map = blk(x, compute_std_attn=is_last)
            if is_last:
                attn_weights = attn_map # Get attn1 if DiffAttn, or standard attn
        return attn_weights # Shape [B, H, N+1, N+1] or None if model incompatible

    @torch.no_grad()
    def get_intermediate_layers_attention(self, x):
        """Gets the 'attn1' map (like standard attention) from all layers."""
        x = self.prepare_tokens(x)
        attentions = []
        for blk in self.blocks:
            x, attn_map = blk(x, compute_std_attn=True) # Ask every block for attn1
            if attn_map is not None:
                 attentions.append(attn_map)
            else:
                 # This case shouldn't happen if MultiHeadAttention returns attn
                 # but as a fallback, append None or zeros
                 print(f"Warning: Could not get attention map from layer {len(attentions)}")
                 attentions.append(None) # Or handle differently
        return attentions # List of tensors [B, H, N+1, N+1]

# --- End of VisionTransformer Definition ---


def load_model_from_config(config_path, model_path, device):
    """Loads a ViT model using config and state dict."""
    print(f"Loading config from: {config_path}")
    print(f"Loading model weights from: {model_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        model_arg_names = ['img_size', 'patch_size', 'embed_dim', 'depth', 'num_heads',
                           'mlp_ratio', 'qkv_bias', 'drop_rate', 'attn_drop_rate',
                           'pos_embed_type']
        model_kwargs = {k: v for k, v in config.items() if k in model_arg_names}
        model_kwargs.setdefault('num_classes', 10)
        # *** IMPORTANT: Ensure Diff-ViT models are loaded correctly ***
        model_kwargs['use_differential_attn'] = True # Assume tuned models are Diff-ViT

        print("Model Kwargs from Config:", model_kwargs)
        model = VisionTransformer(**model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        return model, config
    except FileNotFoundError:
        print(f"Error: Config or model file not found at specified paths.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def get_cifar10_test_data(config, data_dir='./data_cifar', indices=None):
    """Loads CIFAR-10 test data with appropriate transforms."""
    img_size = config.get('img_size', 32)
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
    ])
    try:
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        if indices:
            testset = Subset(testset, indices)
        print(f"Loaded CIFAR-10 test set. Size: {len(testset)}")
        return testset
    except Exception as e:
        print(f"Error loading CIFAR-10 data: {e}")
        exit(1)

def tensor_to_pil(tensor_img):
    """Converts a normalized PyTorch tensor back to a PIL image."""
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(CIFAR_MEAN, CIFAR_STD)],
        std=[1/s for s in CIFAR_STD]
    )
    img_deprocessed = inv_normalize(tensor_img.cpu()).clamp(0, 1) # Ensure tensor is on CPU
    img_pil = transforms.ToPILImage()(img_deprocessed)
    return img_pil

# --- Visualization Functions ---

# Task 2: Last Layer Attention
def visualize_last_layer_attention(model, dataset, config, sample_indices, output_dir):
    print("\n--- Task 2: Visualizing Last Layer Attention (Diff-ViT Model) ---")
    os.makedirs(output_dir, exist_ok=True)

    img_size = config.get('img_size', 32)
    patch_size = config.get('patch_size', 4)
    num_heads = config.get('num_heads', 6)
    grid_size = img_size // patch_size

    for idx in sample_indices:
        try:
            img_tensor, label_idx = dataset[idx]
            label_name = CIFAR_CLASSES[label_idx]
            img_pil = tensor_to_pil(img_tensor)
            img_tensor_batch = img_tensor.unsqueeze(0).to(DEVICE)

            # Use the modified method to get standard-like attention (attn1)
            att_mat = model.get_last_selfattention(img_tensor_batch) # Should be [1, H, N+1, N+1]

            if att_mat is None:
                print(f"Skipping image {idx}: Could not retrieve attention map from last layer.")
                continue

            att_mat = att_mat.squeeze(0).cpu().numpy() # [H, N+1, N+1]
            cls_to_patch_att = att_mat[:, 0, 1:] # [H, N]
            avg_cls_to_patch_att = cls_to_patch_att.mean(axis=0) # [N]

            avg_att_map = avg_cls_to_patch_att.reshape(grid_size, grid_size)
            head_att_maps = cls_to_patch_att.reshape(num_heads, grid_size, grid_size)

            # Plot average attention
            fig_avg, ax_avg = plt.subplots(1, 2, figsize=(8, 4))
            fig_avg.suptitle(f'Avg CLS Attention (Last Layer) - Img {idx} ({label_name})', fontsize=12)
            ax_avg[0].imshow(img_pil); ax_avg[0].set_title("Original Image"); ax_avg[0].axis('off')
            im = ax_avg[1].imshow(avg_att_map, cmap='viridis', interpolation='nearest')
            ax_avg[1].set_title("Avg Attention (CLS->Patches)"); ax_avg[1].axis('off')
            fig_avg.colorbar(im, ax=ax_avg[1])
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path_avg = os.path.join(output_dir, f"last_layer_avg_att_img_{idx}_{label_name}.png")
            plt.savefig(save_path_avg); plt.close(fig_avg)
            print(f"Saved avg attention map: {os.path.basename(save_path_avg)}")

            # Plot head-wise attention
            num_cols = 4
            num_rows = math.ceil((num_heads + 1) / num_cols)
            fig_heads, axes_heads = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
            fig_heads.suptitle(f'Head-wise CLS Attention (Last Layer) - Img {idx} ({label_name})', fontsize=12)
            axes_flat = axes_heads.flatten()
            axes_flat[0].imshow(img_pil); axes_flat[0].set_title("Original"); axes_flat[0].axis('off')
            for h in range(num_heads):
                ax = axes_flat[h + 1]
                im = ax.imshow(head_att_maps[h], cmap='viridis', interpolation='nearest')
                ax.set_title(f"Head {h+1}"); ax.axis('off')
            for i in range(num_heads + 1, len(axes_flat)): axes_flat[i].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path_heads = os.path.join(output_dir, f"last_layer_heads_att_img_{idx}_{label_name}.png")
            plt.savefig(save_path_heads); plt.close(fig_heads)
            print(f"Saved head-wise attention maps: {os.path.basename(save_path_heads)}")

        except Exception as e:
            print(f"Error processing image index {idx}: {e}")
            continue


# Task 2: All Layers Attention (Average across heads)
def visualize_all_layers_attention(model, dataset, config, sample_indices, output_dir):
    print("\n--- Task 2: Visualizing All Layers Attention (Avg Heads, Diff-ViT Model) ---")
    os.makedirs(output_dir, exist_ok=True)

    img_size = config.get('img_size', 32)
    patch_size = config.get('patch_size', 4)
    depth = config.get('depth', 8)
    grid_size = img_size // patch_size

    for idx in sample_indices:
        try:
            img_tensor, label_idx = dataset[idx]
            label_name = CIFAR_CLASSES[label_idx]
            img_pil = tensor_to_pil(img_tensor)
            img_tensor_batch = img_tensor.unsqueeze(0).to(DEVICE)

            # Use the modified method to get standard-like attention (attn1) from all layers
            intermediate_attentions = model.get_intermediate_layers_attention(img_tensor_batch) # List[L] of [B, H, N+1, N+1]

            if not intermediate_attentions or any(a is None for a in intermediate_attentions):
                print(f"Skipping image {idx}: Could not retrieve all intermediate attention maps.")
                continue

            num_cols = 4
            num_rows = math.ceil((depth + 1) / num_cols)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
            fig.suptitle(f'Avg CLS Attention per Layer - Img {idx} ({label_name})', fontsize=12)
            axes_flat = axes.flatten()
            axes_flat[0].imshow(img_pil); axes_flat[0].set_title("Original"); axes_flat[0].axis('off')

            for layer_idx, att_mat in enumerate(intermediate_attentions):
                if att_mat is None: continue # Should not happen with current logic but safe check
                att_mat_np = att_mat.squeeze(0).cpu().numpy() # [H, N+1, N+1]
                cls_to_patch_att = att_mat_np[:, 0, 1:] # [H, N]
                avg_cls_to_patch_att = cls_to_patch_att.mean(axis=0) # [N]
                avg_att_map = avg_cls_to_patch_att.reshape(grid_size, grid_size)

                ax = axes_flat[layer_idx + 1]
                im = ax.imshow(avg_att_map, cmap='viridis', interpolation='nearest')
                ax.set_title(f"Layer {layer_idx+1}"); ax.axis('off')

            for i in range(depth + 1, len(axes_flat)): axes_flat[i].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(output_dir, f"all_layers_avg_att_img_{idx}_{label_name}.png")
            plt.savefig(save_path); plt.close(fig)
            print(f"Saved all layers avg attention map: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"Error processing image index {idx} for all layers: {e}")
            continue


# Task 3: Attention Rollout
class AttentionRollout:
    def __init__(self, model, head_fusion="mean"):
        self.model = model
        self.head_fusion = head_fusion
        self.model.eval()

    def __call__(self, input_tensor):
        # Get intermediate standard-like attentions (attn1)
        attentions = self.model.get_intermediate_layers_attention(input_tensor)

        if not attentions or any(a is None for a in attentions):
             print("Error in AttentionRollout: Could not retrieve all attention maps.")
             return None

        # Ensure attentions are on the correct device
        attentions = [a.to(input_tensor.device) for a in attentions]

        B, H, N_plus_1, _ = attentions[0].shape
        device = input_tensor.device
        result = torch.eye(N_plus_1, device=device).unsqueeze(0).expand(B, -1, -1)

        for attn_layer in attentions:
            if self.head_fusion == "mean": attn_fused = attn_layer.mean(axis=1)
            elif self.head_fusion == "max": attn_fused = attn_layer.max(axis=1)[0]
            elif self.head_fusion == "min": attn_fused = attn_layer.min(axis=1)[0]
            else: raise ValueError("Invalid head_fusion method")

            I = torch.eye(N_plus_1, device=device).unsqueeze(0).expand(B, -1, -1)
            a = attn_fused + I
            a = a / (a.sum(axis=-1, keepdim=True) + 1e-8)
            result = a @ result

        return result

def visualize_attention_rollout(model, dataset, config, sample_indices, output_dir):
    print("\n--- Task 3: Visualizing Attention Rollout (Diff-ViT Model) ---")
    os.makedirs(output_dir, exist_ok=True)

    img_size = config.get('img_size', 32)
    patch_size = config.get('patch_size', 4)
    grid_size = img_size // patch_size
    rollout_calculator = AttentionRollout(model, head_fusion="mean")

    for idx in sample_indices:
        try:
            img_tensor, label_idx = dataset[idx]
            label_name = CIFAR_CLASSES[label_idx]
            img_pil = tensor_to_pil(img_tensor)
            img_tensor_batch = img_tensor.unsqueeze(0).to(DEVICE)

            rollout_matrix = rollout_calculator(img_tensor_batch) # [B, N+1, N+1]

            if rollout_matrix is None:
                print(f"Skipping image {idx}: Attention rollout failed.")
                continue

            rollout_matrix_np = rollout_matrix.squeeze(0).cpu().numpy() # [N+1, N+1]
            # Visualize flow INTO CLS token FROM patch tokens
            rollout_cls_in = rollout_matrix_np[0, 1:] # [N]
            vis_map_reshaped = rollout_cls_in.reshape(grid_size, grid_size)

            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            fig.suptitle(f'Attention Rollout - Img {idx} ({label_name})', fontsize=12)
            ax[0].imshow(img_pil); ax[0].set_title("Original Image"); ax[0].axis('off')
            im = ax[1].imshow(vis_map_reshaped, cmap='viridis', interpolation='nearest')
            ax[1].set_title("Rollout (Patches -> CLS)")
            ax[1].axis('off')
            fig.colorbar(im, ax=ax[1])
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = os.path.join(output_dir, f"rollout_att_img_{idx}_{label_name}.png")
            plt.savefig(save_path); plt.close(fig)
            print(f"Saved attention rollout map: {os.path.basename(save_path)}")

        except Exception as e:
            print(f"Error processing image index {idx} for rollout: {e}")
            continue


# Task 4: Positional Embedding Similarity
def visualize_pos_embed_similarity(model, config, output_dir):
    print("\n--- Task 4: Visualizing Positional Embedding Similarity ---")
    os.makedirs(output_dir, exist_ok=True)

    pe_type = config.get('pos_embed_type', 'unknown')
    print(f"Attempting PE visualization for type: {pe_type}")

    if not hasattr(model, 'pos_embed') or model.pos_embed is None:
        print(f"Skipping PE visualization: Model does not have 'pos_embed' or it's None (Type: {pe_type}).")
        return
    if pe_type != '1d_learned':
         print(f"Skipping PE visualization: Model PE type is '{pe_type}', but required type is '1d_learned'.")
         return

    pos_embed = model.pos_embed.squeeze(0).detach().cpu() # [N+1, D]
    similarity_matrix = torch.matmul(pos_embed, pos_embed.T).numpy() # [N+1, N+1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    im = ax.imshow(similarity_matrix, cmap='viridis')
    ax.set_title(f"Positional Embedding Similarity (Dot Product)\nType: {pe_type}")
    ax.set_xlabel("Token Index (0=CLS, 1..N=Patches)")
    ax.set_ylabel("Token Index")
    fig.colorbar(im, ax=ax, label="Dot Product Similarity")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"pos_embed_similarity_{pe_type}.png")
    plt.savefig(save_path); plt.close(fig)
    print(f"Saved PE similarity map: {os.path.basename(save_path)}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diff-ViT Visualization Script')
    # --- Arguments for Best Model (Tasks 2 & 3 - typically sinusoidal in your log) ---
    parser.add_argument('--best_config_path', type=str,
                        default='results_diff_vit_tuned/best_config_diff_vit_S4_PE_sinusoidal.json',
                        help='Path to overall best Diff-ViT JSON config file (likely sinusoidal)')
    parser.add_argument('--best_model_path', type=str,
                        default='results_diff_vit_tuned/models/best_diff_vit_S4_PE_sinusoidal.pth',
                        help='Path to overall best Diff-ViT .pth state_dict file (likely sinusoidal)')
    # --- Arguments for 1D Learned PE Model (Task 4) ---
    parser.add_argument('--learned1d_config_path', type=str,
                        default='results_diff_vit_tuned/best_config_diff_vit_S4_PE_1d_learned.json',
                        help='Path to 1D learned PE Diff-ViT JSON config file')
    parser.add_argument('--learned1d_model_path', type=str,
                        default='results_diff_vit_tuned/models/best_diff_vit_S4_PE_1d_learned.pth',
                        help='Path to 1D learned PE Diff-ViT .pth state_dict file')
    # --- Other Arguments ---
    parser.add_argument('--data_dir', type=str, default='./data_cifar', help='Directory for CIFAR-10 data')
    parser.add_argument('--output_dir', type=str, default='./diff_vit_visualizations', help='Directory to save output plots')
    parser.add_argument('--img_indices', type=int, nargs='+', default=[0, 10, 25, 50, 100, 150], help='Indices of CIFAR-10 test images to visualize')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Best Model (for Tasks 2 & 3) ---
    print("="*10 + " Loading Best Diff-ViT Model (for Attention Viz) " + "="*10)
    best_model, best_config = load_model_from_config(args.best_config_path, args.best_model_path, DEVICE)
    test_dataset = get_cifar10_test_data(best_config, args.data_dir) # Load full test set

    # --- Run Attention Visualizations (Task 2 & 3) ---
    if best_model:
        attention_output_dir = os.path.join(args.output_dir, "attention_maps")
        visualize_last_layer_attention(best_model, test_dataset, best_config, args.img_indices, attention_output_dir)
        visualize_all_layers_attention(best_model, test_dataset, best_config, args.img_indices, attention_output_dir)
        visualize_attention_rollout(best_model, test_dataset, best_config, args.img_indices, attention_output_dir)
        del best_model, best_config # Free memory
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("Skipping attention visualizations as best model failed to load.")


    # --- Load 1D Learned PE Model (for Task 4) ---
    print("\n" + "="*10 + " Loading 1D Learned PE Diff-ViT Model (for PE Viz) " + "="*10)
    learned1d_model, learned1d_config = load_model_from_config(args.learned1d_config_path, args.learned1d_model_path, DEVICE)

    # --- Run Positional Embedding Visualization (Task 4) ---
    if learned1d_model:
        pe_output_dir = os.path.join(args.output_dir, "pos_embedding")
        visualize_pos_embed_similarity(learned1d_model, learned1d_config, pe_output_dir)
        del learned1d_model, learned1d_config # Free memory
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    else:
        print("Skipping PE visualization as 1D learned PE model failed to load.")


    print(f"\nVisualizations saved in: {args.output_dir}")
    print("Script finished.")