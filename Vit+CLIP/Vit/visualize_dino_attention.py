# -*- coding: utf-8 -*-
"""
visualize_dino_attention.py

Visualizes attention maps from a pretrained DINO ViT model loaded from
PyTorch Hub, applied to selected CIFAR-10 test images.

Corresponds to Task 1.4.1. (Modified to use hooks for attention capture)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import argparse

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# --- Global variable to store hook output ---
global captured_attention


# --- Hook function ---
def get_attention_hook(module, input, output):
    """Saves the attention map output by the hooked module."""
    global captured_attention
    # The structure of the output can vary.
    # If the module itself is MHA, output might be (value, attn_weights)
    # If it's a sub-module like Softmax, output might be just the weights.
    # We need to inspect the DINO model's MHA output or hook deeper.
    # Let's assume the MHA module (or a module within it) outputs weights directly or as second element.
    if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
        # Likely (output_value, attention_weights)
        captured_attention = output[1].detach()
    elif isinstance(output, torch.Tensor):
        # May be just the attention weights (e.g., if hooking softmax)
        # Or maybe just the output value (hooking wrong module/no weights returned)
        # For DINO ViT MHA, typically returns tuple, so this might indicate wrong hook placement
        # Let's assume for now this *could* be the attention if the tuple case fails
        print("Hook Warning: Module output is a single Tensor. Assuming it's the attention map.")
        captured_attention = output.detach()
    else:
        print(f"Hook Warning: Unexpected output type from hooked module: {type(output)}. Cannot capture attention.")
        captured_attention = None


# --- Helper Functions ---
def tensor_to_pil(tensor_img):
    """Converts a normalized PyTorch tensor back to a PIL image for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if tensor_img.is_cuda:
        mean = mean.to(tensor_img.device)
        std = std.to(tensor_img.device)
    img_denormalized = tensor_img * std + mean
    img_clamped = torch.clamp(img_denormalized, 0, 1)
    img_pil = transforms.ToPILImage()(img_clamped.cpu())
    return img_pil

# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DINO Attention Visualization Script (with Hooks)')
    # (Arguments remain the same as before)
    parser.add_argument('--model_name', type=str, default='dino_vits8',
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'],
                        help='Name of the DINO ViT model to load from PyTorch Hub.')
    parser.add_argument('--patch_size', type=int, default=8, choices=[8, 16],
                        help='Patch size of the DINO model (must match model_name).')
    parser.add_argument('--image_indices', type=int, nargs='+', default=[0, 10, 25, 50, 100],
                        help='Indices of CIFAR-10 test images to visualize.')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size expected by the DINO model.')
    parser.add_argument('--output_dir', type=str, default='./dino_attention_visualizations_hook', # Changed default dir slightly
                        help='Directory to save output plots.')
    parser.add_argument('--data_dir', type=str, default='./data_cifar',
                        help='Directory containing CIFAR-10 data.')

    args = parser.parse_args()

    # Validate patch size
    if ('s8' in args.model_name or 'b8' in args.model_name) and args.patch_size != 8: args.patch_size = 8; print("Adjusted patch_size to 8")
    if ('s16' in args.model_name or 'b16' in args.model_name) and args.patch_size != 16: args.patch_size = 16; print("Adjusted patch_size to 16")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load DINO Model ---
    print(f"Loading DINO model '{args.model_name}'...")
    try:
        model_dino = torch.hub.load('facebookresearch/dino:main', args.model_name, pretrained=True)
        model_dino.eval()
        model_dino.to(device)
        print(f"Loaded DINO model '{args.model_name}' successfully.")
    except Exception as e: print(f"Error loading DINO model: {e}"); exit(1)

    # --- Prepare DINO Preprocessing ---
    transform_dino = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # --- Load CIFAR-10 Test Data ---
    try:
        cifar_testset_raw = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=None)
        print(f"Loaded CIFAR-10 test set (raw). Size: {len(cifar_testset_raw)}")
    except Exception as e: print(f"Error loading CIFAR-10 data: {e}"); exit(1)

    # --- Find the target Attention module to hook ---
    # This might require inspecting the `model_dino` structure.
    # Typically, it's the MultiHeadAttention module within the last TransformerBlock.
    try:
        # Common path: model -> blocks -> last block -> attn module
        target_attn_module = model_dino.blocks[-1].attn
        print(f"Hooking target module: {type(target_attn_module)}")
    except AttributeError:
        print("Error: Cannot find the target attention module 'model.blocks[-1].attn'. Model structure might differ.")
        exit(1)

    # --- Visualization Loop ---
    num_patches_h = args.img_size // args.patch_size
    num_patches_w = args.img_size // args.patch_size
    try: num_heads = target_attn_module.num_heads
    except AttributeError: num_heads = 6 if 'vit_small' in args.model_name or 'vits' in args.model_name else 12; print(f"Guessed num_heads={num_heads}")

    print(f"\nVisualizing DINO attention for CIFAR-10 images (indices: {args.image_indices}) using HOOKS...")

    for i, idx in enumerate(args.image_indices):
        captured_attention = None # Reset before each image
        if idx >= len(cifar_testset_raw): print(f"Warning: Index {idx} out of bounds. Skipping."); continue
        

        img_pil, label_idx = cifar_testset_raw[idx]
        label_name = cifar_testset_raw.classes[label_idx]
        img_tensor = transform_dino(img_pil).unsqueeze(0).to(device)

        # --- Register Hook, Run Forward, Remove Hook ---
        hook_handle = target_attn_module.register_forward_hook(get_attention_hook)
        
        captured_attention = None # Reset before forward pass

        with torch.no_grad():
            _ = model_dino(img_tensor)

        hook_handle.remove() # !! IMPORTANT: Remove hook after use !!

        # --- Process Captured Attention ---
        if captured_attention is None:
            print(f"Hook failed to capture attention for image {idx}. Skipping.")
            continue

        att_mat = captured_attention.squeeze(0).cpu().numpy() # [H, N+1, N+1]

        expected_tokens = num_patches_h * num_patches_w + 1
        if att_mat.shape[-1] != expected_tokens:
            print(f"Warning: Unexpected captured attention matrix shape {att_mat.shape} for image {idx}. Expected {expected_tokens} tokens.")
            continue

        cls_to_patch_att = att_mat[:, 0, 1:] # [H, N]
        avg_cls_to_patch_att = cls_to_patch_att.mean(axis=0) # [N]

        try:
            att_map_grid = avg_cls_to_patch_att.reshape(num_patches_h, num_patches_w)
        except ValueError as e: print(f"Error reshaping attention map for image {idx}: {e}."); continue

        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'DINO {args.model_name} Attention (HOOKED) (CLS->Patches)\nCIFAR Image {idx} ({label_name})', fontsize=14)
        axs[0].imshow(img_pil.resize((args.img_size, args.img_size))); axs[0].set_title("Original Image (Resized)"); axs[0].axis('off')
        im = axs[1].imshow(att_map_grid, cmap='viridis', interpolation='nearest')
        axs[1].set_title("Avg Attention (Last Layer)"); axs[1].axis('off')
        fig.colorbar(im, ax=axs[1])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(args.output_dir, f"dino_hooked_attn_cifar_img_{idx}_{label_name}.png")
        plt.savefig(save_path); plt.close(fig)
        print(f"Saved DINO hooked attention visualization to {os.path.basename(save_path)}")

    print(f"\nDINO hooked attention visualizations saved in {args.output_dir}")
    del model_dino
    if torch.cuda.is_available(): torch.cuda.empty_cache()