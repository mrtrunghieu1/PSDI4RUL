import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image
from utils.tools import visualize_embeddings, visualize_gate_weights, visualize_embeddings_difference, visualize_four_modal_features, visualize_gate_weights_binary
from utils.tensor_utils import validate_tensor_compatibility

# Import custom modules
sys.path.append("../")
from src.psdi_kd.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.Query_TimeSeries_Interaction import QueryTimeSeriesInteraction
from layers.TimeSeries_To_Image import time_series_to_simple_image
from layers.models_mae import *
from transformers.models.vilt import *
from transformers import ViTConfig, ViTModel

class PatchMemoryBank:
    def __init__(self, max_size, patch_size, feature_dim, device=None):
        """
        Initialize patch memory bank.
        
        Parameters:
            max_size (int): maximum number of stored patches
            patch_size (int): size of each patch
            feature_dim (int): dimension of each patch feature
            device (torch.device): device (CPU/GPU)
        """
        self.max_size = max_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cpu')
        self.patches = torch.zeros((max_size, feature_dim), device=self.device)  # [100, d_model]
        self.ptr = 0

    def update(self, new_patches):
        """Update patch memory bank"""
        n = new_patches.size(0)
        new_patches_flat = new_patches.mean(dim=1)  # [n, d_model]
        
        if self.ptr + n > self.max_size:
            # If memory bank is full, wrap around
            self.patches[self.ptr:] = new_patches_flat[:self.max_size - self.ptr]
            self.ptr = 0
        else:
            self.patches[self.ptr:self.ptr + n] = new_patches_flat
            self.ptr += n

    def retrieve(self, query_patches, top_k=5):
        """Retrieve most similar patches from memory bank"""
        query_flat = query_patches.mean(dim=1)  # [224, d_model]
        memory_flat = self.patches  # [100, d_model]
        
        similarity = torch.matmul(query_flat, memory_flat.T)  # [224, 100]
        _, indices = similarity.topk(top_k, dim=-1)
        
        retrieved_patches = self.patches[indices]
        return retrieved_patches, indices

class Model(nn.Module):
    """
    Multimodal time series prediction model based on VLM.
    Supports knowledge distillation between teacher and student models.     
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        
        # Store distillation mode status
        self.use_distillation = getattr(config, 'use_distillation', False)
        
        # Determine whether to use gate based on dataset
        self.use_gate = self._should_use_gate(config)
        print(f"Dataset: {getattr(config, 'data', 'unknown')}")
        
        # Adjust configuration based on mode
        self._adjust_config_for_mode()
        
        # Initialize VLM manager
        self.vlm_manager = VLMManager(config)
        self.device = torch.device('cuda:{}'.format(self.config.gpu))
        
        # Store VLM hidden layer size for correct dimension initialization
        self.vlm_hidden_size = self.vlm_manager.hidden_size
        print(f"LVM hidden layer size: {self.vlm_hidden_size}, model d_model: {config.d_model}")
        
        # Initialize patch memory bank
        self.patch_memory_bank = PatchMemoryBank(
            max_size=getattr(config, 'patch_memory_size', 100),
            patch_size=config.patch_len,
            feature_dim=config.d_model,
            device=self.device
        )
        
        # Initialize all model modules
        self._init_modules(config)
        self.vlm_model = self.vlm_manager.model

    def _should_use_gate(self, config):
        """Perform internal compatibility check"""
        return validate_tensor_compatibility(config, tensor_type='gate_mode')

    def _adjust_config_for_mode(self):
        """Adjust configuration based on distillation mode"""
        if self.use_distillation:
            # Distillation mode keeps current configuration
            pass
        else:
            # Non-distillation mode: ensure student model is not used
            if hasattr(self.config, 'is_student'):
                self.config.is_student = False
            
            # In non-distillation mode, directly use teacher_vlm_type
            if hasattr(self.config, 'teacher_vlm_type'):
                original_vlm_type = self.config.vlm_type
                self.config.vlm_type = self.config.teacher_vlm_type
                print(f"Non-distillation mode: switch VLM type from {original_vlm_type} to {self.config.vlm_type} (teacher_vlm_type)")

    def _init_modules(self, config):
        """Initialize all model modules and explicitly organize components"""
        # Embedding layer
        self.patch_embedding = PatchEmbedding(
            config.d_model, 
            config.patch_len, 
            config.stride, 
            config.padding, 
            config.dropout
        )
        self.head_nf = config.d_model * int((config.seq_len - config.patch_len) / config.stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        
        # New ICML version head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout)
        )
        
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Multimodal enhancement
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(self.vlm_hidden_size, config.d_model),  # Only use vision
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.multimodal_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Feature processing layer
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.vlm_hidden_size, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model)
        )
        
        # Initialize gate layer based on whether to use gate
        if self.use_gate:
            # Gate version: two gates
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
                nn.Softmax(dim=-1)
            )
            
            self.prediction_fusion_gate = nn.Sequential(
                nn.Linear(config.pred_len * 2, config.pred_len),
                nn.GELU(),
                nn.Linear(config.pred_len, 2),
                nn.Softmax(dim=-1)
            )
            print("Gate version")
        else:
            # No-gate version: only one gate
            self.gate = nn.Sequential(
                nn.Linear(config.pred_len * 2, config.pred_len),
                nn.GELU(),
                nn.Linear(config.pred_len, 2),
                nn.Softmax(dim=-1)
            )
            print("No-gate version")
            
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Memory-related modules
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Time series to image conversion
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, 
            hidden_dim=48, 
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size, 
            periodicity=config.periodicity
        )
        
        self.query_time_series_interaction = QueryTimeSeriesInteraction(
            num_queries=8, 
            time_series_embedding_dim=config.d_model, 
            query_embedding_dim=64,
            hidden_dim=self.vlm_hidden_size, 
            num_heads=4
        )
        
        # Other parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gate parameter
        self.layer_norm = nn.LayerNorm(config.d_model)

    def _compute_local_memory(self, patches):
        """Compute local memory, by retrieving and fusing similar patches"""
        # Retrieve similar patches from memory bank
        retrieved_patches, _ = self.patch_memory_bank.retrieve(patches, top_k=self.config.top_k)
        
        # Process retrieved patches with local MLP
        local_memory = self.local_memory_mlp(retrieved_patches)
        
        # Average on retrieved patches
        local_memory = local_memory.mean(dim=1, keepdim=True)
        
        # Residual connection with original patches
        local_memory = local_memory + patches
        
        return local_memory

    def _compute_global_memory(self, patches):
        """Compute global memory, by aggregating information between all patches"""
        if self.use_gate:
            # Gate version: directly use attention output
            global_memory, _ = self.memory_attention(
                query=patches,
                key=patches,
                value=patches
            )
        else:
            # No-gate version: use average pooling
            attn_output, _ = self.memory_attention(
                query=patches,
                key=patches,
                value=patches
            )
            # Time pooling to get global context
            global_memory = attn_output.mean(dim=1, keepdim=True)
        
        # Update patch memory bank with current patches
        self.patch_memory_bank.update(patches.detach())
        
        return global_memory

    def forward_prediction(self, x_enc, vision_embeddings, return_attention=False):
        """Main forward prediction branch based on ICML version"""
        B, L, n_vars = x_enc.shape
        
        # 1. Process time features
        patches, _ = self.patch_embedding(x_enc.transpose(1, 2))  # [B * n_vars, n_patches, d_model]
        
        # 2. Compute local and global memory
        local_memory = self._compute_local_memory(patches)  # [B * n_vars, n_patches, d_model]
        global_memory = self._compute_global_memory(patches)  # [B * n_vars, n_patches, d_model]
        
        # 3. Combine local and global memory
        if self.use_gate:
            # Gate version: use gate mechanism
            combined_features = torch.cat([local_memory, global_memory], dim=-1)  # [B * n_vars, n_patches, d_model*2]
            gate_weights = self.memory_fusion_gate(combined_features)  # [B * n_vars, n_patches, 2]
            
            # Weighted fusion
            memory_features = (
                gate_weights[:, :, 0:1] * local_memory +
                gate_weights[:, :, 1:2] * global_memory
            )  # [B * n_vars, n_patches, d_model]
        else:
            # No-gate version: simple addition
            memory_features = local_memory + global_memory  # [B * n_vars, n_patches, d_model]

        # 4. Get time prediction
        memory_features_flat = self.flatten(memory_features)  # [B * n_vars, head_nf]
        temporal_features = self.temporal_head(memory_features_flat)  # [B * n_vars, d_model]
        memory_pred = self.memory_head(memory_features_flat)  # [B * n_vars, pred_len]
        
        # Reshape
        temporal_features = einops.rearrange(temporal_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, d_model]
        memory_pred = einops.rearrange(memory_pred, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, pred_len]
        
        # 5. Process multimodal features
        multimodal_features = self.multimodal_enhancement(vision_embeddings)  # [B, d_model]
        multimodal_features = multimodal_features.unsqueeze(1).expand(-1, n_vars, -1)  # [B, n_vars, d_model]
        multimodal_features = self.layer_norm(multimodal_features)  # [B, n_vars, d_model]
        
        # 6. Cross-modal attention enhancement
        temporal_features = temporal_features / torch.norm(temporal_features, dim=-1, keepdim=True)
        multimodal_features = multimodal_features / torch.norm(multimodal_features, dim=-1, keepdim=True)
        
        # Save attention weights for distillation
        attention_weights = None
        if return_attention:
            multimodal_features, attention_weights = self.cross_attention(
                query=temporal_features,
                key=multimodal_features,
                value=multimodal_features,
                need_weights=True
            )  # [B, n_vars, d_model], [B, n_vars, n_vars]
        else:
            multimodal_features, _ = self.cross_attention(
                query=temporal_features,
                key=multimodal_features,
                value=multimodal_features
            )  # [B, n_vars, d_model]
        
        # 7. Standardize cross-attention output
        multimodal_features = self.layer_norm(multimodal_features)  # [B, n_vars, d_model]
        multimodal_pred = self.multimodal_head(multimodal_features)  # [B, n_vars, pred_len]
        
        # 8. Compute gate weights
        combined_features = torch.cat([memory_pred, multimodal_pred], dim=-1)  # [B, n_vars, pred_len * 2]
        if self.use_gate:
            gate_weights = self.prediction_fusion_gate(combined_features)  # [B, n_vars, 2]
        else:
            gate_weights = self.gate(combined_features)  # [B, n_vars, 2]
        
        # 9. Weighted fusion
        fused_features = (
            gate_weights[:, :, 0:1] * memory_pred +
            gate_weights[:, :, 1:2] * multimodal_pred
        ) # [B, n_vars, pred_len]
        
        # 10. Final fusion
        predictions = self.fusion_layer(
            torch.cat([memory_pred, fused_features], dim=-1)
        ) + memory_pred  # [B, n_vars, pred_len]
        
        # Visualization (optional)
        if self.config.visualize_embeddings:
            visualize_embeddings_difference(memory_pred, multimodal_pred, save_path='embedding_difference.png')
            visualize_gate_weights_binary(gate_weights, save_path='binary_gate_weights.png')
        
        if return_attention:
            return predictions.permute(0, 2, 1), attention_weights  # [B, pred_len, n_vars], [B, n_vars, n_vars]
        else:
            return predictions.permute(0, 2, 1)  # [B, pred_len, n_vars]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """Main forward pass"""
        B, L, D = x_enc.shape
        
        # === Ablation at the beginning: directly block modal information flow at the source === 
        # Ablate temporal modality: set time series input to zero
        if getattr(self.config, 'ablate_temporal_modality', False):
            x_enc = torch.zeros_like(x_enc)
        
        # Normalize input
        x_enc, means, stdev = self._normalize_input(x_enc)
        
        # Convert time series to image
        images = self.vision_augmented_learner(x_enc, self.config.image_size, 
                                              self.config.seq_len, self.config.periodicity)
        
        # Process input with VLM
        vision_embeddings = self._process_with_vlm(B, images)
        
        # Ablate vision modality: set vision embedding to zero
        if getattr(self.config, 'ablate_vision_modality', False):
            vision_embeddings = torch.zeros_like(vision_embeddings)
        
        # Main prediction branch
        predictions = self.forward_prediction(x_enc, vision_embeddings, return_attention=False)
        
        # Denormalize output
        y = self._denormalize_output(predictions, means, stdev)
        return y
    
    def _should_use_mask(self, mask_ratio):
        """Uniform mask usage decision logic"""
        return (
            mask_ratio > 0 and 
            getattr(self.config, 'is_student', False) and 
            self.use_distillation
        )

    def _process_with_vlm(self, B, images):
        """Process images with VLM based on current mode - fixed version"""
        mask_ratio = getattr(self.config, 'mask_ratio', 0.0)
        
        if self._should_use_mask(mask_ratio):
            try:
                return self.vlm_manager.process_inputs_vision_only_masked(B, images, mask_ratio)
            except Exception as e:
                print(f"Mask version failed, using standard version: {e}")
                return self.vlm_manager.process_inputs_vision_only(B, images)
        else:
            return self.vlm_manager.process_inputs_vision_only(B, images)

    def forward_with_vision_features(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, mask_ratio=0.0, return_attention=False):
        """Return vision features and prediction output for distillation - fixed version"""
        B, L, D = x_enc.shape
        
        # === Ablation at the beginning: directly block modal information flow at the source ===
        # Ablate temporal modality: set time series input to zero
        if getattr(self.config, 'ablate_temporal_modality', False):
            x_enc = torch.zeros_like(x_enc)
        
        # Normalize input
        x_enc, means, stdev = self._normalize_input(x_enc)
        
        # Convert time series to image
        images = self.vision_augmented_learner(x_enc, self.config.image_size, self.config.seq_len, self.config.periodicity)
        
        # Process input with VLM - using fixed mask logic
        if self._should_use_mask(mask_ratio):
            try:
                vision_embeddings = self.vlm_manager.process_inputs_vision_only_masked(B, images, mask_ratio)
            except Exception as e:
                print(f"Mask version failed, using standard version: {e}")
                vision_embeddings = self.vlm_manager.process_inputs_vision_only(B, images)
        else:
            vision_embeddings = self.vlm_manager.process_inputs_vision_only(B, images)
        
        # Ablate vision modality: set vision embedding to zero
        if getattr(self.config, 'ablate_vision_modality', False):
            vision_embeddings = torch.zeros_like(vision_embeddings)
        
        # Main prediction branch and capture attention weights
        predictions, attention_weights = self.forward_prediction(x_enc, vision_embeddings, return_attention=return_attention)
        
        # Denormalize output
        y = self._denormalize_output(predictions, means, stdev)
        
        return y, vision_embeddings, attention_weights


    def _normalize_input(self, x):
        """Normalize input time series data"""
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        """Denormalize model prediction"""
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data to image tensor.
        """
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:            
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        
        # Normalize image to [0, 255] uint8
        images = self._normalize_images(images)
        
        # Optional save image
        if self.config.save_images:
            self.save_images(images)

        return images
    
    @staticmethod
    def _normalize_images(images):
        """
        Normalize image tensor to [0, 255] uint8.
        Assume image is in [0, 1] or needs scaling.
        
        Parameters:
        - images (Tensor): input image, shape [B, C, H, W]
        
        Returns:
        - Tensor: normalized image, shape [B, C, H, W]
        """
        # Calculate minimum and maximum values for each image across all channels and spatial dimensions
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        # Normalize to [0, 1]
        images = (images - min_vals) / scale
        # Scale to [0, 255] and clamp to ensure valid range
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        
        return images

    @torch.no_grad()
    def save_images(self, images):
        """
        Save generated images.

        Parameters:
        - images: Tensor containing images to save.
        """
        save_dir = "ts-images/occamvts"
        os.makedirs(save_dir, exist_ok=True)
        for i, img_tensor in enumerate(images):
            img_tensor = img_tensor.cpu().numpy().transpose(1, 2, 0)  # Convert to [H, W, C]
            img_tensor = img_tensor.astype(np.uint8)
            img = Image.fromarray(img_tensor)
            img.save(os.path.join(save_dir, f"image_{i}.png"))


def check_image_channel(np_img):
    """
    Check the number of channels in the image and adjust the dimension order.

    Parameters:
    - np_img: Input image.

    Returns:
    - np_img: Adjusted image.
    - mode: Image mode (e.g., 'RGB', 'L').
    """
    # Check the number of channels and adjust the dimension order
    if np_img.shape[0] == 3:
        # RGB image: convert from [C, H, W] to [H, W, C]
        np_img = np.transpose(np_img, (1, 2, 0))  # [224, 224, 3]
        mode = 'RGB'
    elif np_img.shape[0] == 1:
        # Grayscale image: convert from [C, H, W] to [H, W]
        np_img = np.squeeze(np_img, 0)  # [224, 224]
        mode = 'L'
    else:
        print(f"Unexpected number of channels: {np_img.shape[0]}")

    return np_img, mode