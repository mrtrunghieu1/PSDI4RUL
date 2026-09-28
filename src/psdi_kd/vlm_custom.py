# vlm_custom.py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel, ViTConfig
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class BackboneFactory:
    """Factory class for creating different visual backbone networks"""
    
    @staticmethod
    def create_backbone(vision_type, config):
        """
        Create visual backbone network
        
        Parameters:
            vision_type: backbone network type
            config: configuration parameters
            
        Returns:
            nn.Module: visual backbone network
        """
        # Use simple mapping table to select backbone network creation function
        backbone_creators = {
            'tiny_vit': BackboneFactory._create_tiny_vit,
            'efficientnet': BackboneFactory._create_efficientnet,
            'mobilenet': BackboneFactory._create_mobilenet
        }
        
        creator = backbone_creators.get(vision_type)
        if creator:
            return creator(config)
        else:
            raise ValueError(f"Unsupported visual backbone network type: {vision_type}")
    
    @staticmethod
    def _create_tiny_vit(config):
        """Create lightweight ViT"""
        vit_config = ViTConfig(
            hidden_size=config.d_model,
            num_hidden_layers=4,  # Reduce number of layers
            num_attention_heads=4,  # Reduce number of attention heads
            intermediate_size=config.d_model * 2,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout
        )
        return ViTModel(vit_config)
    
    @staticmethod
    def _create_efficientnet(config):
        """Create EfficientNet"""
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # Modify the last classifier to linear projection
        model.classifier = nn.Linear(1280, config.d_model)
        return model
    
    @staticmethod
    def _create_mobilenet(config):
        """Create MobileNet"""
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # Modify the last classifier to linear projection
        model.classifier[3] = nn.Linear(1024, config.d_model)
        return model

class CustomVLM(nn.Module):
    """
    Custom visual language model, only processing visual features.
    Supports teacher and student models with different visual backbones.
    """
    def __init__(self, config):
        super(CustomVLM, self).__init__()
        self.config = config
        self.device = self._acquire_device()
        
        # Initialize hidden_size
        self.is_student = getattr(config, 'is_student', False)
        self.hidden_size = getattr(config, 'student_hidden_size' if self.is_student else 'hidden_size', 768)
        
        # Initialize vision encoder based on model type
        self._init_vision_encoder()

    def _acquire_device(self):
        """Get appropriate device"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _init_vision_encoder(self):
        """Initialize vision encoder based on model type"""
        # Always use the same processor for compatibility
        self.vision_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        if self.is_student:
            self._init_student_encoder()
        else:
            self._init_teacher_encoder()
    
    def _init_teacher_encoder(self):
        """Initialize teacher vision encoder (standard ViT)"""
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vision_encoder.to(self.device)
        self._set_requires_grad(self.vision_encoder, self.config.finetune_vlm)

    def _init_student_encoder(self):
        """Initialize student vision encoder (lightweight)"""
        vision_type = getattr(self.config, 'student_vision_type', 'tiny_vit')
        self.vision_encoder = BackboneFactory.create_backbone(vision_type, self.config)
        self.vision_encoder.to(self.device)
        self._set_requires_grad(self.vision_encoder, True)  # Student model is always trainable

    def _set_requires_grad(self, model, value):
        """Set requires_grad for all model parameters"""
        for param in model.parameters():
            param.requires_grad = value

    def get_vision_embeddings(self, images, mask_ratio=0.):
        """
        Extract visual embeddings, optional masking
        
        Parameters:
            images: input image
            mask_ratio: patch ratio of mask (0.0 means no mask)
            
        Returns:
            torch.Tensor: visual embeddings
        """
        # Determine if mask should be used
        should_use_mask = mask_ratio > 0. and self.is_student
        
        # Determine model type
        is_cnn_model = any(model_type in str(type(self.vision_encoder)).lower() 
                        for model_type in ['mobilenet', 'efficientnet'])
        
        if should_use_mask:
            if is_cnn_model:
                return self.mask_forward_cnn(images, mask_ratio)
            else:
                return self.mask_forward_transformer(images, mask_ratio)
        else:
            # Standard forward pass - process based on specific model type
            processed_images = self.vision_processor(images=images, return_tensors="pt")
            pixel_values = processed_images.pixel_values.to(self.device)
            
            if is_cnn_model:
                # For CNN model, directly process image
                outputs = self.vision_encoder(pixel_values)
                return outputs
            else:
                # For regular ViT model, need full input dictionary
                inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)
                outputs = self.vision_encoder(**inputs)
                return outputs.last_hidden_state.mean(dim=1)  # Average pool patches
    
    def apply_patch_mask(self, x, mask_ratio):
        """
        Apply mask to input features
        
        Parameters:
            x: input features [B, L, D]
            mask_ratio: mask ratio
            
        Returns:
            masked features
        """
        B, L, D = x.shape
        
        # Determine number of patches to keep
        num_keep = int(L * (1 - mask_ratio))
        
        # Generate random indices for each sample
        keep_indices = []
        for i in range(B):
            # Randomly select indices to keep
            indices = torch.randperm(L, device=x.device)[:num_keep]
            keep_indices.append(indices)
        
        # Create masked features
        x_masked = x.clone()
        
        # Apply mask to each sample
        for i in range(B):
            # Create mask (1 means masked, 0 means kept)
            mask = torch.ones(L, device=x.device, dtype=torch.bool)
            mask[keep_indices[i]] = False
            
            # Set masked positions to 0
            x_masked[i][mask] = 0
            
        return x_masked
    
    def mask_forward_transformer(self, images, mask_ratio):
        """
        Forward pass with masked images using transformer-based model
        
        Parameters:
            images: input image
            mask_ratio: mask ratio
            
        Returns:
            masked output features
        """
        # Process image
        inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)
        
        # ViT processing flow
        # 1. Get patch embeddings
        embeddings = self.vision_encoder.embeddings(inputs.pixel_values)
        
        # 2. Apply mask
        masked_embeddings = self.apply_patch_mask(embeddings, mask_ratio)
        
        # 3. Forward pass through encoder
        encoder_outputs = self.vision_encoder.encoder(masked_embeddings)
        sequence_output = encoder_outputs[0]
        
        # 4. Pool to get final features
        pooled_output = sequence_output.mean(dim=1)
        
        return pooled_output
    
    def mask_forward_cnn(self, images, mask_ratio):
        """
        Apply mask to patches of images using CNN-based model
            
        Parameters:
            images: input image [B, C, H, W]
            mask_ratio: mask ratio
                
        Returns:
            Tensor: masked feature embeddings
        """
        B, C, H, W = images.shape
        
        # Create mask for image (divide image into patch grid)
        patch_size = 8  # Define appropriate patch size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w
        
        # Determine number of patches to keep
        num_keep = int(total_patches * (1 - mask_ratio))
        
        # Create masked image batch
        masked_images = images.clone()
        
        for i in range(B):
            # Generate random indices for patches to keep
            keep_indices = torch.randperm(total_patches)[:num_keep]
            
            # Create boolean mask (True = masked)
            mask = torch.ones(total_patches, dtype=torch.bool, device=images.device)
            mask[keep_indices] = False
            
            # Reshape mask to match patches
            mask = mask.reshape(num_patches_h, num_patches_w)
            
            # Expand mask to full image resolution
            mask = mask.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
            
            # Apply mask to each channel
            for c in range(C):
                masked_images[i, c][mask] = 0.0
        
        # Process masked images
        processed_images = self.vision_processor(images=masked_images, return_tensors="pt")
        pixel_values = processed_images.pixel_values.to(self.device)
        
        # Forward pass through CNN
        outputs = self.vision_encoder(pixel_values)
        
        return outputs