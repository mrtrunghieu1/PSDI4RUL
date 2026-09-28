# vlm_manager.py
import sys
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from transformers import ViTMAEModel, ViTMAEConfig
from transformers import ViTImageProcessor, ViTModel, ViTConfig
from PIL import Image  # Add this line for MAE model processing      
import numpy as np  # Add this line for array processing
# Import custom module
from src.psdi_kd.vlm_custom import CustomVLM
# Add imports for EfficientNet and ResNet
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision import transforms

class VLMManager:
    """
    LVM manager class for handling different types of visual language models, providing standardized interfaces.
    Supports knowledge distillation for teacher and student models.
    """
    def __init__(self, config):
        self.config = config
        self.device = self._acquire_device()
        
        # Determine running mode
        self.is_student = getattr(config, 'is_student', False)
        self.use_distillation = getattr(config, 'use_distillation', False)
        self.vlm_type = config.vlm_type.lower()
        
        # Initialize LVM
        self._init_vlm()
        
    def _acquire_device(self):
        """Get appropriate device (GPU/CPU)"""
        if self.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.config.gpu}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _init_vlm(self):
        """Initialize appropriate LVM according to configuration"""
        # Print current mode
        mode_str = "Distillation" if self.use_distillation else "Standard"
        model_type = "Student" if self.is_student else "Teacher"
        print(f"Initializing LVM in {mode_str} mode with {model_type} model")
        
        # Use factory pattern to create model
        if self.is_student and self.use_distillation:
            self._init_student_vlm()
        else:
            model_initializers = {
                "clip": self._init_clip,
                "custom": self._init_custom,
                "mae_base": self._init_mae_base,  # Add MAE base support
                "efficientnet_b3": self._init_efficientnet_b3,  # Add EfficientNet-B3 support
                "resnet101": self._init_resnet101,  # Add ResNet 101 support
            }
            
            initializer = model_initializers.get(self.vlm_type)
            if initializer:
                initializer()
            else:
                valid_types = list(model_initializers.keys())
                raise ValueError(f"Unsupported vlm_type: {self.vlm_type}. Please choose from {valid_types}.")
        
        # Move model to device and print statistics
        self.model.to(self.device)
        learnable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"LVM learnable parameters: {learnable_params:,}")
        print(f"LVM hidden layer size: {self.hidden_size}")
    
    # LVM initialization function
    def _init_clip(self):
        """Initialize CLIP model"""
        CLIP_ARCH = 'openai/clip-vit-base-patch32'
        self.processor = CLIPProcessor.from_pretrained(CLIP_ARCH)
        self.model = CLIPModel.from_pretrained(CLIP_ARCH, output_hidden_states=True)
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        self.hidden_size = 512
        self.fusion_dim = self.hidden_size

    def _init_mae_base(self):
        """Initialize MAE model (load from Hugging Face)
        Supports three sizes: base, large, huge
        """
        
        # Determine MAE model size
        mae_size = getattr(self.config, 'mae_size', 'base')
        
        # Determine model path and hidden layer size based on size
        model_mappings = {
            'base': {
                'path': 'facebook/vit-mae-base',
                'hidden_size': 768
            },
            'large': {
                'path': 'facebook/vit-mae-large',
                'hidden_size': 1024
            },
            'huge': {
                'path': 'facebook/vit-mae-huge',
                'hidden_size': 1280
            }
        }
        
        if mae_size not in model_mappings:
            print(f"Warning: Unknown MAE size '{mae_size}', using default 'base'")
            mae_size = 'base'
        
        # Get current size configuration
        model_config = model_mappings[mae_size]
        
        # Determine pretrained model path
        if hasattr(self.config, 'mae_pretrained_path') and self.config.mae_pretrained_path:
            pretrained_path = self.config.mae_pretrained_path
            print(f"Loading MAE-{mae_size} pretrained model from specified path: {pretrained_path}")
        else:
            pretrained_path = model_config['path']
            print(f"Using default MAE-{mae_size} pretrained model: {pretrained_path}")
        
        # Load model
        try:
            self.model = ViTMAEModel.from_pretrained(pretrained_path)
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            print("Using default configuration to initialize MAE model (no pretrained weights)")
            mae_config = ViTMAEConfig(hidden_size=model_config['hidden_size'])
            self.model = ViTMAEModel(mae_config)
        
        # Set parameters to be trainable based on finetune_vlm flag
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        
        # Set hidden layer dimension
        self.hidden_size = self.model.config.hidden_size
        self.fusion_dim = self.hidden_size
        
        print(f"Initialized MAE-{mae_size} model, hidden layer size: {self.hidden_size}")
        
    def _init_custom(self):
        """Initialize custom LVM"""
        self.model = CustomVLM(self.config)
        self.hidden_size = self.model.hidden_size

    def _init_student_vlm(self):
        """Initialize student LVM model (lightweight)"""
        self.config.is_student = True 
        self.model = CustomVLM(self.config)
        self.hidden_size = self.model.hidden_size
        self.fusion_dim = self.hidden_size

    def _init_efficientnet_b3(self):
        """Initialize EfficientNet-B3 model"""
        print("Initializing EfficientNet-B3 model")
        self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        
        # Modify classifier to unify feature dimension
        self.model.classifier = nn.Identity()  # Remove classifier layer
        
        # EfficientNet-B3 feature output dimension
        self.hidden_size = 1536  # EfficientNet-B3 feature dimension
        self.fusion_dim = self.hidden_size
        
        # Set required image transformation
        self.transform = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Set parameters to be trainable based on finetune_vlm flag
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        
        print(f"EfficientNet-B3 initialized, feature dimension: {self.hidden_size}")
    
    def _init_resnet101(self):
        """Initialize ResNet 101 model"""
        print("Initializing ResNet 101 model")
        self.model = resnet101(weights=ResNet101_Weights.DEFAULT)
        
        # Remove last fully connected layer, only keep feature extraction
        self.model.fc = nn.Identity()
        
        # ResNet 101 feature output dimension
        self.hidden_size = 2048  # ResNet 101 feature dimension
        self.fusion_dim = self.hidden_size
        
        # Set required image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Set parameters to be trainable based on finetune_vlm flag
        self._set_requires_grad(self.model, self.config.finetune_vlm)
        
        print(f"ResNet 101 initialized, feature dimension: {self.hidden_size}")

    def _set_requires_grad(self, model, value):
        """Set requires_grad for all model parameters"""
        for param in model.parameters():
            param.requires_grad = value
        for child in model.children():
            self._set_requires_grad(child, value)

    # Standardized interface processing method
    def process_inputs_vision_only(self, B, images):
        """Process only visual input, no text"""
        try: 
            processors = {
                "clip": self._process_clip_inputs_vision_only,
                "custom": self._process_custom_inputs_vision_only,
                "mae_base": self._process_mae_base_inputs_vision_only,  # Add MAE base support
                "efficientnet_b3": self._process_efficientnet_inputs_vision_only,  # Add EfficientNet-B3 support
                "resnet101": self._process_resnet_inputs_vision_only,  # Add ResNet 101 support
            }
            
            processor = processors.get(self.vlm_type)
            if processor:
                return processor(B, images)
            else:
                raise ValueError(f"No processor for LVM_type: {self.vlm_type}")
        except Exception as e:
            print(f"Error processing input: {e}")
            print(f"Image shape: {images.shape}")
            raise e

    def process_inputs_vision_only_masked(self, B, images, mask_ratio=0.5):
        """Process visual input with masking, unify masking processing for different model types"""
        # Check if mask should be used (based on feature_w parameter)
        feature_w = getattr(self.config, 'feature_w', 0.0)
        
        # Only apply mask when feature_w is greater than 0
        if feature_w <= 0:
            return self.process_inputs_vision_only(B, images)
            
        # For custom model and student model, use their built-in masking method
        if (self.vlm_type == "custom" or self.is_student) and hasattr(self.model, 'get_vision_embeddings'):
            return self.model.get_vision_embeddings(images, mask_ratio=mask_ratio)
        
        # For other models (including MAE), use unified masking processing method
        features = self.process_inputs_vision_only(B, images)
        
        # Generate random mask
        mask = torch.rand_like(features) < mask_ratio
        features_masked = features.clone()
        features_masked[mask] = 0
        
        return features_masked

    # Specific processor for each LVM type
    def _process_clip_inputs_vision_only(self, B, images):
        """Process only CLIP visual input"""
        dummy_prompts = [""] * B
        encoding = self.processor(images=images, text=dummy_prompts, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding, output_hidden_states=True)
        return outputs.image_embeds  # [B, hidden_size]
    
    def _process_custom_inputs_vision_only(self, B, images):
        """Process only custom visual input"""
        return self.model.get_vision_embeddings(images)  # [B, hidden_size]
    
    def _process_mae_base_inputs_vision_only(self, B, images):
        """Process only MAE visual input, no text
        
        Parameters:
            B: batch size
            images: input images
            
        Returns:
            visual feature embeddings
        """
        
        # Determine MAE model size
        mae_size = getattr(self.config, 'mae_size', 'base')
        
        # Determine model path for image processor
        if hasattr(self.config, 'mae_pretrained_path') and self.config.mae_pretrained_path:
            processor_path = self.config.mae_pretrained_path
        else:
            processor_paths = {
                'base': 'facebook/vit-mae-base',
                'large': 'facebook/vit-mae-large',
                'huge': 'facebook/vit-mae-huge'
            }
            processor_path = processor_paths.get(mae_size, 'facebook/vit-mae-base')
        
        # Create image processor (if not already created)
        if not hasattr(self, 'image_processor'):
            self.image_processor = ViTImageProcessor.from_pretrained(processor_path)
        
        # Convert images from uint8 [0, 255] to PIL image list
        images_cpu = images.cpu().numpy()
        pil_images = []
        for i in range(images.shape[0]):
            img = images_cpu[i].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            pil_images.append(Image.fromarray(img))
        
        # Prepare input using ViT image processor
        pixel_values = self.image_processor(pil_images, return_tensors="pt").pixel_values.to(self.device)
        
        # Use MAE model to encode images - removed mask_ratio parameter
        with torch.no_grad():
            outputs = self.model(
                pixel_values, 
                output_hidden_states=True,
                return_dict=True
            )
        
        # Select feature representation based on configuration
        if hasattr(self.config, 'use_cls_token') and self.config.use_cls_token:
            # Use CLS token
            last_hidden_state = outputs.last_hidden_state
            return last_hidden_state[:, 0]  # Return CLS token
        else:
            # Use average pooling of all tokens
            last_hidden_state = outputs.last_hidden_state
            # Exclude CLS token (first position) for average pooling
            patch_tokens = last_hidden_state[:, 1:]
            return patch_tokens.mean(dim=1)  # Average pool patch embeddings

    def process_inputs_vision_patch_tokens(self, B, images):
        """Process visual input and return patch-level tokens (no pooling).

        Only supported for MAE-based models.
        Returns: [B, N_patches, hidden_size]  (e.g. [B, 196, 768] for MAE-base)
        """
        if self.vlm_type != "mae_base":
            raise ValueError(
                f"Patch token extraction not supported for vlm_type={self.vlm_type}. "
                "Only mae_base is supported."
            )
        return self._process_mae_base_patch_tokens(B, images)

    def _process_mae_base_patch_tokens(self, B, images):
        """Extract MAE patch tokens [B, N_patches, 768] without pooling."""
        # Reuse the same image preprocessing as the pooled version
        mae_size = getattr(self.config, 'mae_size', 'base')
        if hasattr(self.config, 'mae_pretrained_path') and self.config.mae_pretrained_path:
            processor_path = self.config.mae_pretrained_path
        else:
            processor_paths = {
                'base': 'facebook/vit-mae-base',
                'large': 'facebook/vit-mae-large',
                'huge': 'facebook/vit-mae-huge'
            }
            processor_path = processor_paths.get(mae_size, 'facebook/vit-mae-base')

        if not hasattr(self, 'image_processor'):
            self.image_processor = ViTImageProcessor.from_pretrained(processor_path)

        images_cpu = images.cpu().numpy()
        pil_images = []
        for i in range(images.shape[0]):
            img = images_cpu[i].transpose(1, 2, 0)
            pil_images.append(Image.fromarray(img))

        pixel_values = self.image_processor(
            pil_images, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                pixel_values,
                output_hidden_states=True,
                return_dict=True
            )

        # last_hidden_state: [B, 1+N_patches, hidden_size]
        # index 0 = CLS token, 1: = patch tokens
        last_hidden_state = outputs.last_hidden_state
        patch_tokens = last_hidden_state[:, 1:]            # [B, N_patches, hidden_size]
        pooled_feat = patch_tokens.mean(dim=1)             # [B, hidden_size]
        return pooled_feat, patch_tokens

    def _process_efficientnet_inputs_vision_only(self, B, images):
        """Process only EfficientNet visual input, no text
        
        Parameters:
            B: batch size
            images: input images
            
        Returns:
            visual feature embeddings
        """
        # First ensure image is float type
        if images.dtype != torch.float32:
            images = images.float()
            # If uint8 type (0-255), normalize to 0-1 range
            if images.max() > 1.0:
                images = images / 255.0
        
        # Apply image transformation
        transformed_images = self.transform(images)
        
        # Forward propagation to get features
        with torch.no_grad():
            # Forward propagation to get features
            features = self.model.features(transformed_images)
            
            # Apply global average pooling
            x = self.model.avgpool(features)
            x = torch.flatten(x, 1)
            
        return x  # [B, hidden_size]
    
    def _process_resnet_inputs_vision_only(self, B, images):
        """Process only ResNet visual input, no text
        
        Parameters:
            B: batch size
            images: input images
            
        Returns:
            visual feature embeddings
        """
        # First ensure image is float type
        if images.dtype != torch.float32:
            images = images.float()
            # If uint8 type (0-255), normalize to 0-1 range
            if images.max() > 1.0:
                images = images / 255.0
        
        # Apply image transformation
        transformed_images = self.transform(images)
        
        # Forward propagation to get features
        with torch.no_grad():
            x = self.model.conv1(transformed_images)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            
        return x  # [B, hidden_size]