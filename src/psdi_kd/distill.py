# distill.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LearnableWeights(nn.Module):
    """Learnable distillation weight module"""
    def __init__(self, num_weights=4, init_values=None):
        super(LearnableWeights, self).__init__()
        
        if init_values is None:
            init_values = [0.01, 1.0, 0.5, 0.01]  # feature, fcst, recon, att
        
        # Use log parameterization to ensure weights are positive.
        # Clamp init to avoid log(0) -> -inf.
        init_tensor = torch.tensor(init_values, dtype=torch.float32)
        init_tensor = torch.clamp(init_tensor, min=1e-8)
        self.log_weights = nn.Parameter(torch.log(init_tensor))
        
        # Optional weight regularization
        self.weight_regularization = 0.001
        
    def forward(self):
        """Return normalized weights"""
        weights = torch.exp(self.log_weights)
        return weights
        
    def get_regularization_loss(self):
        """Return weight regularization loss to prevent weights from being too large"""
        weights = torch.exp(self.log_weights)
        return self.weight_regularization * torch.sum(weights ** 2)

class AdaptiveTemperature(nn.Module):
    """Adaptive temperature parameter"""
    def __init__(self, init_temp=4.0, min_temp=1.0, max_temp=10.0):
        super(AdaptiveTemperature, self).__init__()
        self.min_temp = min_temp
        self.max_temp = max_temp
        # Use sigmoid to ensure temperature is within a reasonable range
        self.raw_temp = nn.Parameter(torch.tensor(init_temp, dtype=torch.float32))
        
    def forward(self):
        # Use sigmoid to map the original value to the [min_temp, max_temp] range
        normalized = torch.sigmoid(self.raw_temp)
        return self.min_temp + normalized * (self.max_temp - self.min_temp)

class MultiScaleFeatureAligner(nn.Module):
    """Multi-scale feature aligner"""
    def __init__(self, teacher_dim, student_dim, num_scales=3):
        super(MultiScaleFeatureAligner, self).__init__()
        
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.num_scales = num_scales
        
        # Main projection layer
        self.main_proj = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.LayerNorm(teacher_dim),
            nn.GELU()
        )
        
        # Multi-scale projection layers
        self.scale_projections = nn.ModuleList()
        for i in range(num_scales):
            hidden_dim = max(student_dim, teacher_dim) // (2 ** i)
            hidden_dim = max(hidden_dim, 64)  # Ensure minimum dimension
            
            proj = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, teacher_dim),
                nn.LayerNorm(teacher_dim)
            )
            self.scale_projections.append(proj)
            
        # Scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales + 1) / (num_scales + 1))
        
    def forward(self, student_features):
        # Main projection
        main_out = self.main_proj(student_features)
        
        # Multi-scale projection
        scale_outs = [proj(student_features) for proj in self.scale_projections]
        
        # Weighted combination
        normalized_weights = F.softmax(self.scale_weights, dim=0)
        
        final_out = normalized_weights[0] * main_out
        for i, scale_out in enumerate(scale_outs):
            final_out += normalized_weights[i + 1] * scale_out
            
        return final_out

class DistillationLoss:
    """Improved knowledge distillation loss, supporting self-learning hyperparameters"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}') if args.use_gpu else torch.device('cpu')
        
        # Check if adaptive weights are enabled
        self.enable_adaptive_weights = getattr(args, 'enable_adaptive_weights', True)
        self.use_degradation_weighted_fcst = getattr(
            args, 'use_degradation_weighted_fcst', False
        )
        self.degradation_lambda = getattr(args, 'degradation_lambda', 2.0)
        self.degradation_gamma = getattr(args, 'degradation_gamma', 2.0)
        
        # Store original fixed weights (from args)
        self.fixed_weights = torch.tensor([
            getattr(args, 'feature_w', 0.01),
            getattr(args, 'fcst_w', 1.0),
            getattr(args, 'recon_w', 0.5),
            getattr(args, 'att_w', 0.01)
        ], device=self.device, dtype=torch.float32)
        
        self.fixed_temperature = torch.tensor(
            getattr(args, 'init_temperature', 4.0), 
            device=self.device, dtype=torch.float32
        )
        
        # Initialize learnable weights (always created, but only used if enable_adaptive_weights is True)
        self._init_learnable_weights(args)
        
        # Initialize adaptive temperature (always created, but only used if enable_adaptive_weights is True)
        self._init_adaptive_temperature(args)
        
        # Initialize feature aligner
        self._init_feature_aligner(args)
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Initialize loss balancer
        self._init_loss_balancer()
        
        # Training statistics
        self.loss_history = {
            'feature_loss': [],
            'fcst_loss': [],
            'recon_loss': [],
            'att_loss': []
        }
        
        # Print current mode
        if self.enable_adaptive_weights:
            print("Distillation weight mode: adaptive learning weights")
            print(f"Distillation temperature mode: adaptive learning temperature (initial value: {getattr(args, 'init_temperature', 4.0)})")
        else:
            print(f"Distillation weight mode: fixed weights {self.fixed_weights.tolist()}")
            print(f"Distillation temperature mode: fixed temperature {self.fixed_temperature.item()}")
        
    def _init_learnable_weights(self, args):
        """Initialize learnable loss weights (always created)"""
        init_weights = self.fixed_weights.cpu().tolist()
        
        self.learnable_weights = LearnableWeights(
            num_weights=4, 
            init_values=init_weights
        ).to(self.device)
        
    def _init_adaptive_temperature(self, args):
        """Initialize adaptive temperature parameter (always created)"""
        init_temp = getattr(args, 'init_temperature', 4.0)
        
        self.adaptive_temp = AdaptiveTemperature(
            init_temp=init_temp,
            min_temp=1.0,
            max_temp=10.0
        ).to(self.device)
        
    def _init_feature_aligner(self, args):
        """Initialize feature aligner (projects student features to teacher dimension)"""
        teacher_dim = getattr(args, 'teacher_hidden_size', 512)
        student_dim = getattr(args, 'student_hidden_size', getattr(args, 'd_model', 128))

        if teacher_dim != student_dim:
            # MultiScaleFeatureAligner projects student_dim → teacher_dim
            self.feature_aligner = MultiScaleFeatureAligner(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                num_scales=3
            ).to(self.device)
        else:
            self.feature_aligner = nn.Identity().to(self.device)
            
    def _init_loss_functions(self):
        """Initialize loss functions"""
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
    def _init_loss_balancer(self):
        """Initialize loss balancer"""
        self.loss_scale_factor = 1.0
        self.loss_momentum = 0.9
        self.running_loss_scales = {
            'feature_loss': 1.0,
            'fcst_loss': 1.0,
            'recon_loss': 1.0,
            'att_loss': 1.0
        }
        
    def _compute_feature_distillation_loss(self, student_features, teacher_features, temperature):
        """Compute improved feature distillation loss"""
        if student_features is None or teacher_features is None:
            return torch.tensor(0.0, device=self.device)
            
        # Feature alignment
        aligned_student = self.feature_aligner(student_features)
        
        # 1. MSE loss
        mse_loss = self.mse_loss(aligned_student, teacher_features)
        
        # 2. Cosine similarity loss
        target = torch.ones(aligned_student.size(0), device=self.device)
        cosine_loss = self.cosine_loss(
            aligned_student, teacher_features, target
        )
        
        # 3. KL divergence loss (using temperature parameter)
        student_probs = F.softmax(aligned_student / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_features / temperature, dim=-1)
        kl_loss = F.kl_div(
            F.log_softmax(aligned_student / temperature, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Combined loss
        total_feature_loss = 0.4 * mse_loss + 0.3 * cosine_loss + 0.3 * kl_loss
        
        return total_feature_loss
        
    def _compute_attention_distillation_loss(self, student_att, teacher_att, temperature):
        """Compute attention distillation loss"""
        if student_att is None or teacher_att is None:
            return torch.tensor(0.0, device=self.device)
            
        # Ensure attention weight shapes are consistent
        if student_att.shape != teacher_att.shape:
            # If shapes are different, try adjusting
            min_dim = min(student_att.size(-1), teacher_att.size(-1))
            student_att = student_att[..., :min_dim]
            teacher_att = teacher_att[..., :min_dim]
            
        # KL divergence loss
        student_att_soft = F.softmax(student_att / temperature, dim=-1)
        teacher_att_soft = F.softmax(teacher_att / temperature, dim=-1)
        
        att_loss = F.kl_div(
            F.log_softmax(student_att / temperature, dim=-1),
            teacher_att_soft,
            reduction='batchmean'
        ) * (temperature ** 2)

        return att_loss
        
    def _update_loss_balancer(self, loss_items):
        """Update loss balancer"""
        for loss_name, loss_value in loss_items.items():
            if loss_name in self.running_loss_scales and loss_value > 0:
                # Use exponential moving average to update loss scale
                current_scale = abs(loss_value)
                self.running_loss_scales[loss_name] = (
                    self.loss_momentum * self.running_loss_scales[loss_name] + 
                    (1 - self.loss_momentum) * current_scale
                )
                
    def compute_total_loss(self, ts_enc, prompt_enc, ts_out, prompt_out, ts_att=None,
                           prompt_att=None, real=None, batch_meta=None):
        """
        Compute total distillation loss
        
        Parameters:
            ts_enc: student model feature encoding [B, D]
            prompt_enc: teacher model feature encoding [B, D]
            ts_out: student model prediction output [B, L, N]
            prompt_out: teacher model prediction output [B, L, N]
            ts_att: student model attention weights
            prompt_att: teacher model attention weights
            real: real target values [B, L, N]
            
        Returns:
            total_loss: total loss
            loss_items: dictionary containing loss components
        """
        # Get current weights and temperature
        if self.enable_adaptive_weights:
            # Use adaptive learning weights and temperature
            weights = self.learnable_weights()
            temperature = self.adaptive_temp()
        else:
            # Use fixed weights and temperature (ignore learned values)
            weights = self.fixed_weights
            temperature = self.fixed_temperature
        
        loss_items = {}
        losses = {}
        
        # 1. Feature distillation loss
        feature_loss = self._compute_feature_distillation_loss(
            ts_enc, prompt_enc, temperature
        )
        losses['feature_loss'] = feature_loss
        loss_items['feature_loss'] = feature_loss.item()
        
        # 2. Prediction task loss
        if ts_out is not None and real is not None:
            elementwise_fcst_loss = F.smooth_l1_loss(ts_out, real, reduction='none')
            if self.use_degradation_weighted_fcst:
                deg_weight = 1.0 + self.degradation_lambda * torch.pow(
                    torch.clamp(1.0 - real, min=0.0, max=1.0),
                    self.degradation_gamma
                )
                fcst_loss = (deg_weight * elementwise_fcst_loss).mean()
                loss_items['deg_weight_mean'] = deg_weight.mean().item()
            else:
                fcst_loss = elementwise_fcst_loss.mean()
                loss_items['deg_weight_mean'] = 1.0
            losses['fcst_loss'] = fcst_loss
            loss_items['fcst_loss'] = fcst_loss.item()
        else:
            losses['fcst_loss'] = torch.tensor(0.0, device=self.device)
            loss_items['fcst_loss'] = 0.0
            loss_items['deg_weight_mean'] = 1.0
            
        # 3. Reconstruction loss (difference between teacher model prediction and real value)
        if prompt_out is not None and real is not None:
            recon_loss = self.smooth_l1_loss(prompt_out, real)
            losses['recon_loss'] = recon_loss
            loss_items['recon_loss'] = recon_loss.item()
        else:
            losses['recon_loss'] = torch.tensor(0.0, device=self.device)
            loss_items['recon_loss'] = 0.0
            
        # 4. Attention distillation loss
        att_loss = self._compute_attention_distillation_loss(
            ts_att, prompt_att, temperature
        )
        losses['att_loss'] = att_loss
        loss_items['att_loss'] = att_loss.item()

        # Update loss balancer
        self._update_loss_balancer(loss_items)
        
        # Compute weighted total loss
        total_loss = (
            weights[0] * losses['feature_loss'] +
            weights[1] * losses['fcst_loss'] +
            weights[2] * losses['recon_loss'] +
            weights[3] * losses['att_loss']
        )
        
        # Add weight regularization (only applied in adaptive weight mode)
        if self.enable_adaptive_weights:
            weight_reg_loss = self.learnable_weights.get_regularization_loss()
        else:
            weight_reg_loss = torch.tensor(0.0, device=self.device)
        total_loss += weight_reg_loss
        
        # Record weight information
        loss_items['weights'] = weights.detach().cpu().numpy().tolist()
        loss_items['temperature'] = temperature.item()
        loss_items['weight_reg'] = weight_reg_loss.item()
        
        # Update history
        for key in ['feature_loss', 'fcst_loss', 'recon_loss', 'att_loss']:
            self.loss_history[key].append(loss_items[key])
            
        return total_loss, loss_items
        
    def get_learnable_parameters(self):
        """Return learnable parameters for optimizer"""
        params = []
        
        # Determine whether to include weights and temperature parameters based on enable_adaptive_weights
        if self.enable_adaptive_weights:
            params.extend(self.learnable_weights.parameters())
            params.extend(self.adaptive_temp.parameters())
        
        # Feature aligner parameters are always included (if they exist)
        if hasattr(self.feature_aligner, 'parameters'):
            params.extend(self.feature_aligner.parameters())
            
        return params
        
    def get_current_weights(self):
        """Get current weight values"""
        with torch.no_grad():
            if self.enable_adaptive_weights:
                weights = self.learnable_weights()
                temp = self.adaptive_temp()
            else:
                weights = self.fixed_weights
                temp = self.fixed_temperature
            
            return {
                'feature_w': weights[0].item(),
                'fcst_w': weights[1].item(),
                'recon_w': weights[2].item(),
                'att_w': weights[3].item(),
                'temperature': temp.item()
            }
            
    def print_loss_statistics(self):
        """Print loss statistics"""
        if len(self.loss_history['feature_loss']) > 10:
            print("\n=== Distillation loss statistics ===")
            for loss_name, history in self.loss_history.items():
                recent_losses = history[-10:]
                avg_loss = np.mean(recent_losses)
                print(f"{loss_name}: recent 10 steps average = {avg_loss:.6f}")
                
            weights_info = self.get_current_weights()
            print(f"Current weights: {weights_info}")
            print("=" * 50)
