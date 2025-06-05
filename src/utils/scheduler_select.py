import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    ReduceLROnPlateau, 
    StepLR, 
    ExponentialLR,
    CyclicLR,
    OneCycleLR,
    LambdaLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR
)
from src.utils import config as cfg
scheduler_params_dict = {
    "cosine": {
        "T_max": cfg.NUM_EPOCHS,
        "eta_min": 1e-6,
    },
    "cosine_warm_restarts": {
        "T_0": 5,
        "T_mult": 2,
        "eta_min": 1e-6,
    },
    "reduce_on_plateau": {
        "mode": "min",
        "factor": 0.5,
        "patience": 3,
        "verbose": True,
        "min_lr": 1e-6,
    },
    "step": {
        "step_size": 5,
        "gamma": 0.5,
    },
    "multistep": {
        "milestones": [5, 10, 15],
        "gamma": 0.1,
    },
    "exponential": {
        "gamma": 0.95,
    },
    "cyclic": {
        "base_lr": 1e-5,
        "max_lr": 5e-4,
        "step_size_up": 2000,
        "mode": "triangular2",
    },
    "onecycle": {
        "max_lr": 1e-3,
        "epochs": cfg.NUM_EPOCHS,
        "pct_start": 0.3,
        "div_factor": 25,
        "final_div_factor": 10000,
    },
    "lambda": {
        "lr_lambda": lambda epoch: 0.95 ** epoch,
    },
    "gan_custom": {
        "warmup_epochs": 2,
        "decay_start": 5,
        "decay_factor": 0.5,
    },
}

def get_scheduler(optimizer, scheduler_type, params, steps_per_epoch=None):
    """
    GAN için optimize edilmiş scheduler factory
    """
    
    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer, 
            T_max=params.get("T_max", 10),
            eta_min=params.get("eta_min", 1e-6)
        )
    
    elif scheduler_type == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.get("T_0", 5),
            T_mult=params.get("T_mult", 2),
            eta_min=params.get("eta_min", 1e-6)
        )
    
    elif scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=params.get("mode", "min"),
            factor=params.get("factor", 0.5),
            patience=params.get("patience", 3),
            min_lr=params.get("min_lr", 1e-6)
        )
    
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=params.get("step_size", 5),
            gamma=params.get("gamma", 0.5)
        )
    
    elif scheduler_type == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=params.get("milestones", [5, 10, 15]),
            gamma=params.get("gamma", 0.1)
        )
    
    elif scheduler_type == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=params.get("gamma", 0.95)
        )
    
    elif scheduler_type == "cyclic":
        return CyclicLR(
            optimizer,
            base_lr=params.get("base_lr", 1e-5),
            max_lr=params.get("max_lr", 5e-4),
            step_size_up=params.get("step_size_up", 2000),
            mode=params.get("mode", "triangular2"),
            cycle_momentum=False  # GAN için önemli
        )
    
    elif scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("OneCycleLR requires steps_per_epoch")
        return OneCycleLR(
            optimizer,
            max_lr=params.get("max_lr", 1e-3),
            epochs=params.get("epochs", 10),
            steps_per_epoch=steps_per_epoch,
            pct_start=params.get("pct_start", 0.3),
            div_factor=params.get("div_factor", 25),
            final_div_factor=params.get("final_div_factor", 10000)
        )
    
    elif scheduler_type == "lambda":
        lr_lambda = params.get("lr_lambda", lambda epoch: 0.95 ** epoch)
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_type == "gan_custom":
        # GAN'lar için özel scheduler
        return GANScheduler(optimizer, **params)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class GANScheduler:
    """
    GAN eğitimi için özel scheduler
    İlk epochlarda yüksek LR, sonra yavaş azalma
    """
    def __init__(self, optimizer, warmup_epochs=2, decay_start=5, decay_factor=0.5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.decay_start = decay_start
        self.decay_factor = decay_factor
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
        elif self.current_epoch >= self.decay_start:
            # Decay phase
            decay_epochs = self.current_epoch - self.decay_start
            lr = self.initial_lr * (self.decay_factor ** decay_epochs)
        else:
            # Stable phase
            lr = self.initial_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]