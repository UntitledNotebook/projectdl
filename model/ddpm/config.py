# config.py

# -- Data Configuration --
data_config = {
    "data_file_path": "/home/f3f3x0/Documents/projectdl/model/ddpm/data.npy", 
    "bit_representation_length": 5,  # Channels for x_t, and also for x_self_cond if used
    "image_spatial_shape": (32, 32, 32), 
    "batch_size": 8,
    "num_workers": 4,
    "shuffle_data": True,
}

# -- Model (UNet3D) Configuration --
# The UNet3D model itself is now simpler. Its `input_channels` will be set in main.py
# based on whether self-conditioning is active in the diffusion process.
# `output_channels` will be data_config["bit_representation_length"].
model_config = {
    # input_channels_xt (for x_t) and num_self_condition_channels_unet are effectively superseded
    # by data_config["bit_representation_length"] and diffusion_config["self_condition_diffusion_process"]
    # The UNet will receive a single input tensor with combined channels if self-cond is on.

    "model_channels": 64,       
    "channel_mults": (1, 2, 4), 
    "num_residual_blocks_per_stage": 2,
    "time_embedding_dim": 128,
    "time_mlp_hidden_dim": 512,
    "time_final_emb_dim": 512, 
    "attention_resolutions_indices": (1,), 
    "attention_heads": 8,
    "dropout": 0.1,
    "groups": 8,
    "initial_conv_kernel_size": 3,
}

# -- Diffusion Process Configuration --
diffusion_config = {
    "analog_bit_scale": 1.0, 
    "self_condition_diffusion_process": True, # Master flag for enabling self-conditioning behavior in BitDiffusion
    "gamma_ns": 0.0002,
    "gamma_ds": 0.00025,
}

# -- Training Configuration --
train_config = {
    "num_train_epochs": 200,
    "learning_rate": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-6, 
    "adam_epsilon": 1e-8,
    "lr_scheduler_type": "cosine", 
    "lr_warmup_steps": 500,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "no",  
    "output_dir": "outputs/diffusion_minecraft_simplified_v2", 
    "seed": 42,
    
    "log_with_wandb": True,
    "wandb_project_name": "minecraft_bit_diffusion_simplified_v2", 
    "wandb_entity_name": None, 
    "wandb_group": "ddpm_self_cond_agnostic_unet",
    
    "sampling_steps_train": 50, 
    "time_difference_td": 0.0, 
    "num_samples_to_log": 4,   
    "log_samples_every_n_steps": 10000, 
    "ema_decay": 0.9999, 
}

# No specific consistency checks needed here for UNet input channels,
# as it's now handled dynamically in main.py based on diffusion_config.

