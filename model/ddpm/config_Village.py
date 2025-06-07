# config.py
import time

# -- Data Configuration --
data_config = {
    "data_file_path": "/root/autodl-tmp/projectdl/data/block_id.npy", 
    "bit_representation_length": 9,  # Channels for x_t, and also for x_self_cond if used
    "image_spatial_shape": (32, 24, 32), 
    "batch_size": 16,
    "num_workers": 4,
    "shuffle_data": True,
}

# -- Diffusion Process Configuration --
diffusion_config = {
    "analog_bit_scale": 1.0,
    "self_condition_diffusion_process": True, # Master flag for enabling self-conditioning behavior in BitDiffusion
    "gamma_ns": 0.0002,
    "gamma_ds": 0.00025,
}

# -- Model (UNet3D) Configuration --
model_config = {
    "input_channels": data_config["bit_representation_length"] * 2 if diffusion_config["self_condition_diffusion_process"] else data_config["bit_representation_length"],
    "model_channels": 64,
    "output_channels": data_config["bit_representation_length"],
    "channel_mults": (1, 2, 4, 8), 
    "num_residual_blocks_per_stage": 3,
    "time_embedding_dim": 128,
    "time_mlp_hidden_dim": 512,
    "time_final_emb_dim": 512, 
    "attention_resolutions_indices": (1, 2, 3),
    "attention_type": "linear",
    "attention_heads": 8,
    "dropout": 0.1,
    "groups": 8,
    "initial_conv_kernel_size": 3,
}


# -- Training Configuration --
train_config = {
    "num_train_epochs": 20,
    "learning_rate": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-6, 
    "adam_epsilon": 1e-8,
    "lr_scheduler_type": "cosine", 
    "lr_warmup_steps": 500,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "no",  
    "output_dir": f"outputs/{time.strftime('%Y%m%d-%H%M%S')}", 
    "seed": 42,
    
    "log_with_wandb": True,
    "wandb_project_name": "projectdl", 
    "wandb_entity_name": None, 
    "wandb_group": "ddpm",
    
    "sampling_steps_train": 1000, 
    "time_difference_td": 0.0, 
    "num_samples_to_log": 4,   
    "log_samples_every_n_steps": 5000,
    "ema_decay": 0.9999, 
}