# main.py
import os
import logging
import math 

import numpy as np 
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler 
from diffusers.training_utils import EMAModel 
from tqdm.auto import tqdm

import config as TrainConfig 
from data import get_dataloader, floats_to_ids 
from model import UNet3D 
from diffusion import BitDiffusion 
from utils import log_samples_to_wandb 

logger = get_logger(__name__, log_level="INFO")


def main():
    # --- Accelerator and Output Dir Setup ---
    project_config = ProjectConfiguration(
        project_dir=TrainConfig.train_config["output_dir"],
        logging_dir=os.path.join(TrainConfig.train_config["output_dir"], "logs")
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=TrainConfig.train_config["gradient_accumulation_steps"],
        mixed_precision=TrainConfig.train_config["mixed_precision"],
        log_with="wandb" if TrainConfig.train_config["log_with_wandb"] else None,
        project_config=project_config,
    )

    # --- Logging Setup ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # --- Seed ---
    if TrainConfig.train_config["seed"] is not None:
        set_seed(TrainConfig.train_config["seed"])

    # --- Output Directory ---
    if accelerator.is_main_process:
        if TrainConfig.train_config["output_dir"] is not None:
            os.makedirs(TrainConfig.train_config["output_dir"], exist_ok=True)

    # --- WandB Initialization ---
    wandb_run_active = False
    if accelerator.is_main_process and TrainConfig.train_config["log_with_wandb"]:
        try:
            import wandb 
            wandb_kwargs = {}
            if TrainConfig.train_config.get("wandb_group"): 
                 wandb_kwargs["group"] = TrainConfig.train_config["wandb_group"]
            if TrainConfig.train_config.get("wandb_entity_name"):
                wandb_kwargs["entity"] = TrainConfig.train_config["wandb_entity_name"]
            
            serializable_config = {}
            for key, value in TrainConfig.__dict__.items():
                if not key.startswith("__"): 
                    if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                        serializable_config[key] = value
                    else:
                        serializable_config[key] = str(value) 

            if wandb.run is None: # Initialize only if no active run
                accelerator.init_trackers(
                    project_name=TrainConfig.train_config["wandb_project_name"],
                    config=serializable_config, 
                    init_kwargs={"wandb": wandb_kwargs}
                )
            wandb_run_active = wandb.run is not None # Check if run is active after attempting init
            if wandb_run_active:
                logger.info("Weights & Biases initialized and run is active.")
            else:
                logger.warning("WandB run initialization failed or no active run. Skipping WandB logging.")
                TrainConfig.train_config["log_with_wandb"] = False

        except ImportError:
            logger.warning("wandb not installed. Skipping WandB logging.")
            TrainConfig.train_config["log_with_wandb"] = False 
        except Exception as e:
            logger.error(f"Error initializing WandB: {e}. Skipping WandB logging.")
            TrainConfig.train_config["log_with_wandb"] = False


    # --- Data ---
    logger.info("Loading dataset...")
    train_dataloader = get_dataloader(
        npy_file_path=TrainConfig.data_config["data_file_path"],
        bit_length=TrainConfig.data_config["bit_representation_length"],
        batch_size=TrainConfig.data_config["batch_size"],
        shuffle=TrainConfig.data_config["shuffle_data"],
        num_workers=TrainConfig.data_config["num_workers"]
    )

    # --- Model ---
    logger.info("Initializing UNet3D model...")
    
    unet_input_channels_xt = TrainConfig.data_config["bit_representation_length"]
    unet_total_input_channels = unet_input_channels_xt
    if TrainConfig.diffusion_config["self_condition_diffusion_process"]:
        unet_total_input_channels += unet_input_channels_xt 
        logger.info(f"UNet3D will be initialized with {unet_total_input_channels} total input channels (self-conditioning active: x_t ({unet_input_channels_xt}) + x_self_cond ({unet_input_channels_xt})).")
    else:
        logger.info(f"UNet3D will be initialized with {unet_total_input_channels} total input channels (self-conditioning INACTIVE).")

    unet = UNet3D(
        input_channels=unet_total_input_channels, 
        model_channels=TrainConfig.model_config["model_channels"],
        output_channels=TrainConfig.data_config["bit_representation_length"], 
        channel_mults=TrainConfig.model_config["channel_mults"],
        num_residual_blocks_per_stage=TrainConfig.model_config["num_residual_blocks_per_stage"],
        time_embedding_dim=TrainConfig.model_config["time_embedding_dim"],
        time_mlp_hidden_dim=TrainConfig.model_config["time_mlp_hidden_dim"],
        time_final_emb_dim=TrainConfig.model_config["time_final_emb_dim"],
        attention_resolutions_indices=TrainConfig.model_config["attention_resolutions_indices"],
        attention_type=TrainConfig.model_config["attention_type"],
        attention_heads=TrainConfig.model_config["attention_heads"],
        dropout=TrainConfig.model_config["dropout"],
        groups=TrainConfig.model_config["groups"],
        initial_conv_kernel_size=TrainConfig.model_config["initial_conv_kernel_size"]
    )
    
    if accelerator.is_main_process:
        num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        logger.info(f"UNet3D initialized. Trainable parameters: {num_params / 1e6:.2f} M")
        if TrainConfig.train_config["log_with_wandb"] and wandb_run_active:
            wandb.summary["total_trainable_params_M"] = num_params / 1e6


    # --- Diffusion Process ---
    logger.info("Initializing BitDiffusion process...")
    diffusion_process = BitDiffusion(
        model=unet, 
        analog_bit_scale=TrainConfig.diffusion_config["analog_bit_scale"],
        self_condition_enabled_in_model=TrainConfig.diffusion_config["self_condition_diffusion_process"],
        gamma_ns=TrainConfig.diffusion_config["gamma_ns"],
        gamma_ds=TrainConfig.diffusion_config["gamma_ds"]
    )

    # --- EMA Model ---
    if TrainConfig.train_config["ema_decay"] > 0:
        ema_model = EMAModel(
            unet.parameters(), 
            decay=TrainConfig.train_config["ema_decay"],
        )
        logger.info(f"EMA enabled with decay {TrainConfig.train_config['ema_decay']}.")
    else:
        ema_model = None

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=TrainConfig.train_config["learning_rate"],
        betas=(TrainConfig.train_config["adam_beta1"], TrainConfig.train_config["adam_beta2"]),
        weight_decay=TrainConfig.train_config["adam_weight_decay"],
        eps=TrainConfig.train_config["adam_epsilon"],
    )

    # --- LR Scheduler ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / TrainConfig.train_config["gradient_accumulation_steps"])
    max_train_steps_calculated = TrainConfig.train_config["num_train_epochs"] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        TrainConfig.train_config["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=TrainConfig.train_config["lr_warmup_steps"] * accelerator.num_processes,
        num_training_steps=max_train_steps_calculated * accelerator.num_processes, 
    )

    # --- Prepare with Accelerator ---
    unet, optimizer, train_dataloader, lr_scheduler, diffusion_process = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, diffusion_process
    )
    if ema_model:
        ema_model.to(accelerator.device) 

    # --- Training Loop ---
    global_step = 0
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}") # This was already correct
    logger.info(f"  Num Epochs = {TrainConfig.train_config['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {TrainConfig.data_config['batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TrainConfig.data_config['batch_size'] * accelerator.num_processes * TrainConfig.train_config['gradient_accumulation_steps']}")
    logger.info(f"  Gradient Accumulation steps = {TrainConfig.train_config['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps_calculated}")
    
    progress_bar = tqdm(range(max_train_steps_calculated), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(TrainConfig.train_config["num_train_epochs"]):
        unet.train() 

        train_loss_epoch = 0.0
        for step, batch_x_start_bits in enumerate(train_dataloader):
            with accelerator.accumulate(unet): 
                t = torch.rand(batch_x_start_bits.shape[0], device=accelerator.device)
                loss = diffusion_process.p_losses(batch_x_start_bits, t)
                train_loss_epoch += loss.detach().item()

                accelerator.backward(loss)
                if accelerator.sync_gradients : 
                    if hasattr(accelerator.scaler, "_scale") and accelerator.scaler._scale is not None:
                         accelerator.clip_grad_norm_(unet.parameters(), 1.0) 
                    elif not hasattr(accelerator.scaler, "_scale"): 
                         accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if ema_model:
                    ema_model.step(unet.parameters()) 

                if TrainConfig.train_config["log_with_wandb"] and accelerator.is_main_process and wandb_run_active:
                    logs = {
                        "train_loss_step": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    accelerator.log(logs, step=global_step)
                
                if global_step > 0 and global_step % TrainConfig.train_config["log_samples_every_n_steps"] == 0:
                    if accelerator.is_main_process:
                        logger.info(f"Logging samples at global step {global_step}...")
                        
                        current_bit_length_at_sample_log = TrainConfig.data_config["bit_representation_length"]
                        unet_total_input_channels_for_temp_unet = current_bit_length_at_sample_log
                        if TrainConfig.diffusion_config["self_condition_diffusion_process"]:
                            unet_total_input_channels_for_temp_unet += current_bit_length_at_sample_log

                        temp_sampling_unet = UNet3D( 
                            input_channels=unet_total_input_channels_for_temp_unet,
                            model_channels=TrainConfig.model_config["model_channels"],
                            output_channels=current_bit_length_at_sample_log, # Output should be original bit length
                            channel_mults=TrainConfig.model_config["channel_mults"],
                            num_residual_blocks_per_stage=TrainConfig.model_config["num_residual_blocks_per_stage"],
                            time_embedding_dim=TrainConfig.model_config["time_embedding_dim"],
                            time_mlp_hidden_dim=TrainConfig.model_config["time_mlp_hidden_dim"],
                            time_final_emb_dim=TrainConfig.model_config["time_final_emb_dim"],
                            attention_resolutions_indices=TrainConfig.model_config["attention_resolutions_indices"],
                            attention_type=TrainConfig.model_config["attention_type"],
                            attention_heads=TrainConfig.model_config["attention_heads"],
                            dropout=TrainConfig.model_config["dropout"],
                            groups=TrainConfig.model_config["groups"],
                            initial_conv_kernel_size=TrainConfig.model_config["initial_conv_kernel_size"]
                        ).to(accelerator.device)
                        

                        if ema_model:
                            logger.info("Applying EMA weights to temp model for sample logging...")
                            ema_model.copy_to(temp_sampling_unet.parameters())
                        else:
                            logger.info("Using current training weights for sample logging (EMA not active)...")
                            temp_sampling_unet.load_state_dict(accelerator.unwrap_model(unet).state_dict())
                        
                        temp_sampling_unet.eval()
                        
                        sampling_diffusion_process_for_log = BitDiffusion( 
                            model=temp_sampling_unet, 
                            analog_bit_scale=TrainConfig.diffusion_config["analog_bit_scale"],
                            self_condition_enabled_in_model=TrainConfig.diffusion_config["self_condition_diffusion_process"],
                            gamma_ns=TrainConfig.diffusion_config["gamma_ns"],
                            gamma_ds=TrainConfig.diffusion_config["gamma_ds"]
                        )
                        
                        samples_analog_bits = sampling_diffusion_process_for_log.sample(
                            batch_size=TrainConfig.train_config["num_samples_to_log"],
                            shape=(
                                TrainConfig.data_config["bit_representation_length"], 
                                *TrainConfig.data_config["image_spatial_shape"]
                            ), 
                            device=accelerator.device,
                            num_steps=TrainConfig.train_config["sampling_steps_train"],
                            time_difference_td=TrainConfig.train_config["time_difference_td"]
                        )
                        log_samples_to_wandb(
                            "generated_samples", 
                            samples_analog_bits, 
                            TrainConfig.data_config["bit_representation_length"], 
                            global_step, 
                            accelerator,
                            num_to_log=TrainConfig.train_config["num_samples_to_log"],
                        )
                        unet.train() # Ensure main model is back in train mode

            logs_postfix = {"step_loss": loss.detach().item()}
            if lr_scheduler.get_last_lr(): 
                logs_postfix["lr"] = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(**logs_postfix)

            if global_step >= max_train_steps_calculated:
                break
        
        avg_epoch_loss = train_loss_epoch / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        if TrainConfig.train_config["log_with_wandb"] and accelerator.is_main_process and wandb_run_active:
            accelerator.log({"train_loss_epoch": avg_epoch_loss, "epoch": epoch}, step=global_step)
        logger.info(f"Epoch {epoch} finished. Average Loss: {avg_epoch_loss:.4f}")

        if global_step >= max_train_steps_calculated:
            logger.info("Max training steps reached. Exiting training.")
            break
            
    # --- End of Training ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(unet) 
        model_save_path = os.path.join(TrainConfig.train_config["output_dir"], "final_unet_model.pt")
        accelerator.save(unwrapped_model.state_dict(), model_save_path)
        logger.info(f"Saved final UNet model state_dict to {model_save_path}")
        
        if ema_model:
            ema_save_path = os.path.join(TrainConfig.train_config["output_dir"], "final_ema_model.pt")
            # For saving EMA, create a temporary model, load EMA weights, then save its state_dict
            final_ema_unet = UNet3D( # Re-init with same config as training unet
                input_channels=unet_total_input_channels, # Use the same total input channels as the main unet
                model_channels=TrainConfig.model_config["model_channels"],
                output_channels=TrainConfig.data_config["bit_representation_length"],
                channel_mults=TrainConfig.model_config["channel_mults"],
                num_residual_blocks_per_stage=TrainConfig.model_config["num_residual_blocks_per_stage"],
                time_embedding_dim=TrainConfig.model_config["time_embedding_dim"],
                time_mlp_hidden_dim=TrainConfig.model_config["time_mlp_hidden_dim"],
                time_final_emb_dim=TrainConfig.model_config["time_final_emb_dim"],
                attention_resolutions_indices=TrainConfig.model_config["attention_resolutions_indices"],
                attention_type=TrainConfig.model_config["attention_type"],
                attention_heads=TrainConfig.model_config["attention_heads"],
                dropout=TrainConfig.model_config["dropout"],
                groups=TrainConfig.model_config["groups"],
                initial_conv_kernel_size=TrainConfig.model_config["initial_conv_kernel_size"]
            ).to(accelerator.device) # Ensure it's on the right device before loading state
            ema_model.copy_to(final_ema_unet.parameters()) 
            accelerator.save(final_ema_unet.state_dict(), ema_save_path)
            logger.info(f"Saved final EMA model state_dict to {ema_save_path}")


    if TrainConfig.train_config["log_with_wandb"] and accelerator.is_main_process: 
        if wandb_run_active and wandb.run: # Check if wandb.run is active before finishing
            wandb.finish()
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
