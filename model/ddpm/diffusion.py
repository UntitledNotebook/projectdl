import jax
import jax.numpy as jnp
import functools
from typing import Dict, Any, Tuple, Optional
from flax.training.train_state import TrainState
from flax import jax_utils
from tqdm import tqdm # Import tqdm

def get_ddpm_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Computes DDPM parameters for the diffusion process.

    Args:
        config (Dict[str, Any]): Configuration with keys:
            - beta_schedule (str): 'linear' or 'cosine'.
            - timesteps (int): Number of diffusion steps.
            - p2_loss_weight_gamma (float): Gamma for p2 loss weighting.
            - p2_loss_weight_k (float): K for p2 loss weighting.

    Returns:
        Dict[str, Any]: DDPM parameters including betas, alphas, and loss weights.
    """
    timesteps = config['timesteps']
    if config['beta_schedule'] == 'linear':
        beta_start = 0.0001
        beta_end = 0.02
        betas = jnp.linspace(beta_start, beta_end, timesteps)
    elif config['beta_schedule'] == 'cosine':
        def f(t):
            s = 0.008
            return jnp.cos((t / timesteps + s) / (1 + s) * jnp.pi / 2) ** 2
        steps = jnp.arange(timesteps + 1, dtype=jnp.float32) / timesteps
        alphas_cumprod = f(steps) / f(0)
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = jnp.clip(betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {config['beta_schedule']}")
    
    alphas = 1 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), mode='constant', constant_values=1.0)
    sqrt_recip_alphas = jnp.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - alphas_cumprod)
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    
    p2_loss_weight = (config['p2_loss_weight_k'] + alphas_cumprod / (1 - alphas_cumprod)) ** -config['p2_loss_weight_gamma']
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance,
        'p2_loss_weight': p2_loss_weight
    }

def q_sample(x_start: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray, ddpm_params: Dict[str, Any]) -> jnp.ndarray:
    """Samples from q(x_t | x_0) in the diffusion process.

    Args:
        x_start (jnp.ndarray): Initial data, shape (batch_size, *shape).
        t (jnp.ndarray): Timestep indices, shape (batch_size,).
        noise (jnp.ndarray): Noise, shape (batch_size, *shape).
        ddpm_params (Dict[str, Any]): DDPM parameters from get_ddpm_params.

    Returns:
        jnp.ndarray: Noisy data x_t, shape (batch_size, *shape).
    """
    sqrt_alphas_cumprod = ddpm_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = ddpm_params['sqrt_one_minus_alphas_cumprod']
    return (sqrt_alphas_cumprod[t][:, None, None, None, None] * x_start +
            sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None] * noise)

def noise_to_x0(noise_pred: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray, ddpm_params: Dict[str, Any]) -> jnp.ndarray:
    """Converts predicted noise to x_0 estimate.

    Args:
        noise_pred (jnp.ndarray): Predicted noise, shape (batch_size, *shape).
        x_t (jnp.ndarray): Noisy data at timestep t, shape (batch_size, *shape).
        t (jnp.ndarray): Timestep indices, shape (batch_size,).
        ddpm_params (Dict[str, Any]): DDPM parameters.

    Returns:
        jnp.ndarray: Estimated x_0, shape (batch_size, *shape).
    """
    sqrt_alphas_cumprod = ddpm_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = ddpm_params['sqrt_one_minus_alphas_cumprod']
    return ( (x_t - sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None] * noise_pred) /
         sqrt_alphas_cumprod[t][:, None, None, None, None] )

def x0_to_noise(x0_pred: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray, ddpm_params: Dict[str, Any]) -> jnp.ndarray:
    """Converts predicted x_0 to noise estimate.

    Args:
        x0_pred (jnp.ndarray): Predicted x_0, shape (batch_size, *shape).
        x_t (jnp.ndarray): Noisy data at timestep t, shape (batch_size, *shape).
        t (jnp.ndarray): Timestep indices, shape (batch_size,).
        ddpm_params (Dict[str, Any]): DDPM parameters.

    Returns:
        jnp.ndarray: Estimated noise, shape (batch_size, *shape).
    """
    sqrt_alphas_cumprod = ddpm_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = ddpm_params['sqrt_one_minus_alphas_cumprod']
    return ((x_t - sqrt_alphas_cumprod[t][:, None, None, None] * x0_pred) /
            sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None])

def get_posterior_mean_variance(x_t: jnp.ndarray, t: jnp.ndarray, x0_pred: jnp.ndarray, noise_pred: jnp.ndarray,
                               ddpm_params: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes posterior mean and log variance for p(x_{t-1} | x_t, x_0).

    Args:
        x_t (jnp.ndarray): Noisy data at timestep t, shape (batch_size, *shape).
        t (jnp.ndarray): Timestep indices, shape (batch_size,).
        x0_pred (jnp.ndarray): Predicted x_0, shape (batch_size, *shape).
        noise_pred (jnp.ndarray): Predicted noise, shape (batch_size, *shape).
        ddpm_params (Dict[str, Any]): DDPM parameters.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Posterior mean and log variance, each shape (batch_size, *shape).
    """
    posterior_variance = ddpm_params['posterior_variance']
    alphas = ddpm_params['alphas']
    alphas_cumprod = ddpm_params['alphas_cumprod']
    alphas_cumprod_prev = jnp.pad(alphas_cumprod[:-1], (1, 0), mode='constant', constant_values=1.0)
    posterior_mean_coef1 = (jnp.sqrt(alphas_cumprod_prev[t]) * ddpm_params['betas'][t] / (1 - alphas_cumprod[t]))
    posterior_mean_coef2 = (jnp.sqrt(alphas[t]) * (1 - alphas_cumprod_prev[t]) / (1 - alphas_cumprod[t]))
    posterior_mean = (posterior_mean_coef1[:, None, None, None, None] * x0_pred +
                      posterior_mean_coef2[:, None, None, None, None] * x_t)
    posterior_log_variance = jnp.log(posterior_variance[t] + 1e-8)
    return posterior_mean, posterior_log_variance

def p_loss(rng: jax.random.PRNGKey, state: Any, batch: Dict[str, jnp.ndarray], ddpm_params: Dict[str, Any],
           loss_fn: callable, self_condition: bool, is_pred_x0: bool, pmap_axis: str = 'batch') -> Tuple[Any, Dict[str, float]]:
    """Computes the diffusion loss for a batch.

    Args:
        rng (jax.random.PRNGKey): Random key for sampling timesteps and noise.
        state (Any): Training state with model parameters and apply_fn.
        batch (Dict[str, jnp.ndarray]): Batch with 'voxel' key, shape (batch_size, *voxel_shape, embedding_dim).
        ddpm_params (Dict[str, Any]): DDPM parameters.
        loss_fn (callable): Loss function (e.g., l2_loss).
        self_condition (bool): If True, uses self-conditioning.
        is_pred_x0 (bool): If True, model predicts x_0; else, predicts noise.
        pmap_axis (str): Axis name for pmap (default: 'batch').

    Returns:
        Tuple[Any, Dict[str, float]]: Updated state and metrics (loss, loss_ema, scale if applicable).
    """
    x = batch['voxel']
    assert x.dtype in [jnp.float32, jnp.float64]
    B, D, H, W, C = x.shape
    rng, t_rng = jax.random.split(rng)
    batched_t = jax.random.randint(t_rng, shape=(B,), dtype=jnp.int32, minval=0, maxval=len(ddpm_params['betas']))
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)
    target = x if is_pred_x0 else noise
    x_t = q_sample(x, batched_t, noise, ddpm_params)
    if self_condition:
        rng, condition_rng = jax.random.split(rng)
        zeros = jnp.zeros_like(x_t)
        def estimate_x0(_):
            x0, _ = model_predict(state, x_t, zeros, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=False)
            return x0
        x0 = jax.lax.cond(
            jax.random.uniform(condition_rng, shape=(1,))[0] < 0.5,
            estimate_x0,
            lambda _: zeros,
            None
        )
        x_t = jnp.concatenate([x_t, x0], axis=-1)
    p2_loss_weight = ddpm_params['p2_loss_weight']
    
    def compute_loss(params):
        pred = state.apply_fn({'params': params}, x_t, batched_t)
        loss = loss_fn(pred.reshape(B, -1), target.reshape(B, -1))
        assert loss.shape == (B,)
        loss = loss * p2_loss_weight[batched_t]
        return loss.mean()
    
    dynamic_scale = state.dynamic_scale
    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(compute_loss, axis_name=pmap_axis)
        dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
    else:
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
    loss = jax.lax.pmean(loss, axis_name=pmap_axis)
    loss_ema = jax.lax.pmean(compute_loss(state.params_ema), axis_name=pmap_axis)
    metrics = {'loss': loss, 'loss_ema': loss_ema}
    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        new_state = new_state.replace(
            opt_state=jax.tree_map(functools.partial(jnp.where, is_fin), new_state.opt_state, state.opt_state),
            params=jax.tree_map(functools.partial(jnp.where, is_fin), new_state.params, state.params),
            dynamic_scale=dynamic_scale
        )
        metrics['scale'] = float(dynamic_scale.scale)
    return new_state, metrics

def model_predict(state: Any, x: jnp.ndarray, x0: Optional[jnp.ndarray], t: jnp.ndarray, ddpm_params: Dict[str, Any],
                 self_condition: bool, is_pred_x0: bool, use_ema: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Predicts x_0 or noise using the model.

    Args:
        state (Any): Training state with model parameters and apply_fn.
        x (jnp.ndarray): Noisy data at timestep t, shape (batch_size, *shape).
        x0 (Optional[jnp.ndarray]): Self-conditioning x_0 estimate, shape (batch_size, *shape).
        t (jnp.ndarray): Timestep indices, shape (batch_size,).
        ddpm_params (Dict[str, Any]): DDPM parameters.
        self_condition (bool): If True, uses self-conditioning.
        is_pred_x0 (bool): If True, model predicts x_0; else, predicts noise.
        use_ema (bool): If True, uses EMA parameters (default: True).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Predicted x_0 and noise, each shape (batch_size, *shape).
    """
    variables = {'params': state.params_ema} if use_ema else {'params': state.params}
    if self_condition:
        pred = state.apply_fn(variables, jnp.concatenate([x, x0], axis=-1), t)
    else:
        pred = state.apply_fn(variables, x, t)
    if is_pred_x0:
        x0_pred = pred
        noise_pred = x0_to_noise(pred, x, t, ddpm_params)
    else:
        noise_pred = pred
        x0_pred = noise_to_x0(pred, x, t, ddpm_params)
    return x0_pred, noise_pred

def ddpm_sample_step(state: Any, rng: jax.random.PRNGKey, x: jnp.ndarray, t: jnp.ndarray, x0_last: jnp.ndarray,
                     ddpm_params: Dict[str, Any], self_condition: bool, is_pred_x0: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Performs one step of DDPM sampling.

    Args:
        state (Any): Training state with model parameters and apply_fn.
        rng (jax.random.PRNGKey): Random key for sampling noise.
        x (jnp.ndarray): Current sample, shape (batch_size, *shape).
        t (jnp.ndarray): Current timestep, scalar on each device.
        x0_last (jnp.ndarray): Previous x_0 estimate, shape (batch_size, *shape).
        ddpm_params (Dict[str, Any]): DDPM parameters.
        self_condition (bool): If True, uses self-conditioning.
        is_pred_x0 (bool): If True, model predicts x_0; else, predicts noise.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Updated sample and x_0 estimate, each shape (batch_size, *shape).
    """
    batched_t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t # t is scalar here, x.shape[0] is batch_size_per_device
    if self_condition:
        x0, v = model_predict(state, x, x0_last, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=True)
    else:
        x0, v = model_predict(state, x, None, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=True)
    # Pass batched_t to get_posterior_mean_variance instead of scalar t
    posterior_mean, posterior_log_variance = get_posterior_mean_variance(x, batched_t, x0, v, ddpm_params)
    # Expand posterior_log_variance to match noise shape
    noise_scale = jnp.exp(0.5 * posterior_log_variance[:, None, None, None, None])
    x = posterior_mean + noise_scale * jax.random.normal(rng, x.shape)
    return x, x0

def ddpm_inpaint_sample_step(state: Any, rng: jax.random.PRNGKey, x: jnp.ndarray, t: jnp.ndarray, x0_last: jnp.ndarray,
                            ddpm_params: Dict[str, Any], self_condition: bool, is_pred_x0: bool, x_true: jnp.ndarray, mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # This function can only be run on a single device, and both x and x_true should have batch_size=1
    assert x.shape[0] == 1 and x.shape == x_true.shape == mask
    batched_t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t # t is scalar here, x.shape[0] is batch_size_per_device
    if self_condition:
        x0, v = model_predict(state, x, x0_last, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=True)
    else:
        x0, v = model_predict(state, x, None, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=True)
    # Pass batched_t to get_posterior_mean_variance instead of scalar t
    posterior_mean, posterior_log_variance = get_posterior_mean_variance(x, batched_t, x0, v, ddpm_params)
    model_based_x_prev = posterior_mean + jnp.exp(0.5 * posterior_log_variance[:, None, None, None, None]) * jax.random.normal(rng, x.shape)
    if t > 0:
        t_minus_1_batched = jnp.full((x.shape[0],), t - 1, dtype=jnp.int32)
        noise_for_q = jax.random.normal(rng, x.shape)

        known_region_x_prev = q_sample(
            x_true, t_minus_1_batched, noise_for_q, ddpm_params
        )

        x_prev_corrected = model_based_x_prev * mask + known_region_x_prev * (1 - mask)
    else: # t == 0
        x_prev_corrected = x0 * mask + x_true * (1 - mask)
    return x_prev_corrected, x0


def sample_loop(rng: jax.random.PRNGKey, state: TrainState, shape: Tuple[int, ...], p_sample_step: callable, timesteps: int) -> jnp.ndarray:
    """Generates samples using the DDPM sampling loop.

    Args:
        rng (jax.random.PRNGKey): Random key for sampling.
        state (TrainState): Training state.
        shape (Tuple[int, ...]): Shape of samples (num_devices, batch_size_per_device, *voxel_shape, embedding_dim).
        p_sample_step (callable): Pmapped DDPM sampling step function.
        timesteps (int): Number of diffusion timesteps.

    Returns:
        jnp.ndarray: Generated embeddings, shape (*shape).
    """
    rng, x_rng = jax.random.split(rng)
    x = jax.random.normal(x_rng, shape)
    x0 = jnp.zeros_like(x)
    # Wrap the loop with tqdm for progress visualization
    for t in tqdm(reversed(jnp.arange(timesteps)), desc="Sampling", total=timesteps):
        rng, *step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)
        x, x0 = p_sample_step(state, step_rng, x, jax_utils.replicate(t), x0)
    return x  # Return raw embeddings