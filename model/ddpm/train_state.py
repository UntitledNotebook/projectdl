from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib
from typing import Any, Optional

class TrainState(train_state.TrainState):
    """Custom training state for diffusion model.

    Extends Flax's TrainState to include EMA parameters and dynamic scaling for
    mixed-precision training.

    Attributes:
        params (Any): Model parameters.
        params_ema (Any): Exponential moving average of model parameters.
        opt_state (Any): Optimizer state.
        apply_fn (callable): Function to apply the model.
        tx (optax.GradientTransformation): Optimizer.
        dynamic_scale (Optional[dynamic_scale_lib.DynamicScale]): Dynamic scaling
            for mixed-precision training (default: None).
    """
    params_ema: Any = None
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None