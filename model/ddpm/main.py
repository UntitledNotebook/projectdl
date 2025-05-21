import logging
from train import train

def main():
    """Main function to start training."""
    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())
    train()

if __name__ == '__main__':
    import jax
    main()