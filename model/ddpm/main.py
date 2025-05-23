import logging
from .train import train

def main():
    """Main function to start training."""
    train()

if __name__ == '__main__':
    # Set the logging level to INFO
    logging.basicConfig(level=logging.INFO, force=True)
    main()