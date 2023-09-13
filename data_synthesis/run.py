import argparse
from self_instruct import generate_instructions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate instructions based on a config file.')
    parser.add_argument('--config-path', type=str, default="config.yaml", help='Path to the config file')
    
    args = parser.parse_args()
    generate_instructions(args)