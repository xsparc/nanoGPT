# Getting Started
## Install

    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121    
    
    pip install transformers datasets tiktoken wandb tqdm

## Command

    python circuit_gen_prepare.py

    python circuit_gen_train.py

    python circuit_gen_train.py config/train_spice.py

    python sample.py --out_dir=out-spice
