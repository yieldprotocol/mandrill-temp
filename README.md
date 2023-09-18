## Setup

* Initialize the submodules
```bash
git submodule init
git submodule update
```

* Install the requirements
```bash
conda create -n venv
conda activate venv
pip install -r requirements.txt
```

* Finetune with QLoRA quantization
```bash
python llama2_qlora.py
```