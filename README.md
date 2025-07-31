# AlgoVault

## Environment setup using conda

Please create a conda environment and install the required packages.

```bash
# Recreate environment from file
conda env create -f environment.yml

# Activate environment
conda activate algovault

# Verify installation (optional)
conda list
```

## Environment setup using pip
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you use InsCode, you don't have to create a virtual environment.
```bash
$ git clone https://gitcode.com/xxx/algovault.git
$ pip install -r requirements.txt
$ export TUSHARE_API_KEY='your_api_key'
$ jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --config=./jupyter_notebook_config.py
```

## Tushare configuration
```bash
export TUSHARE_API_KEY='your_api_key'
```