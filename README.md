# Install

Create a virtual environment and install the requirements.
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Install Rust and `md2txt`
- Install Rust
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
- Install `md2txt`
```sh
cd md2txt
maturin develop --release
```
(If you get an error, especially on Windows, try `maturin develop --release -i python`).

# Generate Tokenizer
```
python generate_tk.py
```

# Train
## Logistic Regression
On CUDA:
```
python ./train.py -m logistic -d cuda -n 50
```

On CPU:
```
python ./train.py -m logistic -d cpu -n 50
```