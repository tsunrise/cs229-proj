# CS229 Final Project: Software Classification using NLP

## Setup

### Install Dependencies
Create a virtual environment and install the requirements.
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Install Rust
On Linux and macOS, run the following command in your terminal. On Windows, download and run the executable from [here](https://www.rust-lang.org/tools/install).
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
### Compile Markdown Normalizer
The markdown normalizer is written in Rust, so you need to compile it before using it. On a virtual environment, run the following commands:
```sh
cd md2txt
maturin develop --release
```
(If you get an error, especially on Windows, try `maturin develop --release -i python`).

If you do not want to use a virtual environment, you can install the `md2txt` package using globally by running:
```sh
chmod +x install_rust_backend.sh
./install_rust_backend.sh
```
## Generate Tokenizer
A pre-trained tokenizer is already provided in this repository. If you want to generate a new tokenizer, run the following command:
```
python generate_tk.py
```

## Train
There are three possible models to train: `logistic`, `nn`, and `bert`. To train a model, run the following command:
```
python train.py -m <model> -d <device> -n <num_epochs>
```
where `<model>` is one of `logistic`, `nn`, or `bert`, `<device>` is the device to use (either `cpu` or `cuda`), and `<num_epochs>` is the number of epochs to train the model for. For example, to train a logistic regression model on the GPU for 200 epochs, run:
```
python train.py -m logistic -d cuda -n 200
```
There are additional flags provided:
- `-fc`: If set, the trainer will clean the cached preprocessed data.
- `-fd`: If set, the trainer will re-download the dataset.

The trained model will be saved in the `.cs229_cache/snapshots` directory. Training curve will be saved in `runs` directory and can be viewed using TensorBoard.

## Evaluate
To evaluate a model, run the following command:
```sh
python evaluate.py -m <model> -d <device> -c <checkpoint>
```
where `<model>` is one of `logistic`, `nn`, or `bert`, `<device>` is the device to use (either `cpu` or `cuda`), and `<checkpoint>` is the path to the saved model to evaluate. For example, to evaluate a logistic regression model on the GPU using the checkpoint `logistic_200.pt`, run:
```sh
python evaluate.py -m logistic -d cuda -c .cs229_cache/snapshots/logistic_200.pt
```

Evaluation result will be saved in `runs` directory, and can be viewed using TensorBoard `Text` tab.