# Install

Create a virtual environment and install the requirements.
```sh
python3 -m venv env
pip install -r requirements.txt
```

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