cd md2txt
maturin build --release
python3 -m pip install target/wheels/*.whl --force-reinstall