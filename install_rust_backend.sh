cd md2txt
maturin build --release
python3 -m pip install --force-reinstall target/wheels/*.whl