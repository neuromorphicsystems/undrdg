UNDRDG provides building blocks to convert existing datasets to the UNDR format.

The scripts used to convert the datasets that we re-distribute can be found in _scripts_. The converted datasets can be browsed at https://www.undr.space.

# Use UNDRDG in your own project

```sh
python3 -m pip install undrdg
```

See _scripts_ for usage examples.

# Add scripts to this directory

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

# Contribute to UNDRDG

1. Install the development version

```sh
python3 -m venv .venv
source .venv/bin/activate
pip remove undrdg # to avoid name clashes
pip install -e .
```

2. Edit files in _undrg_. Changes are automatically reflected in scripts that import the library.

3. Format and lint.

```sh
isort .; black .; pyright .
```
