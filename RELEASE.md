# Releasing `cftsad` for `pip install`

This project is already structured as a Python package (`src/` layout + `pyproject.toml`).
Use the steps below to publish it so users can install with `pip`.

## 1) Bump version

Update version in:

- `pyproject.toml` -> `project.version`
- `src/cftsad/__init__.py` -> `__version__`

Recommended scheme: semantic versioning (`MAJOR.MINOR.PATCH`), for example `0.1.1`.

## 2) Build distributions

```bash
python3 -m pip install --upgrade pip
pip install -e ".[dev]"
pip wheel . -w dist --no-deps
```

This generates a wheel:

- `dist/*.whl` (wheel)

## 3) Validate package metadata

```bash
twine check dist/*
```

## 4) Upload to TestPyPI (recommended first)

Create an API token at TestPyPI and export it:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-testpypi-token>
```

Upload:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Install test:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple cftsad
```

## 5) Upload to PyPI

Create an API token at PyPI and export it:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-pypi-token>
```

Upload:

```bash
twine upload dist/*
```

## 6) Verify `pip` install

```bash
pip install cftsad
python -c "import cftsad; print(cftsad.__version__)"
```

## 7) Optional: install from GitHub before first release

If not yet on PyPI:

```bash
pip install "git+https://github.com/<org-or-user>/cftsad.git"
```
