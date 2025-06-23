# Code Ramblings

Random collections of scripts written to test concepts or to explore limits of how to implement ML related systems.

- [Code Ramblings](#code-ramblings)
  - [Install](#install)
    - [Install as dependency](#install-as-dependency)
      - [With uv:](#with-uv)
      - [With pip:](#with-pip)
    - [Installing as project](#installing-as-project)
      - [Assuming you have uv installed](#assuming-you-have-uv-installed)
      - [Running the tests](#running-the-tests)
  - [Webpage](#webpage)
  - [Notebooks](#notebooks)


## Install

We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) as project manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install as dependency

#### With uv:

```bash
uv add "code-ramblings @ git+https://github.com/BreytMN/code-ramblings"
```

#### With pip:

```bash
pip install git+https://github.com/BreytMN/code-ramblings.git
```

### Installing as project

#### Assuming you have uv installed

```bash
git clone git@github.com:BreytMN/code-ramblings.git
cd code-ramblings/
uv self update
```

No optional dependencies

```bash
uv sync
```

Optional torch cpu. Project's optional dependency will default to the cpu version.

```bash
uv sync --extra torch
```

Optional torch cuda. Exclude the configured sources to install the cuda version.

```bash
uv sync --extra torch --no-sources
```

#### Running the tests

The pyproject.toml is already configured to run all tests correctly:

```bash
pytest
```

## Webpage

Everything tested and analyzed here will eventually be displayed on [code-ramblings](https://code-ramblings.breytmn.com/)

## Notebooks

* [Mixture Models](./notebooks/MixtureModels.ipynb) ([web version](https://code-ramblings.breytmn.com/))
* ... coming soon
