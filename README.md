[//]: # (Logo:)

<div align="center">

PySR searches for symbolic expressions which optimize a particular objective.

https://github.com/astroautomata/PySR/assets/7593028/c8511a49-b408-488f-8f18-b1749078268f


# PySR: High-Performance Symbolic Regression in Python and Julia

| **Docs** | **Forums** | **Paper** | **colab demo** |
|:---:|:---:|:---:|:---:|
|[![Documentation](https://github.com/astroautomata/PySR/actions/workflows/docs.yml/badge.svg)](https://ai.damtp.cam.ac.uk/pysr/)|[![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/astroautomata/PySR/discussions)|[![Paper](https://img.shields.io/badge/arXiv-2305.01582-b31b1b)](https://arxiv.org/abs/2305.01582)|[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/github/astroautomata/PySR/blob/master/examples/pysr_demo.ipynb)|

| **pip** | **conda** | **Stats** |
| :---: | :---: | :---: |
|[![PyPI version](https://badge.fury.io/py/pysr.svg)](https://badge.fury.io/py/pysr)|[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pysr.svg)](https://anaconda.org/conda-forge/pysr)|<div align="center">pip: [![Downloads](https://static.pepy.tech/badge/pysr)](https://pypi.org/project/pysr/)<br>conda: [![Anaconda-Server Badge](https://anaconda.org/conda-forge/pysr/badges/downloads.svg)](https://anaconda.org/conda-forge/pysr)</div>|

</div>

If you find PySR useful, please cite the paper [arXiv:2305.01582](https://arxiv.org/abs/2305.01582).
If you've finished a project with PySR, please submit a PR to showcase your work on the [research showcase page](https://ai.damtp.cam.ac.uk/pysr/papers)!

**Contents**:

- [Why PySR?](#why-pysr)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [→ Documentation](https://ai.damtp.cam.ac.uk/pysr)
- [Contributors](#contributors-)

<div align="center">

### Test status

| **Linux** | **Windows** | **macOS** |
|---|---|---|
|[![Linux](https://github.com/astroautomata/PySR/actions/workflows/CI.yml/badge.svg)](https://github.com/astroautomata/PySR/actions/workflows/CI.yml)|[![Windows](https://github.com/astroautomata/PySR/actions/workflows/CI_Windows.yml/badge.svg)](https://github.com/astroautomata/PySR/actions/workflows/CI_Windows.yml)|[![macOS](https://github.com/astroautomata/PySR/actions/workflows/CI_mac.yml/badge.svg)](https://github.com/astroautomata/PySR/actions/workflows/CI_mac.yml)|
| **Docker** | **Conda** | **Coverage** |
|[![Docker](https://github.com/astroautomata/PySR/actions/workflows/CI_docker.yml/badge.svg)](https://github.com/astroautomata/PySR/actions/workflows/CI_docker.yml)|[![conda-forge](https://github.com/astroautomata/PySR/actions/workflows/CI_conda_forge.yml/badge.svg)](https://github.com/astroautomata/PySR/actions/workflows/CI_conda_forge.yml)|[![codecov](https://codecov.io/gh/astroautomata/PySR/branch/master/graph/badge.svg)](https://codecov.io/gh/astroautomata/PySR)|

</div>

## Why PySR?

PySR is an open-source tool for *Symbolic Regression*: a machine learning
task where the goal is to find an interpretable symbolic expression that optimizes some objective.

Over a period of several years, PySR has been engineered from the ground up
to be (1) as high-performance as possible,
(2) as configurable as possible, and (3) easy to use.
PySR is developed alongside the Julia library [SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl),
which forms the powerful search engine of PySR.
The details of these algorithms are described in the [PySR paper](https://arxiv.org/abs/2305.01582).

Symbolic regression works best on low-dimensional datasets, but
one can also extend these approaches to higher-dimensional
spaces by using "*Symbolic Distillation*" of Neural Networks, as explained in
[2006.11287](https://arxiv.org/abs/2006.11287), where we apply
it to N-body problems. Here, one essentially uses
symbolic regression to convert a neural net
to an analytic equation. Thus, these tools simultaneously present
an explicit and powerful way to interpret deep neural networks.

## Installation

### Pip

You can install PySR with pip:

```bash
pip install pysr
```

Julia dependencies will be installed at first import.

### Conda

Similarly, with conda:

```bash
conda install -c conda-forge pysr
```

<details>
<summary>

### Docker

</summary>

You can also use the `Dockerfile` to install PySR in a docker container

1. Clone this repo.
2. Within the repo's directory, build the docker container:
```bash
docker build -t pysr .
```
3. You can then start the container with an IPython execution with:
```bash
docker run -it --rm pysr ipython
```

For more details, see the [docker section](#docker).

</details>

<details>
<summary>

### Apptainer

</summary>

If you are using PySR on a cluster where you do not have root access,
you can use [Apptainer](https://apptainer.org/) to build a container
instead of Docker. The `Apptainer.def` file is analogous to the `Dockerfile`,
and can be built with:

```bash
apptainer build --notest pysr.sif Apptainer.def
```

and launched with

```bash
apptainer run pysr.sif
```

</details>

<details>
<summary>

### Troubleshooting

</summary>

One issue you might run into can result in a hard crash at import with
a message like "`GLIBCXX_...` not found". This is due to another one of the Python dependencies
loading an incorrect `libstdc++` library. To fix this, you should modify your
`LD_LIBRARY_PATH` variable to reference the Julia libraries. For example, if the Julia
version of `libstdc++.so` is located in `$HOME/.julia/juliaup/julia-1.10.0+0.x64.linux.gnu/lib/julia/`
(which likely differs on your system!), you could add:

```
export LD_LIBRARY_PATH=$HOME/.julia/juliaup/julia-1.10.0+0.x64.linux.gnu/lib/julia/:$LD_LIBRARY_PATH
```

to your `.bashrc` or `.zshrc` file.

</details>


## Quickstart

You might wish to try the interactive tutorial [here](https://colab.research.google.com/github/astroautomata/PySR/blob/master/examples/pysr_demo.ipynb), which uses the notebook in `examples/pysr_demo.ipynb`.

In practice, I highly recommend using IPython rather than Jupyter, as the printing is much nicer.
Below is a quick demo here which you can paste into a Python runtime.
First, let's import numpy to generate some test data:

```python
import numpy as np

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
```

We have created a dataset with 100 datapoints, with 5 features each.
The relation we wish to model is $2.5382 \cos(x_3) + x_0^2 - 0.5$.

Now, let's create a PySR model and train it.
PySR's main interface is in the style of scikit-learn:

```python
from pysr import PySRRegressor

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)
```

This will set up the model for 40 iterations of the search code, which contains hundreds of thousands of mutations and equation evaluations.

Let's train this model on our dataset:

```python
model.fit(X, y)
```

Internally, this launches a Julia process which will do a multithreaded search for equations to fit the dataset.

Equations will be printed during training, and once you are satisfied, you may
quit early by hitting 'q' and then \<enter\>.

After the model has been fit, you can run `model.predict(X)`
to see the predictions on a given dataset using the automatically-selected expression,
or, for example, `model.predict(X, 3)` to see the predictions of the 3rd equation.

You may run:

```python
print(model)
```

to print the learned equations:

```python
PySRRegressor.equations_ = [
	   pick     score                                           equation       loss  complexity
	0        0.000000                                          4.4324794  42.354317           1
	1        1.255691                                          (x0 * x0)   3.437307           3
	2        0.011629                          ((x0 * x0) + -0.28087974)   3.358285           5
	3        0.897855                              ((x0 * x0) + cos(x3))   1.368308           6
	4        0.857018                ((x0 * x0) + (cos(x3) * 2.4566472))   0.246483           8
	5  >>>>       inf  (((cos(x3) + -0.19699033) * 2.5382123) + (x0 *...   0.000000          10
]
```

This arrow in the `pick` column indicates which equation is currently selected by your
`model_selection` strategy for prediction.
(You may change `model_selection` after `.fit(X, y)` as well.)

`model.equations_` is a pandas DataFrame containing all equations, including callable format
(`lambda_format`),
SymPy format (`sympy_format` - which you can also get with `model.sympy()`), and even JAX and PyTorch format
(both of which are differentiable - which you can get with `model.jax()` and `model.pytorch()`).

Note that `PySRRegressor` stores the state of the last search, and will restart from where you left off the next time you call `.fit()`, assuming you have set `warm_start=True`.
This will cause problems if significant changes are made to the search parameters (like changing the operators). You can run `model.reset()` to reset the state.

You will notice that PySR will save two files:
`hall_of_fame...csv` and `hall_of_fame...pkl`.
The csv file is a list of equations and their losses, and the pkl file is a saved state of the model.
You may load the model from the `pkl` file with:

```python
model = PySRRegressor.from_file("hall_of_fame.2022-08-10_100832.281.pkl")
```

There are several other useful features such as denoising (e.g., `denoise=True`),
feature selection (e.g., `select_k_features=3`).
For examples of these and other features, see the [examples page](https://ai.damtp.cam.ac.uk/pysr/examples).
For a detailed look at more options, see the [options page](https://ai.damtp.cam.ac.uk/pysr/options).
You can also see the full API at [this page](https://ai.damtp.cam.ac.uk/pysr/api).
There are also tips for tuning PySR on [this page](https://ai.damtp.cam.ac.uk/pysr/tuning).

### Detailed Example

The following code makes use of as many PySR features as possible.
Note that is just a demonstration of features and you should not use this example as-is.
For details on what each parameter does, check out the [API page](https://ai.damtp.cam.ac.uk/pysr/api/).

```python
model = PySRRegressor(
    populations=8,
    # ^ Assuming we have 4 cores, this means 2 populations per core, so one is always running.
    population_size=50,
    # ^ Slightly larger populations, for greater diversity.
    ncycles_per_iteration=500,
    # ^ Generations between migrations.
    niterations=10000000,  # Run forever
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 24,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=50,
    # ^ Allow greater complexity.
    maxdepth=10,
    # ^ But, avoid deep nesting.
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["square", "cube", "exp", "cos2(x)=cos(x)^2"],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    # ^ Limit the complexity within each argument.
    # "inv": (-1, 9) states that the numerator has no constraint,
    # but the denominator has a max complexity of 9.
    # "exp": 9 simply states that `exp` can only have
    # an expression of complexity 9 as input.
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    },
    # ^ Nesting constraints on operators. For example,
    # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
    complexity_of_operators={"/": 2, "exp": 3},
    # ^ Custom complexity of particular operators.
    complexity_of_constants=2,
    # ^ Punish constants more than variables
    select_k_features=4,
    # ^ Train on only the 4 most important features
    progress=True,
    # ^ Can set to false if printing to a file.
    weight_randomize=0.1,
    # ^ Randomize the tree much more frequently
    cluster_manager=None,
    # ^ Can be set to, e.g., "slurm", to run a slurm
    # cluster. Just launch one script from the head node.
    precision=64,
    # ^ Higher precision calculations.
    warm_start=True,
    # ^ Start from where left off.
    turbo=True,
    # ^ Faster evaluation (experimental)
    extra_sympy_mappings={"cos2": lambda x: sympy.cos(x)**2},
    # extra_torch_mappings={sympy.cos: torch.cos},
    # ^ Not needed as cos already defined, but this
    # is how you define custom torch operators.
    # extra_jax_mappings={sympy.cos: "jnp.cos"},
    # ^ For JAX, one passes a string.
)
```

### Docker

You can also test out PySR in Docker, without
installing it locally, by running the following command in
the root directory of this repo:

```bash
docker build -t pysr .
```

This builds an image called `pysr` for your system's architecture,
which also contains IPython. You can select a specific version
of Python and Julia with:

```bash
docker build -t pysr --build-arg JLVERSION=1.10.0 --build-arg PYVERSION=3.11.6 .
```

You can then run with this dockerfile using:

```bash
docker run -it --rm -v "$PWD:/data" pysr ipython
```

which will link the current directory to the container's `/data` directory
and then launch ipython.

If you have issues building for your system's architecture,
you can emulate another architecture by including `--platform linux/amd64`,
before the `build` and `run` commands.

<div align="center">

### Contributors ✨

</div>

We are eager to welcome new contributors! Check out our contributors [guide](https://github.com/astroautomata/PySR/blob/master/CONTRIBUTORS.md) for tips 🚀.
If you have an idea for a new feature, don't hesitate to share it on the [issues](https://github.com/astroautomata/PySR/issues) or [discussions](https://github.com/astroautomata/PySR/discussions) page.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://www.linkedin.com/in/markkittisopikul/"><img src="https://avatars.githubusercontent.com/u/8062771?v=4?s=50" width="50px;" alt="Mark Kittisopikul"/><br /><sub><b>Mark Kittisopikul</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=mkitti" title="Code">💻</a> <a href="#ideas-mkitti" title="Ideas, planning, and feedback.">💡</a> <a href="#infra-mkitti" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#platform-mkitti" title="Packaging/porting to new platform">📦</a> <a href="#promotion-mkitti" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Amkitti" title="Reviewed Pull Requests">👀</a> <a href="#tool-mkitti" title="Tools">🔧</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=mkitti" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/tttc3"><img src="https://avatars.githubusercontent.com/u/97948946?v=4?s=50" width="50px;" alt="T Coxon"/><br /><sub><b>T Coxon</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3Atttc3" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=tttc3" title="Code">💻</a> <a href="#plugin-tttc3" title="Plugin/utility libraries">🔌</a> <a href="#ideas-tttc3" title="Ideas, planning, and feedback.">💡</a> <a href="#infra-tttc3" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-tttc3" title="Maintenance">🚧</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Atttc3" title="Reviewed Pull Requests">👀</a> <a href="#tool-tttc3" title="Tools">🔧</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=tttc3" title="Tests">⚠️</a> <a href="#userTesting-tttc3" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/DhananjayAshok"><img src="https://avatars.githubusercontent.com/u/46792537?v=4?s=50" width="50px;" alt="Dhananjay Ashok"/><br /><sub><b>Dhananjay Ashok</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=DhananjayAshok" title="Code">💻</a> <a href="#example-DhananjayAshok" title="Examples.">🌍</a> <a href="#ideas-DhananjayAshok" title="Ideas, planning, and feedback.">💡</a> <a href="#maintenance-DhananjayAshok" title="Maintenance">🚧</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=DhananjayAshok" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://gitlab.com/johanbluecreek"><img src="https://avatars.githubusercontent.com/u/852554?v=4?s=50" width="50px;" alt="Johan Blåbäck"/><br /><sub><b>Johan Blåbäck</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3Ajohanbluecreek" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=johanbluecreek" title="Code">💻</a> <a href="#ideas-johanbluecreek" title="Ideas, planning, and feedback.">💡</a> <a href="#maintenance-johanbluecreek" title="Maintenance">🚧</a> <a href="#promotion-johanbluecreek" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Ajohanbluecreek" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=johanbluecreek" title="Tests">⚠️</a> <a href="#userTesting-johanbluecreek" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://mathopt.de/people/martensen/index.php"><img src="https://avatars.githubusercontent.com/u/20998300?v=4?s=50" width="50px;" alt="JuliusMartensen"/><br /><sub><b>JuliusMartensen</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3AAlCap23" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=AlCap23" title="Code">💻</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=AlCap23" title="Documentation">📖</a> <a href="#plugin-AlCap23" title="Plugin/utility libraries">🔌</a> <a href="#ideas-AlCap23" title="Ideas, planning, and feedback.">💡</a> <a href="#infra-AlCap23" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-AlCap23" title="Maintenance">🚧</a> <a href="#platform-AlCap23" title="Packaging/porting to new platform">📦</a> <a href="#promotion-AlCap23" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3AAlCap23" title="Reviewed Pull Requests">👀</a> <a href="#tool-AlCap23" title="Tools">🔧</a> <a href="#userTesting-AlCap23" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ngam"><img src="https://avatars.githubusercontent.com/u/67342040?v=4?s=50" width="50px;" alt="ngam"/><br /><sub><b>ngam</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=ngam" title="Code">💻</a> <a href="#infra-ngam" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#platform-ngam" title="Packaging/porting to new platform">📦</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Angam" title="Reviewed Pull Requests">👀</a> <a href="#tool-ngam" title="Tools">🔧</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=ngam" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://cjdoris.github.io/"><img src="https://avatars.githubusercontent.com/u/1844215?v=4?s=50" width="50px;" alt="Christopher Rowley"/><br /><sub><b>Christopher Rowley</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=cjdoris" title="Code">💻</a> <a href="#ideas-cjdoris" title="Ideas, planning, and feedback.">💡</a> <a href="#infra-cjdoris" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#platform-cjdoris" title="Packaging/porting to new platform">📦</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Acjdoris" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/kazewong"><img src="https://avatars.githubusercontent.com/u/8803931?v=4?s=50" width="50px;" alt="Kaze Wong"/><br /><sub><b>Kaze Wong</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3Akazewong" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=kazewong" title="Code">💻</a> <a href="#ideas-kazewong" title="Ideas, planning, and feedback.">💡</a> <a href="#infra-kazewong" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-kazewong" title="Maintenance">🚧</a> <a href="#promotion-kazewong" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Akazewong" title="Reviewed Pull Requests">👀</a> <a href="#research-kazewong" title="Research">🔬</a> <a href="#userTesting-kazewong" title="User Testing">📓</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ChrisRackauckas"><img src="https://avatars.githubusercontent.com/u/1814174?v=4?s=50" width="50px;" alt="Christopher Rackauckas"/><br /><sub><b>Christopher Rackauckas</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3AChrisRackauckas" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=ChrisRackauckas" title="Code">💻</a> <a href="#plugin-ChrisRackauckas" title="Plugin/utility libraries">🔌</a> <a href="#ideas-ChrisRackauckas" title="Ideas, planning, and feedback.">💡</a> <a href="#infra-ChrisRackauckas" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#promotion-ChrisRackauckas" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3AChrisRackauckas" title="Reviewed Pull Requests">👀</a> <a href="#research-ChrisRackauckas" title="Research">🔬</a> <a href="#tool-ChrisRackauckas" title="Tools">🔧</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=ChrisRackauckas" title="Tests">⚠️</a> <a href="#userTesting-ChrisRackauckas" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://kidger.site/"><img src="https://avatars.githubusercontent.com/u/33688385?v=4?s=50" width="50px;" alt="Patrick Kidger"/><br /><sub><b>Patrick Kidger</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3Apatrick-kidger" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=patrick-kidger" title="Code">💻</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=patrick-kidger" title="Documentation">📖</a> <a href="#plugin-patrick-kidger" title="Plugin/utility libraries">🔌</a> <a href="#ideas-patrick-kidger" title="Ideas, planning, and feedback.">💡</a> <a href="#maintenance-patrick-kidger" title="Maintenance">🚧</a> <a href="#promotion-patrick-kidger" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Apatrick-kidger" title="Reviewed Pull Requests">👀</a> <a href="#research-patrick-kidger" title="Research">🔬</a> <a href="#tool-patrick-kidger" title="Tools">🔧</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=patrick-kidger" title="Tests">⚠️</a> <a href="#userTesting-patrick-kidger" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/OkonSamuel"><img src="https://avatars.githubusercontent.com/u/39421418?v=4?s=50" width="50px;" alt="Okon Samuel"/><br /><sub><b>Okon Samuel</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3AOkonSamuel" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=OkonSamuel" title="Code">💻</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=OkonSamuel" title="Documentation">📖</a> <a href="#maintenance-OkonSamuel" title="Maintenance">🚧</a> <a href="#ideas-OkonSamuel" title="Ideas, planning, and feedback.">💡</a> <a href="#infra-OkonSamuel" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3AOkonSamuel" title="Reviewed Pull Requests">👀</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=OkonSamuel" title="Tests">⚠️</a> <a href="#userTesting-OkonSamuel" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/w2ll2am"><img src="https://avatars.githubusercontent.com/u/16038228?v=4?s=50" width="50px;" alt="William Booth-Clibborn"/><br /><sub><b>William Booth-Clibborn</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=w2ll2am" title="Code">💻</a> <a href="#example-w2ll2am" title="Examples.">🌍</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=w2ll2am" title="Documentation">📖</a> <a href="#userTesting-w2ll2am" title="User Testing">📓</a> <a href="#maintenance-w2ll2am" title="Maintenance">🚧</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Aw2ll2am" title="Reviewed Pull Requests">👀</a> <a href="#tool-w2ll2am" title="Tools">🔧</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=w2ll2am" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://pablo-lemos.github.io/"><img src="https://avatars.githubusercontent.com/u/38078898?v=4?s=50" width="50px;" alt="Pablo Lemos"/><br /><sub><b>Pablo Lemos</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3APablo-Lemos" title="Bug reports">🐛</a> <a href="#ideas-Pablo-Lemos" title="Ideas, planning, and feedback.">💡</a> <a href="#promotion-Pablo-Lemos" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3APablo-Lemos" title="Reviewed Pull Requests">👀</a> <a href="#research-Pablo-Lemos" title="Research">🔬</a> <a href="#userTesting-Pablo-Lemos" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Moelf"><img src="https://avatars.githubusercontent.com/u/5306213?v=4?s=50" width="50px;" alt="Jerry Ling"/><br /><sub><b>Jerry Ling</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3AMoelf" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=Moelf" title="Code">💻</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=Moelf" title="Documentation">📖</a> <a href="#example-Moelf" title="Examples.">🌍</a> <a href="#ideas-Moelf" title="Ideas, planning, and feedback.">💡</a> <a href="#promotion-Moelf" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3AMoelf" title="Reviewed Pull Requests">👀</a> <a href="#userTesting-Moelf" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/CharFox1"><img src="https://avatars.githubusercontent.com/u/35052672?v=4?s=50" width="50px;" alt="Charles Fox"/><br /><sub><b>Charles Fox</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3ACharFox1" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=CharFox1" title="Code">💻</a> <a href="#ideas-CharFox1" title="Ideas, planning, and feedback.">💡</a> <a href="#maintenance-CharFox1" title="Maintenance">🚧</a> <a href="#promotion-CharFox1" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3ACharFox1" title="Reviewed Pull Requests">👀</a> <a href="#research-CharFox1" title="Research">🔬</a> <a href="#userTesting-CharFox1" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/johannbrehmer"><img src="https://avatars.githubusercontent.com/u/17068560?v=4?s=50" width="50px;" alt="Johann Brehmer"/><br /><sub><b>Johann Brehmer</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=johannbrehmer" title="Code">💻</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=johannbrehmer" title="Documentation">📖</a> <a href="#ideas-johannbrehmer" title="Ideas, planning, and feedback.">💡</a> <a href="#promotion-johannbrehmer" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Ajohannbrehmer" title="Reviewed Pull Requests">👀</a> <a href="#research-johannbrehmer" title="Research">🔬</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=johannbrehmer" title="Tests">⚠️</a> <a href="#userTesting-johannbrehmer" title="User Testing">📓</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="http://www.cosmicmar.com/"><img src="https://avatars.githubusercontent.com/u/1510968?v=4?s=50" width="50px;" alt="Marius Millea"/><br /><sub><b>Marius Millea</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=marius311" title="Code">💻</a> <a href="#ideas-marius311" title="Ideas, planning, and feedback.">💡</a> <a href="#promotion-marius311" title="Promotion">📣</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Amarius311" title="Reviewed Pull Requests">👀</a> <a href="#userTesting-marius311" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://gitlab.com/cobac"><img src="https://avatars.githubusercontent.com/u/27872944?v=4?s=50" width="50px;" alt="Coba"/><br /><sub><b>Coba</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3Acobac" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=cobac" title="Code">💻</a> <a href="#ideas-cobac" title="Ideas, planning, and feedback.">💡</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Acobac" title="Reviewed Pull Requests">👀</a> <a href="#userTesting-cobac" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/foxtran"><img src="https://avatars.githubusercontent.com/u/39676482?v=4?s=50" width="50px;" alt="foxtran"/><br /><sub><b>foxtran</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=foxtran" title="Code">💻</a> <a href="#ideas-foxtran" title="Ideas, planning, and feedback.">💡</a> <a href="#maintenance-foxtran" title="Maintenance">🚧</a> <a href="#tool-foxtran" title="Tools">🔧</a> <a href="#userTesting-foxtran" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://smhasan.com/"><img src="https://avatars.githubusercontent.com/u/36223598?v=4?s=50" width="50px;" alt="Shah Mahdi Hasan "/><br /><sub><b>Shah Mahdi Hasan </b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3Atanweer-mahdi" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=tanweer-mahdi" title="Code">💻</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Atanweer-mahdi" title="Reviewed Pull Requests">👀</a> <a href="#userTesting-tanweer-mahdi" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/pitmonticone"><img src="https://avatars.githubusercontent.com/u/38562595?v=4?s=50" width="50px;" alt="Pietro Monticone"/><br /><sub><b>Pietro Monticone</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3Apitmonticone" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=pitmonticone" title="Documentation">📖</a> <a href="#ideas-pitmonticone" title="Ideas, planning, and feedback.">💡</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sheevy"><img src="https://avatars.githubusercontent.com/u/1525683?v=4?s=50" width="50px;" alt="Mateusz Kubica"/><br /><sub><b>Mateusz Kubica</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=sheevy" title="Documentation">📖</a> <a href="#ideas-sheevy" title="Ideas, planning, and feedback.">💡</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://jaywadekar.github.io/"><img src="https://avatars.githubusercontent.com/u/5493388?v=4?s=50" width="50px;" alt="Jay Wadekar"/><br /><sub><b>Jay Wadekar</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3AJayWadekar" title="Bug reports">🐛</a> <a href="#ideas-JayWadekar" title="Ideas, planning, and feedback.">💡</a> <a href="#promotion-JayWadekar" title="Promotion">📣</a> <a href="#research-JayWadekar" title="Research">🔬</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ablaom"><img src="https://avatars.githubusercontent.com/u/30517088?v=4?s=50" width="50px;" alt="Anthony Blaom, PhD"/><br /><sub><b>Anthony Blaom, PhD</b></sub></a><br /><a href="#infra-ablaom" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-ablaom" title="Ideas, planning, and feedback.">💡</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Aablaom" title="Reviewed Pull Requests">👀</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Jgmedina95"><img src="https://avatars.githubusercontent.com/u/97254349?v=4?s=50" width="50px;" alt="Jgmedina95"/><br /><sub><b>Jgmedina95</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3AJgmedina95" title="Bug reports">🐛</a> <a href="#ideas-Jgmedina95" title="Ideas, planning, and feedback.">💡</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3AJgmedina95" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/mcabbott"><img src="https://avatars.githubusercontent.com/u/32575566?v=4?s=50" width="50px;" alt="Michael Abbott"/><br /><sub><b>Michael Abbott</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=mcabbott" title="Code">💻</a> <a href="#ideas-mcabbott" title="Ideas, planning, and feedback.">💡</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Amcabbott" title="Reviewed Pull Requests">👀</a> <a href="#tool-mcabbott" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/oscardssmith"><img src="https://avatars.githubusercontent.com/u/11729272?v=4?s=50" width="50px;" alt="Oscar Smith"/><br /><sub><b>Oscar Smith</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=oscardssmith" title="Code">💻</a> <a href="#ideas-oscardssmith" title="Ideas, planning, and feedback.">💡</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://ericphanson.com/"><img src="https://avatars.githubusercontent.com/u/5846501?v=4?s=50" width="50px;" alt="Eric Hanson"/><br /><sub><b>Eric Hanson</b></sub></a><br /><a href="#ideas-ericphanson" title="Ideas, planning, and feedback.">💡</a> <a href="#promotion-ericphanson" title="Promotion">📣</a> <a href="#userTesting-ericphanson" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/henriquebecker91"><img src="https://avatars.githubusercontent.com/u/14113435?v=4?s=50" width="50px;" alt="Henrique Becker"/><br /><sub><b>Henrique Becker</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/commits?author=henriquebecker91" title="Code">💻</a> <a href="#ideas-henriquebecker91" title="Ideas, planning, and feedback.">💡</a> <a href="https://github.com/MilesCranmer/PySR/pulls?q=is%3Apr+reviewed-by%3Ahenriquebecker91" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/qwertyjl"><img src="https://avatars.githubusercontent.com/u/110912592?v=4?s=50" width="50px;" alt="qwertyjl"/><br /><sub><b>qwertyjl</b></sub></a><br /><a href="https://github.com/MilesCranmer/PySR/issues?q=author%3Aqwertyjl" title="Bug reports">🐛</a> <a href="https://github.com/MilesCranmer/PySR/commits?author=qwertyjl" title="Documentation">📖</a> <a href="#ideas-qwertyjl" title="Ideas, planning, and feedback.">💡</a> <a href="#userTesting-qwertyjl" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://huijzer.xyz/"><img src="https://avatars.githubusercontent.com/u/20724914?v=4?s=50" width="50px;" alt="Rik Huijzer"/><br /><sub><b>Rik Huijzer</b></sub></a><br /><a href="#ideas-rikhuijzer" title="Ideas, planning, and feedback.">💡</a> <a href="#infra-rikhuijzer" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/GCaptainNemo"><img src="https://avatars.githubusercontent.com/u/43086239?v=4?s=50" width="50px;" alt="Hongyu Wang"/><br /><sub><b>Hongyu Wang</b></sub></a><br /><a href="#ideas-GCaptainNemo" title="Ideas, planning, and feedback.">💡</a> <a href="#promotion-GCaptainNemo" title="Promotion">📣</a> <a href="#research-GCaptainNemo" title="Research">🔬</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ZehaoJin"><img src="https://avatars.githubusercontent.com/u/50961376?v=4?s=50" width="50px;" alt="Zehao Jin"/><br /><sub><b>Zehao Jin</b></sub></a><br /><a href="#research-ZehaoJin" title="Research">🔬</a> <a href="#promotion-ZehaoJin" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/tmengel"><img src="https://avatars.githubusercontent.com/u/38924390?v=4?s=50" width="50px;" alt="Tanner Mengel"/><br /><sub><b>Tanner Mengel</b></sub></a><br /><a href="#research-tmengel" title="Research">🔬</a> <a href="#promotion-tmengel" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/agrundner24"><img src="https://avatars.githubusercontent.com/u/38557656?v=4?s=50" width="50px;" alt="Arthur Grundner"/><br /><sub><b>Arthur Grundner</b></sub></a><br /><a href="#research-agrundner24" title="Research">🔬</a> <a href="#promotion-agrundner24" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sjwetzel"><img src="https://avatars.githubusercontent.com/u/24393721?v=4?s=50" width="50px;" alt="sjwetzel"/><br /><sub><b>sjwetzel</b></sub></a><br /><a href="#research-sjwetzel" title="Research">🔬</a> <a href="#promotion-sjwetzel" title="Promotion">📣</a> <a href="#userTesting-sjwetzel" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://sauravmaheshkar.github.io/"><img src="https://avatars.githubusercontent.com/u/61241031?v=4?s=50" width="50px;" alt="Saurav Maheshkar"/><br /><sub><b>Saurav Maheshkar</b></sub></a><br /><a href="#tool-SauravMaheshkar" title="Tools">🔧</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
