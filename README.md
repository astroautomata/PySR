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

### For AI agents

If you are an AI agent (or want to teach yours how to use PySR well), there is a self-contained skill file at [`skills/pysr/SKILL.md`](skills/pysr/SKILL.md), distilled from the documentation and hundreds of forum threads. Point your agent at the file, or install it in the [Agent Skills](https://agentskills.io) format:

```bash
mkdir -p ~/.claude/skills/pysr && curl -o ~/.claude/skills/pysr/SKILL.md \
    https://raw.githubusercontent.com/astroautomata/PySR/master/skills/pysr/SKILL.md
```

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
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/adil-soubki"><img src="https://avatars.githubusercontent.com/u/5231841?v=4?s=50" width="50px;" alt="Adil"/><br /><sub><b>Adil</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://cjdoris.github.io/"><img src="https://avatars.githubusercontent.com/u/1844215?v=4?s=50" width="50px;" alt="Christopher Rowley"/><br /><sub><b>Christopher Rowley</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.linkedin.com/in/markkittisopikul/"><img src="https://avatars.githubusercontent.com/u/8062771?v=4?s=50" width="50px;" alt="Mark Kittisopikul"/><br /><sub><b>Mark Kittisopikul</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/tttc3"><img src="https://avatars.githubusercontent.com/u/97948946?v=4?s=50" width="50px;" alt="T Coxon"/><br /><sub><b>T Coxon</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/DhananjayAshok"><img src="https://avatars.githubusercontent.com/u/46792537?v=4?s=50" width="50px;" alt="Dhananjay Ashok"/><br /><sub><b>Dhananjay Ashok</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://gitlab.com/johanbluecreek"><img src="https://avatars.githubusercontent.com/u/852554?v=4?s=50" width="50px;" alt="Johan Blåbäck"/><br /><sub><b>Johan Blåbäck</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://mathopt.de/people/martensen/index.php"><img src="https://avatars.githubusercontent.com/u/20998300?v=4?s=50" width="50px;" alt="JuliusMartensen"/><br /><sub><b>JuliusMartensen</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ngam"><img src="https://avatars.githubusercontent.com/u/67342040?v=4?s=50" width="50px;" alt="ngam"/><br /><sub><b>ngam</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/kazewong"><img src="https://avatars.githubusercontent.com/u/8803931?v=4?s=50" width="50px;" alt="Kaze Wong"/><br /><sub><b>Kaze Wong</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ChrisRackauckas"><img src="https://avatars.githubusercontent.com/u/1814174?v=4?s=50" width="50px;" alt="Christopher Rackauckas"/><br /><sub><b>Christopher Rackauckas</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://kidger.site/"><img src="https://avatars.githubusercontent.com/u/33688385?v=4?s=50" width="50px;" alt="Patrick Kidger"/><br /><sub><b>Patrick Kidger</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/OkonSamuel"><img src="https://avatars.githubusercontent.com/u/39421418?v=4?s=50" width="50px;" alt="Okon Samuel"/><br /><sub><b>Okon Samuel</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/w2ll2am"><img src="https://avatars.githubusercontent.com/u/16038228?v=4?s=50" width="50px;" alt="William Booth-Clibborn"/><br /><sub><b>William Booth-Clibborn</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ayagh19"><img src="https://avatars.githubusercontent.com/u/124587945?v=4?s=50" width="50px;" alt="Aya Ghaleb"/><br /><sub><b>Aya Ghaleb</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/gca30"><img src="https://avatars.githubusercontent.com/u/124273598?v=4?s=50" width="50px;" alt="gca30"/><br /><sub><b>gca30</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/nmheim"><img src="https://avatars.githubusercontent.com/u/29552345?v=4?s=50" width="50px;" alt="Niklas Heim"/><br /><sub><b>Niklas Heim</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/atharvas"><img src="https://avatars.githubusercontent.com/u/20322919?v=4?s=50" width="50px;" alt="Atharva Sehgal"/><br /><sub><b>Atharva Sehgal</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/wkharold"><img src="https://avatars.githubusercontent.com/u/103685?v=4?s=50" width="50px;" alt="wkharold"/><br /><sub><b>wkharold</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://wsmoses.com"><img src="https://avatars.githubusercontent.com/u/1260124?v=4?s=50" width="50px;" alt="William Moses"/><br /><sub><b>William Moses</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://grezde.github.io"><img src="https://avatars.githubusercontent.com/u/43924925?v=4?s=50" width="50px;" alt="Ardeleanu Cristian"/><br /><sub><b>Ardeleanu Cristian</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/gm89uk"><img src="https://avatars.githubusercontent.com/u/127948719?v=4?s=50" width="50px;" alt="gm89uk"/><br /><sub><b>gm89uk</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://pablo-lemos.github.io/"><img src="https://avatars.githubusercontent.com/u/38078898?v=4?s=50" width="50px;" alt="Pablo Lemos"/><br /><sub><b>Pablo Lemos</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Moelf"><img src="https://avatars.githubusercontent.com/u/5306213?v=4?s=50" width="50px;" alt="Jerry Ling"/><br /><sub><b>Jerry Ling</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/CharFox1"><img src="https://avatars.githubusercontent.com/u/35052672?v=4?s=50" width="50px;" alt="Charles Fox"/><br /><sub><b>Charles Fox</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/johannbrehmer"><img src="https://avatars.githubusercontent.com/u/17068560?v=4?s=50" width="50px;" alt="Johann Brehmer"/><br /><sub><b>Johann Brehmer</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.cosmicmar.com/"><img src="https://avatars.githubusercontent.com/u/1510968?v=4?s=50" width="50px;" alt="Marius Millea"/><br /><sub><b>Marius Millea</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://gitlab.com/cobac"><img src="https://avatars.githubusercontent.com/u/27872944?v=4?s=50" width="50px;" alt="Coba"/><br /><sub><b>Coba</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/foxtran"><img src="https://avatars.githubusercontent.com/u/39676482?v=4?s=50" width="50px;" alt="foxtran"/><br /><sub><b>foxtran</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://smhasan.com/"><img src="https://avatars.githubusercontent.com/u/36223598?v=4?s=50" width="50px;" alt="Shah Mahdi Hasan "/><br /><sub><b>Shah Mahdi Hasan </b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://aluthge.com"><img src="https://avatars.githubusercontent.com/u/5619885?v=4?s=50" width="50px;" alt="Dilum Aluthge"/><br /><sub><b>Dilum Aluthge</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/SebastianM-C"><img src="https://avatars.githubusercontent.com/u/31181429?v=4?s=50" width="50px;" alt="Sebastian Micluța-Câmpeanu"/><br /><sub><b>Sebastian Micluța-Câmpeanu</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://neuroscience.wustl.edu/people/timothy-holy-phd/"><img src="https://avatars.githubusercontent.com/u/1525481?v=4?s=50" width="50px;" alt="Tim Holy"/><br /><sub><b>Tim Holy</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/BrotherHa"><img src="https://avatars.githubusercontent.com/u/190199534?v=4?s=50" width="50px;" alt="BrotherHa"/><br /><sub><b>BrotherHa</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://wthompson.space"><img src="https://avatars.githubusercontent.com/u/7330605?v=4?s=50" width="50px;" alt="William Thompson"/><br /><sub><b>William Thompson</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://abzu.ai"><img src="https://avatars.githubusercontent.com/u/2547785?v=4?s=50" width="50px;" alt="Tom Jelen"/><br /><sub><b>Tom Jelen</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.miguelromao.me/"><img src="https://avatars.githubusercontent.com/u/7794475?v=4?s=50" width="50px;" alt="Miguel Crispim Romao"/><br /><sub><b>Miguel Crispim Romao</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/adienes"><img src="https://avatars.githubusercontent.com/u/51664769?v=4?s=50" width="50px;" alt="Andy Dienes"/><br /><sub><b>Andy Dienes</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://singhharsh.in"><img src="https://avatars.githubusercontent.com/u/143034341?v=4?s=50" width="50px;" alt="Harsh Singh "/><br /><sub><b>Harsh Singh </b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/pitmonticone"><img src="https://avatars.githubusercontent.com/u/38562595?v=4?s=50" width="50px;" alt="Pietro Monticone"/><br /><sub><b>Pietro Monticone</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sheevy"><img src="https://avatars.githubusercontent.com/u/1525683?v=4?s=50" width="50px;" alt="Mateusz Kubica"/><br /><sub><b>Mateusz Kubica</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/WilliamBC-SL"><img src="https://avatars.githubusercontent.com/u/118170949?v=4?s=50" width="50px;" alt="William Booth-Clibborn"/><br /><sub><b>William Booth-Clibborn</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://raulpl.github.io/about"><img src="https://avatars.githubusercontent.com/u/3116652?v=4?s=50" width="50px;" alt="Raúl Peralta Lozada"/><br /><sub><b>Raúl Peralta Lozada</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.linkedin.com/in/hvaara/"><img src="https://avatars.githubusercontent.com/u/1535968?v=4?s=50" width="50px;" alt="Roy Hvaara"/><br /><sub><b>Roy Hvaara</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/VishalJ99"><img src="https://avatars.githubusercontent.com/u/51826812?v=4?s=50" width="50px;" alt="Vishal Jain"/><br /><sub><b>Vishal Jain</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/spaette"><img src="https://avatars.githubusercontent.com/u/111918424?v=4?s=50" width="50px;" alt="spaette"/><br /><sub><b>spaette</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.yxliu.group"><img src="https://avatars.githubusercontent.com/u/1089344?v=4?s=50" width="50px;" alt="Yi-Xin Liu"/><br /><sub><b>Yi-Xin Liu</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/spinnau"><img src="https://avatars.githubusercontent.com/u/2995937?v=4?s=50" width="50px;" alt="spinnau"/><br /><sub><b>spinnau</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sunxd3"><img src="https://avatars.githubusercontent.com/u/5433119?v=4?s=50" width="50px;" alt="Xianda Sun"/><br /><sub><b>Xianda Sun</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://jaywadekar.github.io/"><img src="https://avatars.githubusercontent.com/u/5493388?v=4?s=50" width="50px;" alt="Jay Wadekar"/><br /><sub><b>Jay Wadekar</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ablaom"><img src="https://avatars.githubusercontent.com/u/30517088?v=4?s=50" width="50px;" alt="Anthony Blaom, PhD"/><br /><sub><b>Anthony Blaom, PhD</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Jgmedina95"><img src="https://avatars.githubusercontent.com/u/97254349?v=4?s=50" width="50px;" alt="Jgmedina95"/><br /><sub><b>Jgmedina95</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/mcabbott"><img src="https://avatars.githubusercontent.com/u/32575566?v=4?s=50" width="50px;" alt="Michael Abbott"/><br /><sub><b>Michael Abbott</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/oscardssmith"><img src="https://avatars.githubusercontent.com/u/11729272?v=4?s=50" width="50px;" alt="Oscar Smith"/><br /><sub><b>Oscar Smith</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://ericphanson.com/"><img src="https://avatars.githubusercontent.com/u/5846501?v=4?s=50" width="50px;" alt="Eric Hanson"/><br /><sub><b>Eric Hanson</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/henriquebecker91"><img src="https://avatars.githubusercontent.com/u/14113435?v=4?s=50" width="50px;" alt="Henrique Becker"/><br /><sub><b>Henrique Becker</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/qwertyjl"><img src="https://avatars.githubusercontent.com/u/110912592?v=4?s=50" width="50px;" alt="qwertyjl"/><br /><sub><b>qwertyjl</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://huijzer.xyz/"><img src="https://avatars.githubusercontent.com/u/20724914?v=4?s=50" width="50px;" alt="Rik Huijzer"/><br /><sub><b>Rik Huijzer</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/GCaptainNemo"><img src="https://avatars.githubusercontent.com/u/43086239?v=4?s=50" width="50px;" alt="Hongyu Wang"/><br /><sub><b>Hongyu Wang</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ZehaoJin"><img src="https://avatars.githubusercontent.com/u/50961376?v=4?s=50" width="50px;" alt="Zehao Jin"/><br /><sub><b>Zehao Jin</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/tmengel"><img src="https://avatars.githubusercontent.com/u/38924390?v=4?s=50" width="50px;" alt="Tanner Mengel"/><br /><sub><b>Tanner Mengel</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/agrundner24"><img src="https://avatars.githubusercontent.com/u/38557656?v=4?s=50" width="50px;" alt="Arthur Grundner"/><br /><sub><b>Arthur Grundner</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sjwetzel"><img src="https://avatars.githubusercontent.com/u/24393721?v=4?s=50" width="50px;" alt="sjwetzel"/><br /><sub><b>sjwetzel</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://sauravmaheshkar.github.io/"><img src="https://avatars.githubusercontent.com/u/61241031?v=4?s=50" width="50px;" alt="Saurav Maheshkar"/><br /><sub><b>Saurav Maheshkar</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/chris-soelistyo"><img src="https://avatars.githubusercontent.com/u/68875981?v=4?s=50" width="50px;" alt="chris-soelistyo"/><br /><sub><b>chris-soelistyo</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://ilyaorson.gitlab.io"><img src="https://avatars.githubusercontent.com/u/12092488?v=4?s=50" width="50px;" alt="Ilya Orson "/><br /><sub><b>Ilya Orson </b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://hftsoi.github.io"><img src="https://avatars.githubusercontent.com/u/51976330?v=4?s=50" width="50px;" alt="Ho Fung Tsoi"/><br /><sub><b>Ho Fung Tsoi</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/LionessOfCintra"><img src="https://avatars.githubusercontent.com/u/92221853?v=4?s=50" width="50px;" alt="LionessOfCintra"/><br /><sub><b>LionessOfCintra</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/manuel-morales-a"><img src="https://avatars.githubusercontent.com/u/64017590?v=4?s=50" width="50px;" alt="Manuel Morales "/><br /><sub><b>Manuel Morales </b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://paulomontero.github.io"><img src="https://avatars.githubusercontent.com/u/23636178?v=4?s=50" width="50px;" alt="Paulo Montero Camacho"/><br /><sub><b>Paulo Montero Camacho</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/luna026"><img src="https://avatars.githubusercontent.com/u/88938665?v=4?s=50" width="50px;" alt="Writu Dasgupta"/><br /><sub><b>Writu Dasgupta</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/anubhavkamal"><img src="https://avatars.githubusercontent.com/u/23038512?v=4?s=50" width="50px;" alt="Anubhav Kamal"/><br /><sub><b>Anubhav Kamal</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/anthony-sun"><img src="https://avatars.githubusercontent.com/u/115842064?v=4?s=50" width="50px;" alt="anthony-sun"/><br /><sub><b>anthony-sun</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://nithouson.github.io"><img src="https://avatars.githubusercontent.com/u/26868834?v=4?s=50" width="50px;" alt="Hao Guo"/><br /><sub><b>Hao Guo</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/TrailblazerH"><img src="https://avatars.githubusercontent.com/u/177746076?v=4?s=50" width="50px;" alt="Trailblazer"/><br /><sub><b>Trailblazer</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/christospliakos"><img src="https://avatars.githubusercontent.com/u/64842094?v=4?s=50" width="50px;" alt="Christos Pliakos"/><br /><sub><b>Christos Pliakos</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/zouzaxd"><img src="https://avatars.githubusercontent.com/u/103605983?v=4?s=50" width="50px;" alt="Sousa Neto"/><br /><sub><b>Sousa Neto</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/LeoVoltolini"><img src="https://avatars.githubusercontent.com/u/94749527?v=4?s=50" width="50px;" alt="Leonardo Voltolini"/><br /><sub><b>Leonardo Voltolini</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/con13375"><img src="https://avatars.githubusercontent.com/u/19805622?v=4?s=50" width="50px;" alt="Daniel Eduardo Conde Villatoro"/><br /><sub><b>Daniel Eduardo Conde Villatoro</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sambeckers"><img src="https://avatars.githubusercontent.com/u/127021792?v=4?s=50" width="50px;" alt="Sam Beckers"/><br /><sub><b>Sam Beckers</b></sub></a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
