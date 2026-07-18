# Slurm (Multi-node)

PySR supports running across multiple nodes in an existing Slurm allocation with
`cluster_manager="slurm"`. Request the resources with `sbatch` or `salloc`, then
PySR will launch Julia workers across the allocation using
`SlurmClusterManager.jl`.

For example, this job requests three workers on each of two nodes:

```bash
#!/bin/bash
#SBATCH --job-name=pysr
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --time=01:00:00

set -euo pipefail
python pysr_script.py
```

The corresponding `pysr_script.py` is:

```python
import numpy as np
from pysr import PySRRegressor

X = np.random.RandomState(0).randn(1000, 2)
y = X[:, 0] + 2 * X[:, 1]

model = PySRRegressor(
    niterations=200,
    populations=2,
    parallelism="multiprocessing",
    cluster_manager="slurm",
    procs=6,
)
model.fit(X, y)
print(model)
```

Submit the job with:

```bash
sbatch pysr_job.sh
```

`procs` must equal the allocation's total task count, given by `--ntasks` or
`--nodes` multiplied by `--ntasks-per-node`. Run the Python script once inside
the allocation; do not wrap it in `srun`.
