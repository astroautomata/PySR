# Benchmark 1

## Backend comparison

To compare the experimental Rust backend against the default Julia backend on
the same generated dataset, run:

```bash
python benchmarks/compare_backends.py --niterations 3 --repeats 3
```

Use `--mode subprocess` to run each trial in a fresh Python process, which is
useful when measuring startup-heavy workflows:

```bash
python benchmarks/compare_backends.py --mode subprocess --niterations 1 --repeats 3
```

The script reports wall-clock timing plus the best loss, prediction MSE, and
selected equation for each backend.

The following benchmarks were ran with this command on a node on CCA's BNL cluster (40-cores). At no time was the node fully busy. The tags were put into the file `tags.txt`, and the `benchmark.sh` was copied to the root folder. This is the command used:

```bash
for x in $(cat tags.txt); do sleep 120 && git checkout $x &> /dev/null && nohup ./benchmark.sh > performance_v3_$x.txt &; done
```
with this API call in `benchmark.sh`
```python
eq = pysr(X, y, binary_operators=["plus", "mult", "div", "pow"], unary_operators=["sin"], niterations=20, procs=4, parsimony=1e-10, population_size=1000, ncyclesperiteration=1000)
```


Version | Cycles/second
--- | ---
v0.3.2 | 37526
v0.3.3 | 38400
v0.3.4 | 28700
v0.3.5 | 32700
v0.3.6 | 25900
v0.3.7 | 26600
v0.3.8 | 7470
v0.3.9 | 6760
v0.3.10 |
v0.3.11 | 19500
v0.3.12 | 19000
v0.3.13 | 15200
v0.3.14 | 14700
v0.3.15 | 42000
v0.3.23 | 64000

v0.3.10 was frozen.
