# Tuning and Workflow Tips

I give a short guide below on how I like to tune PySR for my applications.

First, my general tips would be to avoid using redundant operators, like how `pow` can do the same things as `square`, or how `-` (binary) and `neg` (unary) are equivalent. The fewer operators the better! Only use operators you need.

When running PySR, I usually do the following:

I run from IPython (Jupyter Notebooks don't work as well[^1]) on the head node of a slurm cluster. Passing `cluster_manager="slurm"` will make PySR set up a run over the entire allocation. I set `procs` equal to the total number of cores over my entire allocation.

I use the [tensorboard feature](https://ai.damtp.cam.ac.uk/pysr/examples/#12-using-tensorboard-for-logging) for experiment tracking.

[^1]: Jupyter Notebooks are supported by PySR, but miss out on some useful features available in IPython and Python: the progress bar, and early stopping with "q". In Jupyter you cannot interrupt a search once it has started; you have to restart the kernel. See [this issue](https://github.com/astroautomata/PySR/issues/260) for updates.

1. I start by using the default parameters.
2. I use only the operators I think it needs and no more.
3. Increase `populations` to `3*num_cores`.
4. If my dataset is more than 1000 points, I either subsample it (low-dimensional and not much noise) or set `batching=True` (high-dimensional or very noisy, so it needs to evaluate on all the data).
5. While on a laptop or single node machine, you might leave the default `ncycles_per_iteration`, on a cluster with ~100 cores I like to set `ncycles_per_iteration` to maybe `5000` or so, until the head node occupation is under `10%`. (A larger value means the workers talk less frequently to eachother, which is useful when you have many workers!)
6. Set `constraints` and `nested_constraints` as strict as possible. These can help quite a bit with exploration. Typically, if I am using `pow`, I would set `constraints={"pow": (9, 1)}`, so that power laws can only have a variable or constant as their exponent. If I am using `sin` and `cos`, I also like to set `nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}`, so that sin and cos can't be nested, which seems to happen frequently. (Although in practice I would just use `sin`, since the search could always add a phase offset!)
7. Set `maxsize` a bit larger than the final size you want. e.g., if you want a final equation of size `30`, you might set this to `35`, so that it has a bit of room to explore.
8. I typically don't use `maxdepth`, but if I do, I set it strictly, while also leaving a bit of room for exploration. e.g., if you want a final equation limited to a depth of `5`, you might set this to `6` or `7`, so that it has a bit of room to explore.
9.  Set `parsimony` equal to about the minimum loss you would expect, divided by 5-10. e.g., if you expect the final equation to have a loss of `0.001`, you might set `parsimony=0.0001`.
10. Set `weight_optimize` to some larger value, maybe `0.001`. This is very important if `ncycles_per_iteration` is large, so that optimization happens more frequently.
11. Set `turbo` to `True`. This turns on advanced loop vectorization, but is still quite experimental. It should give you a nice 20% or more speedup.
12. For final runs, after I have tuned everything, I typically set `niterations` to some very large value, and just let it run for a week until my job finishes (genetic algorithms tend not to converge, they can look like they settle down, but then find a new family of expression, and explore a new space). If I am satisfied with the current equations (which are visible either in the terminal or in the saved csv file), I quit the job early.

Since I am running in IPython, I can just hit `q` and then `<enter>` to stop the job, tweak the hyperparameters, and then start the search again.
I can also use `warm_start=True` if I wish to continue where I left off (though note that changing some parameters, like `maxsize`, are incompatible with warm starts).

Some things I try out to see if they help:

1. Play around with `complexity_of_operators`. Set operators you dislike (e.g., `pow`) to have a larger complexity.
2. Try setting `adaptive_parsimony_scaling` a bit larger, maybe up to `1000`.
3. Sometimes I try using `warmup_maxsize_by`. This is useful if you find that the search finds a very complex equation very quickly, and then gets stuck. It basically forces it to start at the simpler equations and build up complexity slowly.
4. Play around with different losses:
    - I typically try `L2DistLoss()` and `L1DistLoss()`. L1 loss is more robust to outliers compared to L2 (L1 finds the median, while L2 finds the mean of a random variable), so is often a good choice for a noisy dataset.
    - I might also provide the `weights` parameter to `fit` if there is some reasonable choice of weighting. For example, maybe I know the signal-to-noise of a particular row of `y` - I would set that SNR equal to the weights. Or, perhaps I do some sort of importance sampling, and weight the rows by importance.

Very rarely I might also try tuning the mutation weights, the crossover probability, or the optimization parameters. I never use `denoise` or `select_k_features` as I find they aren't very useful.

For large datasets I usually just randomly sample ~1000 points or so. In case all the points matter, I might use `batching=True`.

If I find the equations get very complex and I'm not sure if they are numerically precise, I might set `precision=64`.

Once a run is finished, I use the `PySRRegressor.from_file` function to load the saved search in a different process (requires the pickle file, and possibly also the `.csv` file if you quit early). I can then explore the equations, convert them to LaTeX, and plot their output.

## More Tips

You might also wish to explore the [discussions](https://github.com/astroautomata/PySR/discussions/) page for more tips, and to see if anyone else has had similar questions.
Be sure to also read through the [reference](api.md).
