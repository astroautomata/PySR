# API Reference

`PySRRegressor` has many options for controlling a symbolic regression search.
Let's look at them below.
See the [backend documentation](backend.md#rust-backend) for installation
and selection, and [Rust options](options.md#rust-options) for limitations.

PARAMSKEY

## PySRRegressor Functions

::: pysr.PySRRegressor.fit
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.predict
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.from_file
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.sympy
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.latex
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.pytorch
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.jax
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.latex_table
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.PySRRegressor.refresh
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

## Expression Specifications

::: pysr.ExpressionSpec
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.TemplateExpressionSpec
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.AbstractExpressionSpec
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

## Type Specifications

::: pysr.TypeSpec
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

## Logger Specifications

::: pysr.TensorBoardLoggerSpec
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.AbstractLoggerSpec
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

## Mutations

::: pysr.ConstantMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.OperatorMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.FeatureMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.SwapOperandsMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.AddNodeMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.InsertNodeMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.DeleteNodeMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.RotateTreeMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.BacksolveMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.SimplifyMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.RandomizeMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.OptimizeMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.DoNothingMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.AbstractMutation
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

## Plugins

::: pysr.AdaptiveMutationWeightsPlugin
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.AdaptiveParsimonyPlugin
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.MutationBurstPlugin
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.SimulatedAnnealingPlugin
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false

::: pysr.AbstractPlugin
    options:
        show_root_heading: true
        heading_level: 3
        show_root_full_path: false
