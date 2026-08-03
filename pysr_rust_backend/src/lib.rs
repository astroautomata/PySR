use std::ops::AddAssign;

use dynamic_expressions::operator_enum::presets::{BuiltinOpsF32, BuiltinOpsF64};
use dynamic_expressions::strings::{string_tree, StringTreeOptions};
use dynamic_expressions::Operators;
use ndarray::{Array1, Array2};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use numpy::{Element, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use symbolic_regression::{equation_search, Dataset, EarlyStop, Options, OutputStyle};

const MAX_ARITY: usize = 3;
const SUPPORTED_OPTIONS: &[&str] = &[
    "seed",
    "niterations",
    "populations",
    "population_size",
    "ncycles_per_iteration",
    "batch_size",
    "complexity_of_constants",
    "complexity_of_variables",
    "maxsize",
    "maxdepth",
    "warmup_maxsize_by",
    "parsimony",
    "adaptive_parsimony_scaling",
    "crossover_probability",
    "perturbation_factor",
    "probability_negate_constant",
    "tournament_selection_n",
    "tournament_selection_p",
    "alpha",
    "optimizer_nrestarts",
    "optimizer_probability",
    "optimizer_iterations",
    "optimizer_f_calls_limit",
    "fraction_replaced",
    "fraction_replaced_hof",
    "topn",
    "print_precision",
    "max_evals",
    "timeout_in_seconds",
    "use_frequency",
    "use_frequency_in_tournament",
    "skip_mutation_failures",
    "annealing",
    "should_optimize_constants",
    "migration",
    "hof_migration",
    "progress",
    "should_simplify",
    "batching",
    "deterministic",
    "parallelism",
    "early_stop_condition",
    "mutation_weights",
];
const SUPPORTED_MUTATION_WEIGHTS: &[&str] = &[
    "mutate_constant",
    "mutate_operator",
    "mutate_feature",
    "swap_operands",
    "rotate_tree",
    "add_node",
    "insert_node",
    "delete_node",
    "simplify",
    "randomize",
    "do_nothing",
    "optimize",
    "form_connection",
    "break_connection",
];

#[pyfunction]
#[pyo3(signature = (x, y, *, options, operators, variable_names))]
fn search(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    options: &Bound<'_, PyDict>,
    operators: &Bound<'_, PyDict>,
    variable_names: Vec<String>,
) -> PyResult<Py<PyAny>> {
    if let (Ok(x), Ok(y)) = (
        x.extract::<PyReadonlyArray2<'_, f32>>(),
        y.extract::<PyReadonlyArray1<'_, f32>>(),
    ) {
        return run_search::<f32, BuiltinOpsF32>(py, x, y, options, operators, variable_names);
    }

    if let (Ok(x), Ok(y)) = (
        x.extract::<PyReadonlyArray2<'_, f64>>(),
        y.extract::<PyReadonlyArray1<'_, f64>>(),
    ) {
        return run_search::<f64, BuiltinOpsF64>(py, x, y, options, operators, variable_names);
    }

    Err(PyTypeError::new_err(
        "X and y must be NumPy arrays with matching float32 or float64 dtypes",
    ))
}

fn run_search<T, Ops>(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, T>,
    y: PyReadonlyArray1<'_, T>,
    options_dict: &Bound<'_, PyDict>,
    operators_dict: &Bound<'_, PyDict>,
    variable_names: Vec<String>,
) -> PyResult<Py<PyAny>>
where
    T: Float
        + AddAssign
        + FromPrimitive
        + ToPrimitive
        + std::fmt::Display
        + Send
        + Sync
        + Element
        + 'static,
    Ops: dynamic_expressions::OperatorSet<T = T> + Send + Sync,
{
    let dataset = build_dataset(x, y, variable_names)?;
    let operator_names = extract_operator_names(operators_dict)?;
    let operators =
        Operators::<MAX_ARITY>::from_names::<Ops, _>(operator_names.iter().map(String::as_str))
            .map_err(|err| PyValueError::new_err(format!("invalid operator selection: {err}")))?;

    let mut options = Options::<T, MAX_ARITY> {
        operators,
        ..Default::default()
    };
    apply_options(&mut options, options_dict)?;
    let parallelism = extract_parallelism(options_dict)?;

    let result = if parallelism.as_deref() == Some("serial") {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .map_err(|err| {
                PyRuntimeError::new_err(format!("failed to create serial thread pool: {err}"))
            })?;
        py.detach(|| pool.install(|| equation_search::<T, Ops, MAX_ARITY>(&dataset, &options)))
    } else {
        py.detach(|| equation_search::<T, Ops, MAX_ARITY>(&dataset, &options))
    };

    let rows = PyList::empty(py);
    for member in result.hall_of_fame.pareto_front() {
        let row = PyDict::new(py);
        row.set_item("complexity", member.complexity)?;
        row.set_item(
            "loss",
            member
                .loss
                .to_f64()
                .ok_or_else(|| PyRuntimeError::new_err("failed to convert loss to float"))?,
        )?;
        row.set_item(
            "equation",
            string_tree(
                &member.expr,
                StringTreeOptions {
                    variable_names: Some(dataset.variable_names.as_slice()),
                    pretty: false,
                    print_precision: Some(options.print_precision),
                },
            ),
        )?;
        rows.append(row)?;
    }

    let out = PyDict::new(py);
    out.set_item("backend_version", env!("CARGO_PKG_VERSION"))?;
    out.set_item("hall_of_fame", rows)?;
    Ok(out.into_any().unbind())
}

fn build_dataset<T>(
    x: PyReadonlyArray2<'_, T>,
    y: PyReadonlyArray1<'_, T>,
    variable_names: Vec<String>,
) -> PyResult<Dataset<T>>
where
    T: Float + FromPrimitive + Element,
{
    let x_view = x.as_array();
    let y_view = y.as_array();
    let shape = x_view.shape();
    let n_rows = shape[0];
    let n_features = shape[1];

    if y_view.len() != n_rows {
        return Err(PyValueError::new_err(format!(
            "X has {n_rows} rows but y has {} elements",
            y_view.len()
        )));
    }
    if !variable_names.is_empty() && variable_names.len() != n_features {
        return Err(PyValueError::new_err(format!(
            "expected {n_features} variable names, got {}",
            variable_names.len()
        )));
    }

    let mut rust_x = Array2::<T>::zeros((n_features, n_rows));
    for row in 0..n_rows {
        for feature in 0..n_features {
            rust_x[(feature, row)] = x_view[(row, feature)];
        }
    }
    let rust_y = Array1::from_iter(y_view.iter().copied());

    Ok(Dataset::with_weights_and_names(
        rust_x,
        rust_y,
        None,
        variable_names,
    ))
}

fn extract_operator_names(operators: &Bound<'_, PyDict>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for (arity_obj, item) in operators.iter() {
        let arity: usize = arity_obj.extract()?;
        if arity > MAX_ARITY {
            return Err(PyValueError::new_err(format!(
                "operators only support arities up to {MAX_ARITY}, got {arity}"
            )));
        }
        let names: Vec<String> = item.extract()?;
        for name in &names {
            let expected_arity = operator_arity(name).ok_or_else(|| {
                PyValueError::new_err(format!("unsupported builtin operator {name:?}"))
            })?;
            if arity != expected_arity {
                return Err(PyValueError::new_err(format!(
                    "operator {name:?} has arity {expected_arity} but was provided under arity {arity}"
                )));
            }
        }
        out.extend(names);
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("at least one operator is required"));
    }
    Ok(out)
}

fn operator_arity(name: &str) -> Option<usize> {
    match name {
        "sin" | "cos" | "tan" | "exp" | "log" | "sqrt" | "abs" | "abs2" => Some(1),
        "+" | "*" | "/" | "sub" => Some(2),
        _ => None,
    }
}

fn extract_parallelism(options: &Bound<'_, PyDict>) -> PyResult<Option<String>> {
    let Some(value) = options.get_item("parallelism")? else {
        return Ok(None);
    };
    let parallelism: String = value.extract()?;
    match parallelism.as_str() {
        "serial" | "multithreading" => Ok(Some(parallelism)),
        other => Err(PyValueError::new_err(format!(
            "parallelism must be 'serial' or 'multithreading', got {other:?}"
        ))),
    }
}

fn apply_options<T: Float + FromPrimitive + Send + Sync + 'static>(
    options: &mut Options<T, MAX_ARITY>,
    source: &Bound<'_, PyDict>,
) -> PyResult<()> {
    options.output_style = OutputStyle::Plain;
    validate_known_keys(source, SUPPORTED_OPTIONS, "options")?;

    assign_u64(source, "seed", &mut options.seed)?;
    assign_usize(source, "niterations", &mut options.niterations)?;
    assign_usize(source, "populations", &mut options.populations)?;
    assign_usize(source, "population_size", &mut options.population_size)?;
    assign_usize(
        source,
        "ncycles_per_iteration",
        &mut options.ncycles_per_iteration,
    )?;
    assign_usize(source, "batch_size", &mut options.batch_size)?;
    assign_u16(
        source,
        "complexity_of_constants",
        &mut options.complexity_of_constants,
    )?;
    assign_u16(
        source,
        "complexity_of_variables",
        &mut options.complexity_of_variables,
    )?;
    assign_usize(source, "maxsize", &mut options.maxsize)?;
    assign_usize(source, "maxdepth", &mut options.maxdepth)?;
    assign_f32(source, "warmup_maxsize_by", &mut options.warmup_maxsize_by)?;
    assign_f64(source, "parsimony", &mut options.parsimony)?;
    assign_f64(
        source,
        "adaptive_parsimony_scaling",
        &mut options.adaptive_parsimony_scaling,
    )?;
    assign_f64(
        source,
        "crossover_probability",
        &mut options.crossover_probability,
    )?;
    assign_f64(
        source,
        "perturbation_factor",
        &mut options.perturbation_factor,
    )?;
    assign_f64(
        source,
        "probability_negate_constant",
        &mut options.probability_negate_constant,
    )?;
    assign_usize(
        source,
        "tournament_selection_n",
        &mut options.tournament_selection_n,
    )?;
    assign_f32(
        source,
        "tournament_selection_p",
        &mut options.tournament_selection_p,
    )?;
    assign_f64(source, "alpha", &mut options.alpha)?;
    assign_usize(
        source,
        "optimizer_nrestarts",
        &mut options.optimizer_nrestarts,
    )?;
    assign_f64(
        source,
        "optimizer_probability",
        &mut options.optimizer_probability,
    )?;
    assign_usize(
        source,
        "optimizer_iterations",
        &mut options.optimizer_iterations,
    )?;
    assign_usize(
        source,
        "optimizer_f_calls_limit",
        &mut options.optimizer_f_calls_limit,
    )?;
    assign_f64(source, "fraction_replaced", &mut options.fraction_replaced)?;
    assign_f64(
        source,
        "fraction_replaced_hof",
        &mut options.fraction_replaced_hof,
    )?;
    assign_usize(source, "topn", &mut options.topn)?;
    assign_usize(source, "print_precision", &mut options.print_precision)?;
    assign_u64(source, "max_evals", &mut options.max_evals)?;
    assign_f64(
        source,
        "timeout_in_seconds",
        &mut options.timeout_in_seconds,
    )?;

    assign_bool(source, "use_frequency", &mut options.use_frequency)?;
    assign_bool(
        source,
        "use_frequency_in_tournament",
        &mut options.use_frequency_in_tournament,
    )?;
    assign_bool(
        source,
        "skip_mutation_failures",
        &mut options.skip_mutation_failures,
    )?;
    assign_bool(source, "annealing", &mut options.annealing)?;
    assign_bool(
        source,
        "should_optimize_constants",
        &mut options.should_optimize_constants,
    )?;
    assign_bool(source, "migration", &mut options.migration)?;
    assign_bool(source, "hof_migration", &mut options.hof_migration)?;
    assign_bool(source, "progress", &mut options.progress)?;
    assign_bool(source, "should_simplify", &mut options.should_simplify)?;
    assign_bool(source, "batching", &mut options.batching)?;
    assign_bool(source, "deterministic", &mut options.deterministic)?;

    if let Some(value) = source.get_item("early_stop_condition")? {
        let threshold_f64: f64 = value.extract()?;
        let threshold = T::from_f64(threshold_f64).ok_or_else(|| {
            PyValueError::new_err("early_stop_condition must be representable as backend float")
        })?;
        options.early_stop_condition = Some(EarlyStop::below(threshold));
    }

    if let Some(value) = source.get_item("mutation_weights")? {
        let weights = value.cast::<PyDict>()?;
        validate_known_keys(weights, SUPPORTED_MUTATION_WEIGHTS, "mutation_weights")?;
        assign_f64(
            weights,
            "mutate_constant",
            &mut options.mutation_weights.mutate_constant,
        )?;
        assign_f64(
            weights,
            "mutate_operator",
            &mut options.mutation_weights.mutate_operator,
        )?;
        assign_f64(
            weights,
            "mutate_feature",
            &mut options.mutation_weights.mutate_feature,
        )?;
        assign_f64(
            weights,
            "swap_operands",
            &mut options.mutation_weights.swap_operands,
        )?;
        assign_f64(
            weights,
            "rotate_tree",
            &mut options.mutation_weights.rotate_tree,
        )?;
        assign_f64(weights, "add_node", &mut options.mutation_weights.add_node)?;
        assign_f64(
            weights,
            "insert_node",
            &mut options.mutation_weights.insert_node,
        )?;
        assign_f64(
            weights,
            "delete_node",
            &mut options.mutation_weights.delete_node,
        )?;
        assign_f64(weights, "simplify", &mut options.mutation_weights.simplify)?;
        assign_f64(
            weights,
            "randomize",
            &mut options.mutation_weights.randomize,
        )?;
        assign_f64(
            weights,
            "do_nothing",
            &mut options.mutation_weights.do_nothing,
        )?;
        assign_f64(weights, "optimize", &mut options.mutation_weights.optimize)?;
        assign_f64(
            weights,
            "form_connection",
            &mut options.mutation_weights.form_connection,
        )?;
        assign_f64(
            weights,
            "break_connection",
            &mut options.mutation_weights.break_connection,
        )?;
    }

    Ok(())
}

fn validate_known_keys(
    source: &Bound<'_, PyDict>,
    allowed: &[&str],
    context: &str,
) -> PyResult<()> {
    let mut unknown = Vec::new();
    for key in source.keys() {
        let key: String = key.extract()?;
        if !allowed.contains(&key.as_str()) {
            unknown.push(key);
        }
    }
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort();
    Err(PyValueError::new_err(format!(
        "unsupported or unknown {context}: {}",
        unknown.join(", ")
    )))
}

fn assign_usize(source: &Bound<'_, PyDict>, name: &str, target: &mut usize) -> PyResult<()> {
    if let Some(value) = source.get_item(name)? {
        *target = value.extract()?;
    }
    Ok(())
}

fn assign_u16(source: &Bound<'_, PyDict>, name: &str, target: &mut u16) -> PyResult<()> {
    if let Some(value) = source.get_item(name)? {
        *target = value.extract()?;
    }
    Ok(())
}

fn assign_u64(source: &Bound<'_, PyDict>, name: &str, target: &mut u64) -> PyResult<()> {
    if let Some(value) = source.get_item(name)? {
        *target = value.extract()?;
    }
    Ok(())
}

fn assign_f32(source: &Bound<'_, PyDict>, name: &str, target: &mut f32) -> PyResult<()> {
    if let Some(value) = source.get_item(name)? {
        *target = value.extract()?;
    }
    Ok(())
}

fn assign_f64(source: &Bound<'_, PyDict>, name: &str, target: &mut f64) -> PyResult<()> {
    if let Some(value) = source.get_item(name)? {
        *target = value.extract()?;
    }
    Ok(())
}

fn assign_bool(source: &Bound<'_, PyDict>, name: &str, target: &mut bool) -> PyResult<()> {
    if let Some(value) = source.get_item(name)? {
        *target = value.extract()?;
    }
    Ok(())
}

#[pymodule]
fn symbolic_regression_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    Ok(())
}
