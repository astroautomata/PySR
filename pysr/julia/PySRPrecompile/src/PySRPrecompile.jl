module PySRPrecompile

using PrecompileTools: @compile_workload
using SymbolicRegression
using PythonCall
using Serialization

precompile(Tuple{Type{PythonCall.PyArray},PythonCall.Py})
for T in (
    SymbolicRegression.Options,
    SymbolicRegression.OperatorEnum,
    SymbolicRegression.ExpressionSpec,
    SymbolicRegression.ExternalStop,
    IOBuffer,
    Symbol,
)
    precompile(
        Tuple{typeof(PythonCall.JlWrap.pyjlany_call),Type{T},PythonCall.Py,PythonCall.Py}
    )
end
let rule = PythonCall.Convert.pyconvert_fix(
        Dict{Symbol,Any}, PythonCall.Convert.pyconvert_rule_mapping
    )
    precompile(Tuple{typeof(rule),PythonCall.Py})
end
for T in (Vector{PythonCall.Py}, Vector{Any}, Vector, Tuple, NamedTuple)
    rule = PythonCall.Convert.pyconvert_fix(T, PythonCall.Convert.pyconvert_rule_iterable)
    precompile(Tuple{typeof(rule),PythonCall.Py})
end
precompile(Tuple{typeof(PythonCall.pyconvert),Type{Array},PythonCall.Py})
precompile(
    Tuple{
        typeof(PythonCall.Convert._pyconvert_rule_iterable),
        Vector{String},
        PythonCall.Py,
        Type{Any},
    },
)
for N in (1, 2)
    precompile(Tuple{typeof(copy),PythonCall.PyArray{Float32,N,true,true,Float32}})
end

function pysr_shaped_workload()
    operators = SymbolicRegression.OperatorEnum(((), (+, -, /, *)))
    for T in SymbolicRegression.PRECOMPILE_TYPES
        X = randn(T, 3, 30)
        y = randn(T, 30)
        options = SymbolicRegression.Options(;
            operators,
            populations=3,
            population_size=50,
            tournament_selection_n=6,
            ncycles_per_iteration=30,
            mutation_weights=(;
                mutate_constant=1.0,
                mutate_operator=1.0,
                swap_operands=1.0,
                add_node=1.0,
                insert_node=1.0,
                delete_node=1.0,
                simplify=1.0,
                randomize=1.0,
                do_nothing=1.0,
                optimize=1.0,
            ),
            fraction_replaced=0.2,
            fraction_replaced_hof=0.2,
            define_helper_functions=false,
            optimizer_probability=0.05,
            save_to_file=false,
        )
        search_kwargs = (;
            external_stop=SymbolicRegression.ExternalStop(),
            variable_names=["x0", "x1", "x2"],
            display_variable_names=["x0", "x1", "x2"],
            y_variable_names=nothing,
            X_units=nothing,
            y_units=nothing,
            logger=nothing,
            run_id="precompile",
            progress=true,
            runtests=true,
            parallelism="multithreading",
            verbosity=1,
        )
        state = SymbolicRegression.equation_search(
            X,
            y;
            niterations=3,
            options,
            return_state=true,
            saved_state=nothing,
            search_kwargs...,
        )
        hof = SymbolicRegression.equation_search(
            X,
            y;
            niterations=0,
            options,
            saved_state=state,
            return_state=false,
            search_kwargs...,
        )
        SymbolicRegression.calculate_pareto_frontier(hof::SymbolicRegression.HallOfFame)
        for value in (options, state)
            buffer = IOBuffer()
            Serialization.serialize(buffer, value)
            seekstart(buffer)
            Serialization.deserialize(buffer)
        end
    end
    return nothing
end

@compile_workload begin
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            SymbolicRegression.do_precompilation(Val(:compile))
            pysr_shaped_workload()
        end
    end
end

end
