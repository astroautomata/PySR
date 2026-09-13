# Examples

These examples show how to set up a symbolic regression search and adapt it to
your data, equation structure, or objective. If you are new to PySR, begin with
Getting started. Otherwise, choose the topic that matches your task.

## [Getting started](/examples/getting-started)

- [Simple search](/examples/getting-started#simple-search): fit generated data and inspect the resulting expressions.
- [Custom operator](/examples/getting-started#custom-operator): add an operator and define its symbolic export.
- [Multiple outputs](/examples/getting-started#multiple-outputs): fit expressions for several target columns.
- [Plotting an expression](/examples/getting-started#plotting-an-expression): export LaTeX and compare predictions with target values.
- [Feature selection](/examples/getting-started#feature-selection): reduce the input variables considered by the search.
- [Denoising](/examples/getting-started#denoising): smooth a noisy target with a Gaussian process before fitting expressions.

## [Expression specifications](/examples/expression-specifications)

- [Expression specifications](/examples/expression-specifications#expression-specifications): constrain equation structure and learn category-dependent parameters.
- [Recovering a magnetic field from force measurements](/examples/expression-specifications#recovering-a-magnetic-field-from-force-measurements): supply the known force law in a template and search for its unknown functions.

## [Objectives and losses](/examples/objectives)

- [Custom objectives](/examples/objectives#custom-objectives): score complete candidate expressions with a Julia function.
- [Writing the objective in Python](/examples/objectives#writing-the-objective-in-python): use a Python function to score candidates during the search.
- [Swinging up a cart-pole with a rollout objective](/examples/objectives#swinging-up-a-cart-pole-with-a-rollout-objective): evaluate each candidate controller in a simulation.
- [Inventing a pseudorandom generator with no target](/examples/objectives#inventing-a-pseudorandom-generator-with-no-target): score generators without observed target outputs.

## [Physics and units](/examples/physics)

- [Dimensional constraints](/examples/physics#dimensional-constraints): attach physical units to the search problem.
- [Using differential operators](/examples/physics#using-differential-operators): include derivatives in an expression template.
- [Discovering a PDE](/examples/physics#discovering-a-pde): search for a differential equation using sampled field data.

## [Search behaviour](/examples/search-behaviour)

- [Automatic batching on a large dataset](/examples/search-behaviour#automatic-batching-on-a-large-dataset): evaluate candidates on batches drawn from the data.
- [Operators of any arity](/examples/search-behaviour#operators-of-any-arity): supply operators that take more than two arguments.
- [Mutations and plugins](/examples/search-behaviour#mutations-and-plugins): configure mutation weights and search plugins.
- [Adaptive mutation weights](/examples/search-behaviour#adaptive-mutation-weights): adjust mutation choices as the search progresses.
- [The backsolve mutation](/examples/search-behaviour#the-backsolve-mutation): derive targets for a replacement subtree from the operators above it.

## [Instrumentation and workflow](/examples/instrumentation)

- [Using TensorBoard for logging](/examples/instrumentation#using-tensorboard-for-logging): log progress and metrics from the search for inspection in TensorBoard.
- [Recording the genealogy of a search](/examples/instrumentation#recording-the-genealogy-of-a-search): inspect the relationships between expressions in a recorded search.
- [Closing an agent loop with `guesses=`](/examples/instrumentation#closing-an-agent-loop-with-guesses): supply candidate expressions as guesses for another search round.

## [Value types](/examples/value-types)

- [Complex numbers](/examples/value-types#complex-numbers): fit expressions to complex-valued data.
- [Julia packages and types](/examples/value-types#julia-packages-and-types): use a Julia package inside a search operator.
- [Custom value types](/examples/value-types#custom-value-types): define the values used by expressions and how PySR handles them.

## [Beyond numeric values](/examples/beyond-numeric-values)

- [Breaking an affine cipher with a letter type](/examples/beyond-numeric-values#breaking-an-affine-cipher-with-a-letter-type): decode text with expressions that operate on letters.
- [Rediscovering Conway's Game of Life](/examples/beyond-numeric-values#rediscovering-conway-s-game-of-life): learn the update rule from a finite table of cell transitions.
- [Searching over machine words](/examples/beyond-numeric-values#searching-over-machine-words): build expressions from bitwise operations.
- [Turtle graphics: searching over drawings](/examples/beyond-numeric-values#turtle-graphics-searching-over-drawings): search for a program that draws a target shape.
- [Evolving a hopping controller](/examples/beyond-numeric-values#evolving-a-hopping-controller): replay a three-joint controller learned through simulated rollouts.

See the [Options section](/options) for other ways to configure the search.
