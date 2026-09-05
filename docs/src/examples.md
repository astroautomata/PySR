# Examples

Each example below is a short, self-contained script you can paste and run.
The groups run outwards from the core of PySR: the first four cover what most
searches need, then come the knobs that shape a search and the tools for
watching one, and the last two search over values that are not numbers at all.

## [Getting started](/examples/getting-started)

- [Simple search](/examples/getting-started#simple-search): the smallest complete search, from data to a printed expression.
- [Custom operator](/examples/getting-started#custom-operator): defining your own operator and its sympy equivalent.
- [Multiple outputs](/examples/getting-started#multiple-outputs): fitting several targets at once.
- [Plotting an expression](/examples/getting-started#plotting-an-expression): getting LaTeX and predictions out of a fitted model.
- [Feature selection](/examples/getting-started#feature-selection): cutting a wide tabular dataset down to the features that matter.
- [Denoising](/examples/getting-started#denoising): preprocessing a noisy target with a Gaussian process.

## [Expression specifications](/examples/expression-specifications)

- [Expression specifications](/examples/expression-specifications#expression-specifications): searching a fixed functional form, with templates, parametric expressions, and other structured search spaces.
- [Recovering a magnetic field from force measurements](/examples/expression-specifications#recovering-a-magnetic-field-from-force-measurements): a template that fixes the known physics and searches only the unknown functions.

## [Objectives and losses](/examples/objectives)

- [Custom objectives](/examples/objectives#custom-objectives): writing a full Julia objective instead of an elementwise loss.
- [Writing the objective in Python](/examples/objectives#writing-the-objective-in-python): a pure-Python objective behind a thin Julia shim.
- [Swinging up a cart-pole with a rollout objective](/examples/objectives#swinging-up-a-cart-pole-with-a-rollout-objective): scoring a candidate by simulating it as a controller.
- [Inventing a pseudorandom generator with no target](/examples/objectives#inventing-a-pseudorandom-generator-with-no-target): an objective that scores behaviour when there is no `y` to fit.

## [Physics and units](/examples/physics)

- [Dimensional constraints](/examples/physics#dimensional-constraints): restricting the search to dimensionally valid expressions.
- [Using differential operators](/examples/physics#using-differential-operators): differentiating candidate expressions inside the search.
- [Discovering a PDE](/examples/physics#discovering-a-pde): recovering a partial differential equation from sampled fields.

## [Search behaviour](/examples/search-behaviour)

- [Automatic batching on a large dataset](/examples/search-behaviour#automatic-batching-on-a-large-dataset): keeping a search cheap when the data is long.
- [Operators of any arity](/examples/search-behaviour#operators-of-any-arity): operator sets keyed by number of arguments.
- [Mutations and plugins](/examples/search-behaviour#mutations-and-plugins): reweighting the mutation table and attaching plugins.
- [Adaptive mutation weights](/examples/search-behaviour#adaptive-mutation-weights): letting the run learn which mutations are paying off.
- [The backsolve mutation](/examples/search-behaviour#the-backsolve-mutation): inverting the operators above a subtree to fit its replacement.

## [Instrumentation and workflow](/examples/instrumentation)

- [Using TensorBoard for logging](/examples/instrumentation#using-tensorboard-for-logging): watching a search with the TensorBoard logger.
- [Recording the genealogy of a search](/examples/instrumentation#recording-the-genealogy-of-a-search): writing a trace of every mutation and crossover.
- [Closing an agent loop with `guesses=`](/examples/instrumentation#closing-an-agent-loop-with-guesses): seeding a search with expressions produced from the previous one.

## [Value types](/examples/value-types)

- [Complex numbers](/examples/value-types#complex-numbers): searching over complex-valued data.
- [Julia packages and types](/examples/value-types#julia-packages-and-types): pulling a Julia package into the search and using it inside an operator.
- [Custom value types](/examples/value-types#custom-value-types): searching over a type of your own instead of a float.

## [Beyond numeric values](/examples/beyond-numeric-values)

- [Breaking an affine cipher with a letter type](/examples/beyond-numeric-values#breaking-an-affine-cipher-with-a-letter-type): a value type carrying letters rather than numbers.
- [Rediscovering Conway's Game of Life](/examples/beyond-numeric-values#rediscovering-conway-s-game-of-life): recovering a cellular automaton rule from its transition table.
- [Searching over machine words](/examples/beyond-numeric-values#searching-over-machine-words): a bit-exact search using the integer instruction set.
- [Turtle graphics: searching over drawings](/examples/beyond-numeric-values#turtle-graphics-searching-over-drawings): expressions whose value is a picture.

For the many other features available in PySR, please read the [Options section](/options).
