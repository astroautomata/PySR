"""Executable coverage for the shipped documentation examples.

Every example in `examples/` is a plain script a reader can run directly. Most also
expose the handful of names this suite needs, so the documented result is checked by
running the documented search rather than by re-describing it here:

    X, y                  the problem
    VARIABLE_NAMES        list[str], or absent
    MODEL_KWARGS          every PySRRegressor argument except random_state
    check(model) -> bool  the example's own definition of success

Two examples cannot be driven that way and carry their own runner, so they get their
own tests below. An example whose result stops reproducing fails here instead of
rotting in the docs.

Wall times are single-seed measurements on a `genx` node, one thread, PySR 2.1.0.
They pick the marker, not an assertion: timing a search on unknown CI hardware would
be a flaky test.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent

# name -> slowest measured single-seed wall time in seconds
EXAMPLES = {
    "adaptive_mutation_weights": 28,
    "affine_cipher": 5,
    "game_of_life": 73,
    "automatic_batching": 90,
    "backsolve_mutation": 179,
    "morton_interleave": 222,
    "mutations_and_plugins": 214,
    "any_arity": 362,
    "turtle_graphics": 2414,
    "cartpole_objective": 6866,
    "prng_period": 3230,
    "magnetic_field": 5507,
}

# Examples that cannot go through a single PySRRegressor fit and carry their own runner.
SELF_DRIVEN = {"search_trace", "agent_loop_guesses"}

# Examples whose search recovers its target on some seeds and not others. The measured
# rate is reported in the docs; asserting recovery on one fixed seed would bake in a
# lucky draw, so these are checked for completing and producing a usable front.
STOCHASTIC = {"affine_freemod": 190}

FAST_SECONDS = 100


def load(name):
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(name):
    from pysr import PySRRegressor

    module = load(name)
    model = PySRRegressor(**module.MODEL_KWARGS, random_state=0)

    fit_kwargs = {}
    if getattr(module, "VARIABLE_NAMES", None):
        fit_kwargs["variable_names"] = module.VARIABLE_NAMES
    model.fit(module.X, module.y, **fit_kwargs)

    assert module.check(model), f"{name} no longer reproduces its documented result"


@pytest.mark.parametrize("name", [n for n, s in EXAMPLES.items() if s <= FAST_SECONDS])
def test_example(name):
    run(name)


@pytest.mark.slow
@pytest.mark.parametrize("name", [n for n, s in EXAMPLES.items() if s > FAST_SECONDS])
def test_example_slow(name):
    run(name)


@pytest.mark.slow
def test_agent_loop_reaches_floor():
    """The loop's premise: a cold search misses the floor and fed-back guesses reach it."""
    module = load("agent_loop_guesses")
    rounds = module.main(seed=0)

    assert len(rounds) > 1, "a cold start alone succeeded; the example has no loop left"
    assert rounds[0]["loss"] > 2 * module.NOISE_FLOOR, "round 0 already at the floor"
    assert (
        rounds[-1]["loss"] <= 2 * module.NOISE_FLOOR
    ), "the loop never reached the floor"


@pytest.mark.slow
def test_search_trace_genealogy(tmp_path):
    """The trace resolves into a graph the winner's ancestry can be walked back through.

    `use_tracing` pulls JSON3 into the PySR Julia environment on first use, so a cold
    environment pays an install here before the search starts.
    """
    module = load("search_trace")
    trace_path = str(tmp_path / "search_trace.jsonl")

    module.run_search(trace_path, niterations=3)
    nodes, edges = module.load_trace(trace_path)

    assert module.check(nodes, edges), "the recorded genealogy no longer resolves"


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(STOCHASTIC))
def test_stochastic_example_completes(name):
    """The search runs and yields a front, whether or not this seed finds the answer."""
    from pysr import PySRRegressor

    module = load(name)
    model = PySRRegressor(**module.MODEL_KWARGS, random_state=0)
    model.fit(module.X, module.y)

    assert not model.equations_.empty, f"{name} produced no front"
    assert len(module.decode(model)) == len(module.PLAIN)


def test_every_example_is_registered():
    """A new file in examples/ must be registered, or it ships with no coverage."""
    harness = {Path(__file__).stem, "conftest"}
    on_disk = {p.stem for p in EXAMPLES_DIR.glob("*.py")} - harness
    known = set(EXAMPLES) | SELF_DRIVEN | set(STOCHASTIC)
    assert (
        on_disk == known
    ), f"unregistered: {sorted(on_disk - known)}, missing from disk: {sorted(known - on_disk)}"


# Examples are ordered by what a reader needs first, not by which part of PySR they
# exercise: the core of a search, then the knobs that shape one, then value types and
# the searches built on them. Reordering is a deliberate act, so this list, the index
# page, and the VitePress sidebar move together.
READING_ORDER = [
    "getting-started",
    "expression-specifications",
    "objectives",
    "physics",
    "search-behaviour",
    "instrumentation",
    "value-types",
    "beyond-numeric-values",
]

DOCS_SRC = EXAMPLES_DIR.parent / "docs" / "src"


def _squash(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def test_docs_examples_follow_one_declared_order():
    """Sidebar, index, and page bodies agree on the reading order, core material first."""
    if not DOCS_SRC.exists():
        pytest.skip("docs tree absent: examples were copied out of the repo")

    config = (DOCS_SRC / ".vitepress" / "config.mts").read_text()
    assert re.findall(r"link: '/examples/([a-z-]+)'", config) == READING_ORDER

    index = (DOCS_SRC / "examples.md").read_text()
    assert (
        re.findall(r"^## \[[^]]+\]\(/examples/([a-z-]+)\)$", index, re.M)
        == READING_ORDER
    )

    groups = dict(zip(READING_ORDER, index.split("\n## ")[1:]))
    for slug in READING_ORDER:
        listed = re.findall(
            rf"^- \[[^]]+\]\(/examples/{slug}#([a-z0-9-]+)\)", groups[slug], re.M
        )
        page = (DOCS_SRC / "examples" / f"{slug}.md").read_text()
        headings = [h for h in re.findall(r"^## (.+)$", page, re.M) if h != "Preamble"]
        assert [_squash(a) for a in listed] == [
            _squash(h) for h in headings
        ], f"{slug}: index lists {listed}, page has {headings}"


def _split_literals(source):
    """Literals written as adjacent parts across lines, which hide their own seams."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Constant) and isinstance(node.value, str)
        ):
            if node.lineno == node.end_lineno:
                continue
            segment = ast.get_source_segment(source, node) or ""
            if not re.match(r"^[fFrRbB]{0,2}(\"\"\"|''')", segment):
                found.append(segment.splitlines()[0].strip())
    return found


def test_no_string_is_split_across_lines():
    """A string spanning lines is triple-quoted, never adjacent parts.

    Implicit concatenation hides whitespace at the seams. These files are Julia source
    a reader copies, where a dropped space silently changes the program.
    """
    offenders = {}
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        split = _split_literals(path.read_text())
        if split:
            offenders[path.name] = split
    for page in sorted((DOCS_SRC / "examples").glob("*.md")):
        for block in re.findall(
            r"^```python\n(.*?)^```", page.read_text(), re.S | re.M
        ):
            try:
                split = _split_literals(block)
            except SyntaxError:
                continue
            if split:
                offenders.setdefault(page.name, []).extend(split)
    assert not offenders, f"strings split across lines: {offenders}"


def _stray_indents(source):
    """Multi-line literals whose body resets to a shallower column than its own opener."""
    lines = source.split("\n")
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.JoinedStr, ast.Constant)):
            continue
        if isinstance(node, ast.Constant) and not isinstance(node.value, str):
            continue
        if node.lineno == node.end_lineno:
            continue
        opener = lines[node.lineno - 1]
        body = [line for line in lines[node.lineno : node.end_lineno] if line.strip()]
        if not body:
            continue
        base = len(opener) - len(opener.lstrip(" "))
        shallowest = min(len(line) - len(line.lstrip(" ")) for line in body)
        if shallowest != base:
            found.append(opener.strip())
    return found


def test_embedded_julia_is_laid_out_like_its_host():
    """A Julia block inside a Python file is indented with the Python around it.

    Julia ignores indentation, so an `end` dragged out to column 0 buys nothing and
    leaves the Python file looking broken at every closer.
    """
    offenders = {}
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        stray = _stray_indents(path.read_text())
        if stray:
            offenders[path.name] = stray
    if DOCS_SRC.exists():
        for page in sorted((DOCS_SRC / "examples").glob("*.md")):
            for block in re.findall(
                r"^```python\n(.*?)^```", page.read_text(), re.S | re.M
            ):
                try:
                    stray = _stray_indents(block)
                except SyntaxError:
                    continue
                if stray:
                    offenders.setdefault(page.name, []).extend(stray)
    assert not offenders, f"literals dedented below their opener: {offenders}"
