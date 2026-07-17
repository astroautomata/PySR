# This dockerfile simulates a user installation that
# tries to manually edit SymbolicRegression.jl and
# use it from PySR.

ARG JLVERSION=1.11
ARG PYVERSION=3.12
ARG BASE_IMAGE=bullseye

FROM julia:${JLVERSION}-${BASE_IMAGE} AS jl
FROM python:${PYVERSION}-${BASE_IMAGE}

# Merge Julia image:
COPY --from=jl /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

WORKDIR /pysr

# Install PySR:
# We do a minimal copy so it doesn't need to rerun at every file change:
ADD ./pyproject.toml /pysr/pyproject.toml
ADD ./LICENSE /pysr/LICENSE
ADD ./README.md /pysr/README.md

RUN mkdir /pysr/pysr
ADD ./pysr/*.py /pysr/pysr/
ADD ./pysr/juliapkg.json /pysr/pysr/juliapkg.json

RUN mkdir /pysr/pysr/_cli
ADD ./pysr/_cli/*.py /pysr/pysr/_cli/

RUN mkdir /pysr/pysr/test

# Now, we create a custom version of SymbolicRegression.jl
# First, we get the version or rev from juliapkg.json:
RUN python3 -c 'import json; pkg = json.load(open("/pysr/pysr/juliapkg.json", "r"))["packages"]["SymbolicRegression"]; print(pkg.get("version", pkg.get("rev", "")))' > /pysr/sr_version

# Remove any = or ^ or ~ from the version:
RUN cat /pysr/sr_version | sed 's/[\^=~]//g' > /pysr/sr_version_processed

# Now, we check out the version of SymbolicRegression.jl that PySR is using:
# If sr_version starts with 'v', use it as-is; otherwise prepend 'v'
RUN if grep -q '^v' /pysr/sr_version_processed; then \
        git clone -b "$(cat /pysr/sr_version_processed)" --single-branch https://github.com/astroautomata/SymbolicRegression.jl /srjl; \
    else \
        git clone -b "v$(cat /pysr/sr_version_processed)" --single-branch https://github.com/astroautomata/SymbolicRegression.jl /srjl; \
    fi

# Edit SymbolicRegression.jl to create a new function.
# We want to put this function immediately after `module SymbolicRegression`:
RUN sed -i 's/module SymbolicRegression/module SymbolicRegression\n__test_function() = 2.3/' /srjl/src/SymbolicRegression.jl

# Edit PySR to use the custom version of SymbolicRegression.jl:
ADD ./pysr/test/generate_dev_juliapkg.py /generate_dev_juliapkg.py
RUN python3 /generate_dev_juliapkg.py /pysr/pysr/juliapkg.json /srjl

# Install and pre-compile
RUN pip3 install --no-cache-dir . && python3 -c 'import pysr'
