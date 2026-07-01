# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.
#
# Targets:
# - pysr-runtime: Python + Julia + PySR installed (no eager Julia package precompile).
# - pysr: dev-friendly image (installs extras and precompiles Julia deps).
# - pysr-slurm: image for running a local Slurm cluster for CI tests.

ARG JLVERSION=1.11
ARG PYVERSION=3.12
ARG BASE_IMAGE=bullseye

FROM julia:${JLVERSION}-${BASE_IMAGE} AS jl
FROM python:${PYVERSION}-${BASE_IMAGE} AS pysr-runtime

# Merge Julia image:
COPY --from=jl /usr/local/julia /usr/local/julia
ENV PATH="/usr/local/julia/bin:${PATH}"

WORKDIR /pysr

# Install PySR:
# We do a minimal copy so it doesn't need to rerun at every file change:
ADD ./pyproject.toml /pysr/pyproject.toml
ADD ./LICENSE /pysr/LICENSE
ADD ./README.md /pysr/README.md
ADD ./pysr /pysr/pysr
RUN pip3 install --no-cache-dir .

# --------------------------------------------------------------------
FROM pysr-runtime AS pysr-slurm

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      gettext-base \
      munge \
      procps \
      slurmctld \
      slurmd \
      util-linux \
    ; \
    rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    chmod 0755 /etc; \
    mkdir -m 0755 -p \
      /var/run/slurm \
      /var/spool/slurm \
      /var/lib/slurm \
      /var/log/slurm \
      /etc/slurm \
    ; \
    mkdir -m 0700 -p /etc/munge; \
    if [ ! -f /etc/munge/munge.key ]; then \
      dd if=/dev/urandom of=/etc/munge/munge.key bs=1 count=1024; \
      chmod 0400 /etc/munge/munge.key; \
      chown munge:munge /etc/munge/munge.key; \
    fi; \
    chown -R slurm:slurm \
      /var/run/slurm \
      /var/spool/slurm \
      /var/lib/slurm \
      /var/log/slurm \
      /etc/slurm

COPY pysr/test/slurm_docker_cluster/config/slurm.conf /etc/slurm/slurm.conf
COPY pysr/test/slurm_docker_cluster/config/cgroup.conf /etc/slurm/cgroup.conf
COPY pysr/test/slurm_docker_cluster/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]

# --------------------------------------------------------------------
FROM pysr-runtime AS pysr

# Install IPython and other useful libraries:
RUN pip install --no-cache-dir ipython matplotlib

# Install Julia pre-requisites:
RUN python3 -c 'import pysr; pysr.load_all_packages()'

CMD ["ipython"]
