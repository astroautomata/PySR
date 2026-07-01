#!/bin/bash
set -euo pipefail

echo "---> Starting MUNGE (munged) ..."
mkdir -p /run/munge
chown munge:munge /run/munge
chmod 0711 /run/munge
runuser -u munge -- /usr/sbin/munged

if [ "${1:-}" = "slurmctld" ]; then
  echo "---> Starting slurmctld ..."
  exec runuser -u slurm -- /usr/sbin/slurmctld -i -D
fi

if [ "${1:-}" = "slurmd" ]; then
  echo "---> Waiting for slurmctld ..."
  until 2>/dev/null >/dev/tcp/slurmctld/6817; do
    echo "-- slurmctld not available; sleeping ..."
    sleep 2
  done
  # Slurm's cgroup/v2 plugin expects this systemd-like slice path.
  mkdir -p /sys/fs/cgroup/system.slice || true
  echo "---> Starting slurmd on $(hostname) ..."
  exec /usr/sbin/slurmd -D
fi

exec "$@"
