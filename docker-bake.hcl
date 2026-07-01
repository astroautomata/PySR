group "default" {
  targets = ["pysr"]
}

group "slurm" {
  targets = ["pysr-slurm"]
}

group "dev" {
  targets = ["pysr-dev"]
}

target "pysr" {
  dockerfile = "Dockerfile"
  target = "pysr"
  tags = ["pysr:local"]
}

target "pysr-runtime" {
  dockerfile = "Dockerfile"
  target = "pysr-runtime"
  tags = ["pysr-runtime:local"]
}

target "pysr-slurm" {
  dockerfile = "Dockerfile"
  target = "pysr-slurm"
  tags = ["pysr-slurm:local"]
}

target "pysr-dev" {
  context = "."
  dockerfile = "pysr/test/test_dev_pysr.dockerfile"
  tags = ["pysr-dev"]
}
