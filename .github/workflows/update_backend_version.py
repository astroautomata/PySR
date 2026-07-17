import json
import sys
from pathlib import Path

new_backend_version = sys.argv[1]

assert not new_backend_version.startswith("v"), "Version should not start with 'v'"

repo_root = Path(__file__).parent / ".." / ".."
juliapkg_json = repo_root / "pysr" / "juliapkg.json"
with open(juliapkg_json) as f:
    juliapkg_data = json.load(f)

# Update backend, maintain current format (either "rev" or "version")
backend_pkg = juliapkg_data["packages"]["SymbolicRegression"]
if "rev" in backend_pkg:
    backend_pkg["rev"] = f"v{new_backend_version}"
elif "version" in backend_pkg:
    backend_pkg["version"] = f"~{new_backend_version}"
else:
    raise ValueError(
        "SymbolicRegression package must have either 'rev' or 'version' field"
    )

with open(juliapkg_json, "w") as f:
    json.dump(juliapkg_data, f, indent=4)
    f.write("\n")
