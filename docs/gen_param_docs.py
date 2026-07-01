#!/usr/bin/env python3
"""
Generate API parameter documentation from param_groupings.yml.
This script reads _api_template.md and substitutes the {params_output}
placeholder with auto-generated parameter docs from param_groupings.yml
and PySRRegressor docstrings.
"""

import re
import sys
from pathlib import Path

from docstring_parser import parse
from yaml import safe_load

sys.path.append("..")

from pysr import PySRRegressor

found_params = []


def str_param_groups(param_groupings, params, cur_heading=2):
    global found_params
    # Recursively print the parameter descriptions, defaults,
    # with headings from the param groupings dict.
    if isinstance(param_groupings, list):
        return "\n\n".join(
            str_param_groups(param, params, cur_heading) for param in param_groupings
        )
    elif isinstance(param_groupings, dict):
        for heading, param_grouping in param_groupings.items():
            return (
                f"{'#' * cur_heading} {heading}"
                + "\n\n"
                + str_param_groups(param_grouping, params, cur_heading + 1)
            )
    elif isinstance(param_groupings, str):
        found_params.append(param_groupings)

        default_value = re.search(
            r"Default is `(.*)`", params[param_groupings].description
        )
        clean_desc = re.sub(r"Default is .*", "", params[param_groupings].description)
        # Prepend every line with 4 spaces:
        clean_desc = "\n".join("    " + line for line in clean_desc.splitlines())
        return (
            f"  - **`{param_groupings}`**"
            + "\n\n"
            + clean_desc
            + (
                "\n\n    " + f"*Default:* `{default_value.group(1)}`"
                if default_value
                else ""
            )
        )
    else:
        raise TypeError(f"Unexpected type {type(param_groupings)}")


if __name__ == "__main__":
    # This is the path to the param_groupings.yml file
    # relative to the current file.
    param_groupings_path = Path(__file__).parent.parent / "pysr" / "param_groupings.yml"
    with open(param_groupings_path, "r") as f:
        param_groupings = safe_load(f)

    # Load parameter descriptions from the docstring of PySRRegressor
    raw_params = parse(PySRRegressor.__doc__).params
    params = {
        param.arg_name: param
        for param in raw_params
        if param.arg_name[-1] != "_" and param.arg_name != "**kwargs"
    }

    # Generate parameter docs
    params_output = str_param_groups(param_groupings, params, cur_heading=3)
    assert (
        len(set(found_params) ^ set(params.keys())) == 0
    ), f"Mismatch between param_groupings.yml and PySRRegressor parameters"

    # Read template and substitute params
    template_path = Path(__file__).parent / "src" / "_api_template.md"
    template = template_path.read_text()
    api_md_content = template.replace("PARAMSKEY", params_output)

    # Write to src/_api.md
    output_path = Path(__file__).parent / "src" / "_api.md"
    output_path.write_text(api_md_content)
    print(f"Generated {output_path}")
