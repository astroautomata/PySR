#!/usr/bin/env python3
"""
Generate API documentation by processing mkdocstrings directives.
This script processes ::: directives in _api.md files and generates api.md files
with rendered API documentation.
"""

import importlib
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Optional

from docstring_parser import parse as parse_docstring

# Add the main PySR directory to sys.path to import pysr
# Find the repo root by looking for pyproject.toml
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir
while repo_root.parent != repo_root:
    if (repo_root / "pyproject.toml").exists():
        break
    repo_root = repo_root.parent
else:
    # If we didn't find pyproject.toml, assume we're in worktrees/vitepress-docs/docs
    repo_root = script_dir.parent.parent.parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def get_object_from_path(path: str) -> Any:
    """Import and return the Python object at the given path."""
    parts = path.split(".")

    # Try importing progressively to handle class.method paths
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_path)
            # Get the remaining attributes from the module
            obj = module
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except (ImportError, AttributeError):
            continue

    raise ImportError(f"Could not import {path}")


def format_signature(obj: Any) -> str:
    """Get the signature of a function/method."""
    try:
        sig = inspect.signature(obj)
        return f"`{obj.__name__}{sig}`"
    except (ValueError, TypeError):
        return f"`{obj.__name__}(...)`"


def format_docstring(docstring: Optional[str], obj: Any = None) -> str:
    """Format a parsed docstring as markdown with nice tables."""
    if not docstring:
        return "*No documentation available.*"

    # Parse the docstring
    parsed = parse_docstring(docstring)
    parts = []

    # Extract parameter defaults from signature if available
    param_defaults = {}
    if obj and callable(obj):
        try:
            sig = inspect.signature(obj)
            for param_name, param_obj in sig.parameters.items():
                if param_obj.default != inspect.Parameter.empty:
                    param_defaults[param_name] = param_obj.default
        except (ValueError, TypeError):
            pass

    # Add short description
    if parsed.short_description:
        parts.append(parsed.short_description)
        parts.append("")

    # Add long description
    if parsed.long_description:
        # Post-process to fix Examples section formatting
        # Replace markdown headings like "# Examples:" with bold text for consistency
        long_desc = parsed.long_description
        long_desc = re.sub(
            r"^# Examples?:\s*$", r"**Examples**", long_desc, flags=re.MULTILINE
        )

        # Dedent the Examples section content (4 spaces -> 0 spaces)
        # This prevents the content from being treated as a code block
        lines = long_desc.split("\n")
        result_lines = []
        in_examples = False
        for line in lines:
            if line.strip() == "**Examples**":
                in_examples = True
                result_lines.append(line)
                result_lines.append("")  # Add blank line after Examples heading
            elif in_examples and line.startswith("    "):
                # Remove 4-space indentation
                result_lines.append(line[4:])
            else:
                result_lines.append(line)

        parts.append("\n".join(result_lines))
        parts.append("")

    # Add parameters as a table
    if parsed.params:
        parts.append("**Parameters**")
        parts.append("")
        parts.append("| Name | Type | Description | Default |")
        parts.append("|------|------|-------------|---------|")

        for param in parsed.params:
            name = f"`{param.arg_name}`" if param.arg_name else ""
            escaped_type = (
                param.type_name.replace("|", r"\|") if param.type_name else None
            )
            type_str = f"`{escaped_type}`" if escaped_type else ""
            desc = (
                param.description.replace("\n", " ").replace("|", "\\|")
                if param.description
                else ""
            )

            # Remove "Default is X" or "Default: X" from description since we have a Default column
            desc = re.sub(
                r"\s*Default is `[^`]+`\.?\s*$", "", desc, flags=re.IGNORECASE
            )
            desc = re.sub(
                r"\s*Default:\s*`[^`]+`\.?\s*$", "", desc, flags=re.IGNORECASE
            )

            # Get default from signature
            if param.arg_name in param_defaults:
                default_val = param_defaults[param.arg_name]
                if default_val is None:
                    default = "`None`"
                elif isinstance(default_val, str):
                    escaped_default = repr(default_val).replace("|", r"\|")
                    default = f"`{escaped_default}`"
                else:
                    escaped_default = str(default_val).replace("|", r"\|")
                    default = f"`{escaped_default}`"
            else:
                default = "*required*"

            parts.append(f"| {name} | {type_str} | {desc} | {default} |")

        parts.append("")

    # Add returns section
    if parsed.returns:
        parts.append("**Returns**")
        parts.append("")
        if parsed.returns.type_name:
            escaped_return_type = parsed.returns.type_name.replace("|", r"\|")
            parts.append(f"- **Type:** `{escaped_return_type}`")
        if parsed.returns.description:
            escaped_return_desc = parsed.returns.description.replace("|", r"\|")
            parts.append(f"- {escaped_return_desc}")
        parts.append("")

    # Add raises section
    if parsed.raises:
        parts.append("**Raises**")
        parts.append("")
        for exc in parsed.raises:
            escaped_exc_type = (
                exc.type_name.replace("|", r"\|") if exc.type_name else None
            )
            exc_type = f"`{escaped_exc_type}`" if escaped_exc_type else "Exception"
            exc_desc = exc.description.replace("|", r"\|") if exc.description else ""
            parts.append(f"- **{exc_type}**: {exc_desc}")
        parts.append("")

    return "\n".join(parts).strip()


def render_api_doc(obj_path: str, options: dict) -> str:
    """Render API documentation for a Python object."""
    try:
        obj = get_object_from_path(obj_path)
    except Exception as e:
        return f"*Error importing {obj_path}: {e}*"

    # Get heading level
    heading_level = options.get("heading_level", 3)
    heading = "#" * heading_level

    # Check if we should document specific members
    members = options.get("members", [])

    if members:
        # Document each member separately
        parts = []
        for member_name in members:
            try:
                member_obj = getattr(obj, member_name)
                member_path = f"{obj_path}.{member_name}"

                # Member heading
                parts.append(f"{heading} {member_name}")
                parts.append("")

                # Add signature if it's callable
                if callable(member_obj):
                    parts.append(format_signature(member_obj))
                    parts.append("")

                # Add docstring
                docstring = inspect.getdoc(member_obj)
                if docstring:
                    parts.append(format_docstring(docstring, member_obj))
                else:
                    parts.append("*No documentation available.*")

                parts.append("")
            except AttributeError:
                parts.append(f"{heading} {member_name}")
                parts.append("")
                parts.append(f"*Error: {member_name} not found in {obj_path}*")
                parts.append("")

        return "\n".join(parts).strip()
    else:
        # Document the object itself
        # Get name to display
        if options.get("show_root_full_path", True):
            name = obj_path
        else:
            name = obj_path.split(".")[-1]

        # Build the documentation
        parts = []

        if options.get("show_root_heading", True):
            parts.append(f"{heading} {name}")
            parts.append("")

        # Add signature if it's a callable
        if callable(obj):
            parts.append(format_signature(obj))
            parts.append("")

        # Add docstring
        docstring = inspect.getdoc(obj)
        if docstring:
            parts.append(format_docstring(docstring, obj))
        else:
            parts.append("*No documentation available.*")

        return "\n".join(parts)


def process_directive(match: re.Match) -> str:
    """Process a single ::: directive."""
    obj_path = match.group(1)
    options_text = match.group(2) or ""

    # Parse options (simple YAML-like parsing)
    options = {}
    current_key = None
    current_list = []

    # Normalize indentation by removing common leading whitespace
    lines = options_text.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]
    if non_empty_lines:
        min_indent = min(len(l) - len(l.lstrip()) for l in non_empty_lines)
        normalized_lines = [
            (l[min_indent:] if len(l) > min_indent else l) for l in lines
        ]
    else:
        normalized_lines = lines

    for line in normalized_lines:
        if not line.strip():
            continue

        # Check indentation level (relative to normalized base)
        indent = len(line) - len(line.lstrip())
        line = line.strip()

        if indent == 0 and ":" in line:
            # Save previous list if exists
            if current_key and current_list:
                options[current_key] = current_list
                current_list = []

            # New key
            key, value = line.split(":", 1)
            current_key = key.strip()
            value = value.strip()

            if value:
                # Inline value
                if value.lower() == "true":
                    options[current_key] = True
                elif value.lower() == "false":
                    options[current_key] = False
                elif value.isdigit():
                    options[current_key] = int(value)
                else:
                    options[current_key] = value
                current_key = None
        elif line.startswith("- ") and current_key:
            # List item
            current_list.append(line[2:].strip())

    # Save final list if exists
    if current_key and current_list:
        options[current_key] = current_list

    return render_api_doc(obj_path, options)


def process_markdown_file(source_path: Path, output_path: Path) -> None:
    """Process a markdown file, replacing ::: directives with rendered docs."""
    content = source_path.read_text()

    # Pattern to match ::: directives with optional multiline options block
    # Matches:
    # ::: module.path
    #     options:
    #         key: value
    #         list_key:
    #             - item1
    #             - item2
    pattern = r"^::: ([\w.]+)\n(?:    options:\n((?:(?:        [\w_]+:.*\n)(?:            - .+\n)*)+))?"

    # Replace all directives
    new_content = re.sub(pattern, process_directive, content, flags=re.MULTILINE)

    # Write the output
    output_path.write_text(new_content)
    print(f"Generated {output_path} from {source_path}")


def process_readme_to_index(readme_path: Path, output_path: Path) -> None:
    """Convert README.md to index.md with VitePress syntax."""
    content = readme_path.read_text()

    # Add the video splash at the top
    video_splash = '<div style="width:100%;height:0px;position:relative;padding-bottom:56.250%;"><iframe src="https://streamable.com/e/ncvqhy" frameborder="0" width="100%" height="100%" allowfullscreen style="width:100%;height:100%;position:absolute;left:0px;top:0px;overflow:hidden;"></iframe></div>\n'
    content = video_splash + content

    # Convert <details> blocks to VitePress ::: details syntax
    # Pattern matches: <details><summary>\n\n### Title\n\n</summary>\ncontent\n</details>
    def replace_details(match):
        summary = match.group(1).strip()
        body = match.group(2).strip()
        # Extract title from summary (remove ### prefix if present)
        title = summary.replace("###", "").strip()
        return f"::: details {title}\n\n{body}\n\n:::"

    # Match <details><summary>...</summary>...</details> blocks
    details_pattern = r"<details>\s*<summary>\s*(.*?)\s*</summary>\s*(.*?)\s*</details>"
    content = re.sub(details_pattern, replace_details, content, flags=re.DOTALL)

    output_path.write_text(content)
    print(f"Generated {output_path} from {readme_path}")


def main():
    """Main entry point."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent.resolve()
    src_dir = script_dir / "src"

    # Generate index.md from README.md
    readme_path = script_dir.parent / "README.md"
    index_path = src_dir / "index.md"
    if readme_path.exists():
        try:
            process_readme_to_index(readme_path, index_path)
        except Exception as e:
            print(f"Error generating index.md: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Process _api.md → api.md and _api-advanced.md → api-advanced.md
    for source_name, output_name in [
        ("_api.md", "api.md"),
        ("_api-advanced.md", "api-advanced.md"),
    ]:
        source_path = src_dir / source_name
        output_path = src_dir / output_name

        if source_path.exists():
            try:
                process_markdown_file(source_path, output_path)
            except Exception as e:
                print(f"Error processing {source_path}: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                sys.exit(1)
        else:
            print(f"Warning: {source_path} not found", file=sys.stderr)

    print("Documentation generation complete!")


if __name__ == "__main__":
    main()
