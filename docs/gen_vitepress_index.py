#!/usr/bin/env python3
"""
Generate VitePress-compatible index.md by combining _index.md template with README.md content.
"""

from pathlib import Path


def process_readme_content(readme_content):
    """
    Extract main content from README.md for VitePress.

    Skips GitHub-specific elements (logo, title) and starts from first ## heading.
    """
    lines = readme_content.split("\n")
    content_lines = []
    found_first_header = False

    for line in lines:
        # Start capturing after we find the first ## header
        if not found_first_header and line.startswith("## "):
            found_first_header = True

        if not found_first_header:
            continue

        # Skip GitHub user-images URLs
        if "user-images" in line:
            continue

        # Adjust header levels: # -> ##
        if line.startswith("# ") and not line.startswith("# PySR"):
            line = "#" + line

        # Transform "### Test status" to "**Test status**"
        if line == "### Test status":
            line = "**Test status**"

        # Fix internal links to use VitePress paths
        line = line.replace("(https://ai.damtp.cam.ac.uk/pysr/papers)", "(/papers)")

        # Skip table of contents entries
        if line.startswith("**Contents**:") or (
            line.startswith("- [")
            and any(
                x in line
                for x in [
                    "Why PySR?",
                    "Installation",
                    "Quickstart",
                    "Documentation",
                    "Contributors",
                ]
            )
        ):
            continue

        content_lines.append(line)

    return "\n".join(content_lines)


def main():
    """Generate index.md from _index.md template and README.md content."""
    script_dir = Path(__file__).parent
    template_path = script_dir / "src" / "_index.md"
    readme_path = script_dir.parent / "README.md"
    output_path = script_dir / "src" / "index.md"

    # Read template
    if not template_path.exists():
        print(f"Error: Template file {template_path} not found!")
        return 1

    template_content = template_path.read_text()

    # Read README
    if not readme_path.exists():
        print(f"Error: README.md not found at {readme_path}!")
        return 1

    readme_content = readme_path.read_text()

    # Process README content
    processed_content = process_readme_content(readme_content)

    # Replace marker in template
    if "<!-- README_CONTENT_MARKER -->" in template_content:
        final_content = template_content.replace(
            "<!-- README_CONTENT_MARKER -->", processed_content
        )
    else:
        # If no marker, append content
        final_content = template_content + "\n" + processed_content

    # Write output
    output_path.write_text(final_content)
    print(f"Generated {output_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
