"""This script generates the papers.md file from the papers.yml file."""

from pathlib import Path

import yaml

data_file = "papers.yml"
papers_header = Path("stylesheets") / "papers_header.txt"
output_file = Path("src") / "papers.md"

# Load YAML file:
with open(data_file, "r") as stream:
    papers = yaml.load(stream, Loader=yaml.SafeLoader)["papers"]

# Load header:
with open(papers_header, "r") as stream:
    header = stream.read()

with open(output_file, "w") as f:
    f.write(header)

    # First, we sort the papers by date.
    # This is in the format of "2022-03-15"
    papers = sorted(papers, key=lambda paper: paper["date"], reverse=True)

    snippets = []
    for paper in papers:
        title = paper["title"]
        authors = (
            ", ".join(paper["authors"]).replace("(", "<sup>").replace(")", "</sup>")
        )
        affiliations = ", ".join(
            f"<sup>{num}</sup>{affil}" for num, affil in paper["affiliations"].items()
        )
        link = paper["link"]
        abstract = paper["abstract"]
        image_file = paper["image"]

        if image_file.startswith("http"):
            absolute_image_file = image_file
        else:
            absolute_image_file = f"/images/{image_file}"

        # Begin:
        paper_snippet = f"""

<div style="text-align: center; margin: 2rem 0;">
  <img src="{absolute_image_file}" alt="{title}" style="max-width: 500px; width: 100%; height: auto; margin: 0 auto; display: block;">
  <h2 style="margin-top: 1rem;">
    <a href="{link}">{title}</a>
  </h2>
  <div style="margin: 1rem 0;">
    {authors}
  </div>
  <div style="font-size: 0.9em; color: #666;">
    {affiliations}
  </div>
</div>

**Abstract:** {abstract}\n\n
"""
        snippets.append(paper_snippet)

    f.write("\n\n---\n\n".join(snippets))
