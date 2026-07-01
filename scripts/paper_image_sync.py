#!/usr/bin/env python3
"""CI helper for paper PRs: upload/normalize images and comment back with URLs.

Runs in pull_request_target; never executes PR-provided code.
"""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any

import requests
import yaml
from PIL import Image, ImageOps

GITHUB_API = "https://api.github.com"

# Unique marker used to identify the paper-image bot comment for updating.
COMMENT_MARKER = "<!-- pysr-paper-image-bot -->"


@dataclass
class PRInfo:
    number: int
    base_repo: str
    head_repo: str
    head_ref: str
    head_sha: str
    is_fork: bool


def env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if v is None or v == "":
        raise SystemExit(f"Missing required env var: {name}")
    return v


def gh_request(method: str, url: str, token: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    resp = requests.request(method, url, headers=headers, timeout=60, **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"GitHub API error {resp.status_code} for {url}: {resp.text[:500]}"
        )
    return resp


def get_pr_info(repo: str, pr_number: int, token: str) -> PRInfo:
    pr = gh_request("GET", f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}", token).json()
    base_repo = pr["base"]["repo"]["full_name"]
    head_repo = pr["head"]["repo"]["full_name"]
    head_ref = pr["head"]["ref"]
    head_sha = pr["head"]["sha"]
    is_fork = base_repo.lower() != head_repo.lower()
    return PRInfo(
        number=pr_number,
        base_repo=base_repo,
        head_repo=head_repo,
        head_ref=head_ref,
        head_sha=head_sha,
        is_fork=is_fork,
    )


def list_pr_files(repo: str, pr_number: int, token: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = gh_request(
            "GET",
            f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files",
            token,
            params={"per_page": 100, "page": page},
        )
        batch = resp.json()
        if not batch:
            break
        files.extend(batch)
        page += 1
    return files


def pr_has_existing_bot_comment(repo: str, pr_number: int, token: str) -> bool:
    """Return True if the paper-image bot has already commented on this PR.

    This is used to avoid posting a confusing warning when a user has already
    synced images to the docs repo and then deletes the local image files from
    their PR (leaving only a `docs/papers.yml` edit pointing to URLs).
    """

    page = 1
    while True:
        resp = gh_request(
            "GET",
            f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
            token,
            params={"per_page": 100, "page": page},
        )
        comments = resp.json()
        if not comments:
            return False

        for c in comments:
            body = c.get("body") or ""
            if COMMENT_MARKER in body:
                return True

        page += 1


def get_file_bytes_at_ref(repo: str, path: str, ref: str, token: str) -> bytes:
    resp = gh_request(
        "GET",
        f"{GITHUB_API}/repos/{repo}/contents/{path}",
        token,
        params={"ref": ref},
    ).json()
    if resp.get("encoding") == "base64":
        return base64.b64decode(resp["content"])
    # Fallback for large files (may return download_url):
    if "download_url" in resp and resp["download_url"]:
        dl = requests.get(resp["download_url"], timeout=60)
        dl.raise_for_status()
        return dl.content
    raise RuntimeError(f"Unable to fetch bytes for {repo}:{path}@{ref}")


def resize_compress_image(
    src: bytes,
    *,
    max_width: int,
    min_width: int,
    png_compress_level: int,
    max_bytes: int,
    quantize_colors: int,
    input_path: str,
) -> tuple[bytes, str]:
    """Return (processed_bytes, output_ext).

    Policy:
    - Always output PNG (lossless) and strip alpha.
    - Any transparency is flattened onto a white background.

    Best-effort size control:
    1) Try PNG compress_level up to 9 (still lossless).
    2) If still too large and enabled, try adaptive palette quantization.
    3) If still too large, iteratively downscale (down to min_width).

    Note: Quantization is technically lossless for flat-color plots/diagrams, but can
    introduce banding for photographic/gradient-heavy images.
    """

    def save_png(im: Image.Image, *, level: int) -> bytes:
        out = io.BytesIO()
        # PNG is lossless; compress_level affects size/time only (0-9).
        im.save(out, format="PNG", optimize=True, compress_level=level)
        return out.getvalue()

    def best_lossless(im: Image.Image) -> bytes:
        best: bytes | None = None
        for lvl in range(max(0, png_compress_level), 10):
            b = save_png(im, level=lvl)
            if best is None or len(b) < len(best):
                best = b
            if len(b) <= max_bytes:
                return b
        assert best is not None
        return best

    def best_quantized(im: Image.Image) -> bytes | None:
        if quantize_colors <= 0:
            return None
        best: bytes | None = None
        # Try a small ladder of palettes (keeps crisp edges for typical plots).
        for colors in [quantize_colors, 128, 64, 32]:
            if not (1 <= colors <= 256):
                continue
            q = im.quantize(
                colors=colors,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            # Keep as paletted PNG (mode "P") for best size.
            b = save_png(q, level=9)
            if best is None or len(b) < len(best):
                best = b
            if len(b) <= max_bytes:
                return b
        return best

    with Image.open(io.BytesIO(src)) as im:
        im = ImageOps.exif_transpose(im)

        # Resize down if needed
        if im.width > max_width:
            new_h = round(im.height * (max_width / im.width))
            im = im.resize((max_width, new_h), Image.Resampling.LANCZOS)

        # Normalize to RGBA so we can consistently flatten alpha if present.
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        im_rgb = bg

        # 1) lossless compression ladder
        best = best_lossless(im_rgb)
        if len(best) <= max_bytes:
            return best, ".png"

        # 2) quantize (optional)
        qbest = best_quantized(im_rgb)
        if qbest is not None and len(qbest) < len(best):
            best = qbest
        if len(best) <= max_bytes:
            return best, ".png"

        # 3) downscale until we hit max_bytes or min_width
        cur = im_rgb
        while cur.width > max(min_width, 1):
            new_w = max(max(min_width, 1), int(cur.width * 0.9))
            if new_w >= cur.width:
                break
            new_h = round(cur.height * (new_w / cur.width))
            cur = cur.resize((new_w, new_h), Image.Resampling.LANCZOS)

            cand = best_lossless(cur)
            if len(cand) <= max_bytes:
                return cand, ".png"

            qcand = best_quantized(cur)
            if qcand is not None and len(qcand) < len(cand):
                cand = qcand
            if len(cand) < len(best):
                best = cand
            if len(best) <= max_bytes:
                return best, ".png"

        return best, ".png"


def create_or_update_file(
    *, repo: str, path: str, branch: str, message: str, content: bytes, token: str
) -> str:
    """Create or update a file in repo on branch.

    Returns the commit SHA created by the contents API.
    """

    # Check existing file to get sha (for updates)
    sha: str | None = None
    try:
        existing = gh_request(
            "GET",
            f"{GITHUB_API}/repos/{repo}/contents/{path}",
            token,
            params={"ref": branch},
        ).json()
        sha = existing.get("sha")
    except Exception:
        sha = None

    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    resp = gh_request(
        "PUT",
        f"{GITHUB_API}/repos/{repo}/contents/{path}",
        token,
        json=payload,
    ).json()

    commit_sha = (resp.get("commit") or {}).get("sha")
    if not commit_sha:
        raise RuntimeError(f"Failed to get commit SHA after updating {repo}:{path}")
    return commit_sha


def ensure_branch(repo: str, branch: str, base_branch: str, token: str) -> None:
    """Ensure branch exists in repo, creating from base_branch if missing."""

    # Does branch exist?
    r = requests.get(
        f"{GITHUB_API}/repos/{repo}/git/ref/heads/{branch}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=60,
    )
    if r.status_code == 200:
        return

    # Get base sha
    base = gh_request(
        "GET", f"{GITHUB_API}/repos/{repo}/git/ref/heads/{base_branch}", token
    ).json()
    base_sha = base["object"]["sha"]

    gh_request(
        "POST",
        f"{GITHUB_API}/repos/{repo}/git/refs",
        token,
        json={"ref": f"refs/heads/{branch}", "sha": base_sha},
    )


def open_pr(
    *, repo: str, head: str, base: str, title: str, body: str, token: str
) -> str:
    """Open a PR. If one already exists for (head->base), return its URL."""

    try:
        resp = gh_request(
            "POST",
            f"{GITHUB_API}/repos/{repo}/pulls",
            token,
            json={
                "title": title,
                "head": head,
                "base": base,
                "body": body,
                "maintainer_can_modify": True,
            },
        ).json()
        return resp["html_url"]
    except Exception:
        # Try to find existing.
        prs = gh_request(
            "GET",
            f"{GITHUB_API}/repos/{repo}/pulls",
            token,
            params={"state": "open", "per_page": 100},
        ).json()
        for pr in prs:
            if (
                pr.get("head", {}).get("ref") == head.split(":")[-1]
                and pr.get("base", {}).get("ref") == base
            ):
                return pr["html_url"]
        raise


def comment_on_pr(repo: str, pr_number: int, token: str, body: str) -> None:
    # Prefix a marker so the workflow can find and update this comment reliably.
    if COMMENT_MARKER not in body:
        body = f"{COMMENT_MARKER}\n{body}"

    # If asked, write the comment to a file instead of posting (so the workflow can
    # create-or-update the PR comment without spamming new ones).
    out_path = os.environ.get("COMMENT_BODY_PATH", "").strip()
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(body.rstrip() + "\n")
        return

    gh_request(
        "POST",
        f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments",
        token,
        json={"body": body},
    )


def try_push_cleanup_commit(
    *, pr: PRInfo, images_to_delete: list[str], papers_yml_new: str | None
) -> str | None:
    """If PR is from same repo, push a cleanup commit to the PR branch.

    Returns commit SHA if pushed, else None.
    """

    if pr.is_fork:
        return None

    # Fetch and checkout the PR branch.
    subprocess.check_call(["git", "config", "user.name", "PySR Paper Bot"])
    subprocess.check_call(["git", "config", "user.email", "actions@github.com"])

    subprocess.check_call(["git", "fetch", "origin", f"{pr.head_ref}:{pr.head_ref}"])
    subprocess.check_call(["git", "checkout", pr.head_ref])

    changed = False

    for p in images_to_delete:
        if os.path.exists(p):
            os.remove(p)
            changed = True

    if papers_yml_new is not None:
        with open("docs/papers.yml", "w", encoding="utf-8") as f:
            f.write(papers_yml_new)
        changed = True

    if not changed:
        return None

    subprocess.check_call(["git", "add", "docs/papers.yml"], stdout=subprocess.DEVNULL)
    for p in images_to_delete:
        if os.path.exists(p):
            subprocess.check_call(["git", "add", "-u", p], stdout=subprocess.DEVNULL)
        else:
            # add -u will record deletion if it existed.
            subprocess.check_call(["git", "add", "-u"], stdout=subprocess.DEVNULL)

    subprocess.check_call(
        [
            "git",
            "commit",
            "-m",
            "chore(papers): move PR images to docs repo and remove from PySR",
        ]
    )
    subprocess.check_call(["git", "push", "origin", f"HEAD:{pr.head_ref}"])

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    return sha


def main() -> None:
    gh_token = env("GH_TOKEN")
    repo = env("REPO")
    pr_number = int(env("PR_NUMBER"))

    docs_repo = env("PYSR_DOCS_REPO")
    docs_token = env("PYSR_DOCS_TOKEN")
    docs_images_dir = env("PYSR_DOCS_IMAGES_DIR")

    max_width = int(env("MAX_WIDTH", "1200"))
    min_width = int(env("MIN_WIDTH", "800"))
    png_compress_level = int(env("PNG_COMPRESS_LEVEL", "6"))

    # Best-effort target size (approximate; PNG is lossless so some images may remain larger).
    # TARGET_KB takes precedence if set (more ergonomic in workflow files).
    target_kb = os.environ.get("TARGET_KB", "")
    if target_kb:
        max_bytes = int(float(target_kb) * 1000)
    else:
        max_bytes = int(env("MAX_BYTES", "300000"))

    # Adaptive palette quantization (0 disables). Helps a lot for plots/diagrams.
    quantize_colors = int(env("QUANTIZE_COLORS", "256"))

    pr = get_pr_info(repo, pr_number, gh_token)
    files = list_pr_files(repo, pr_number, gh_token)

    # Identify added images (by PR file list) within PySR.
    image_re = re.compile(r"\.(png|jpe?g|webp)$", re.IGNORECASE)
    candidate_images: list[str] = []
    touched_papers_yml = False

    for f in files:
        filename = f["filename"]
        status = f["status"]
        if filename == "docs/papers.yml":
            touched_papers_yml = True
        if (
            filename.startswith("docs/src/public/images/")
            and image_re.search(filename)
            and status in {"added", "modified"}
        ):
            candidate_images.append(filename)

    if not touched_papers_yml and not candidate_images:
        # Nothing to do.
        return

    if not candidate_images:
        # Common expected case: images were synced in a prior run and the author
        # removed the local image files from their PR (keeping only URL updates in
        # `docs/papers.yml`). In that case, don't post a confusing warning.
        if not pr_has_existing_bot_comment(repo, pr_number, gh_token):
            comment_on_pr(
                repo,
                pr_number,
                gh_token,
                "Paper-image bot: `docs/papers.yml` changed but no images were added under `docs/src/public/images/`. "
                "If your entry references a local image filename, please add it (or use an absolute `http(s)` URL).",
            )
        return

    # Download + process images from PR head sha.
    processed: list[tuple[str, bytes]] = []  # (basename, bytes)
    for path in candidate_images:
        raw = get_file_bytes_at_ref(pr.head_repo, path, pr.head_sha, gh_token)
        out_bytes, out_ext = resize_compress_image(
            raw,
            max_width=max_width,
            min_width=min_width,
            png_compress_level=png_compress_level,
            max_bytes=max_bytes,
            quantize_colors=quantize_colors,
            input_path=path,
        )
        base = os.path.basename(path)
        # If our compressor changed ext (e.g., jpeg normalization), reflect that.
        if out_ext != os.path.splitext(base)[1].lower():
            base = os.path.splitext(base)[0] + out_ext
        processed.append((base, out_bytes))

    # Create docs repo branch and commit files via contents API.
    branch = f"paper-images/pr-{pr_number}"

    # Respect the docs repo's default branch (some repos still use `master`).
    docs_default_branch = (
        gh_request("GET", f"{GITHUB_API}/repos/{docs_repo}", docs_token)
        .json()
        .get("default_branch", "main")
    )

    ensure_branch(docs_repo, branch, docs_default_branch, docs_token)

    dst_paths: dict[str, str] = {}
    # Use a commit-SHA raw URL rather than a branch URL so that:
    # - CI validation can accept it immediately, and
    # - the link remains stable regardless of whether the branch is deleted/renamed.
    docs_images_commit: str | None = None
    for filename, content in processed:
        dst_path = f"{docs_images_dir}/{filename}".lstrip("/")
        dst_paths[filename] = dst_path
        docs_images_commit = create_or_update_file(
            repo=docs_repo,
            path=dst_path,
            branch=branch,
            message=f"Add paper image {filename} (from PySR PR #{pr_number})",
            content=content,
            token=docs_token,
        )

    docs_pr_url = open_pr(
        repo=docs_repo,
        head=branch,
        base=docs_default_branch,
        title=f"Add paper images from PySR PR #{pr_number}",
        body=(
            f"Automated upload of paper image(s) from MilesCranmer/PySR PR #{pr_number}.\n\n"
            f"Source head: {pr.head_repo}@{pr.head_sha}\n"
            f"Target dir: `{docs_images_dir}`\n"
        ),
        token=docs_token,
    )

    # Construct absolute image URLs to be used in PySR docs.
    # Use a commit SHA rather than a branch name so the URL is stable and passes
    # docs/papers.yml validation immediately.
    assert docs_images_commit is not None
    urls = {
        name: f"https://raw.githubusercontent.com/{docs_repo}/{docs_images_commit}/{dst_paths[name]}"
        for name, _ in processed
    }
    stem_to_uploaded_name = {os.path.splitext(name)[0]: name for name, _ in processed}

    # If we can read papers.yml from PR head, propose edits.
    papers_yml_new: str | None = None
    try:
        papers_yml_bytes = get_file_bytes_at_ref(
            pr.head_repo, "docs/papers.yml", pr.head_sha, gh_token
        )
        papers_obj = yaml.safe_load(papers_yml_bytes.decode("utf-8"))

        # Replace any image fields that match basenames we processed (or original names).
        changed = False
        for paper in papers_obj.get("papers", []):
            img = paper.get("image")
            if not isinstance(img, str):
                continue
            # If PR referenced local filename and we uploaded it, replace with url.
            base = os.path.basename(img)
            if img.startswith("http"):
                continue

            if base in urls:
                paper["image"] = urls[base]
                changed = True
                continue

            # If the PR referenced e.g. foo.jpg but we normalized to foo.png,
            # still update the reference.
            stem = os.path.splitext(base)[0]
            if stem in stem_to_uploaded_name:
                paper["image"] = urls[stem_to_uploaded_name[stem]]
                changed = True

        if changed:
            papers_yml_new = (
                "# NOTE: This file was updated by the paper-image bot.\n"
                + yaml.safe_dump(papers_obj, sort_keys=False, allow_unicode=True)
            )
    except Exception:
        papers_yml_new = None

    pushed_sha = None
    if papers_yml_new is not None:
        pushed_sha = try_push_cleanup_commit(
            pr=pr, images_to_delete=candidate_images, papers_yml_new=papers_yml_new
        )

    # Comment on the PR with next steps.
    lines = [
        "Paper-image bot results:",
        "",
        f"- Opened docs PR: {docs_pr_url}",
        "- Uploaded/resized images:",
    ]
    for name in urls:
        lines.append(f"  - `{name}` → {urls[name]}")

    if pushed_sha:
        lines += [
            "",
            f"- Pushed a cleanup commit to this PR branch: `{pushed_sha}` (deleted local images + updated `docs/papers.yml` to use absolute URLs).",
        ]
    else:
        lines += [
            "",
            "I could not push changes back to the PR branch (likely a fork).",
            "Maintainer options:",
            "1) Merge the docs PR above, then update `docs/papers.yml` in this PR to set `image:` to the URL(s) above.",
            "2) Remove the image file(s) from this PR (they don't need to live in PySR once hosted in PySR_Docs).",
        ]

    comment_on_pr(repo, pr_number, gh_token, "\n".join(lines))


if __name__ == "__main__":
    main()
