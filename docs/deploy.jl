#!/usr/bin/env julia
#
# Build and deploy VitePress documentation to versioned gh-pages directories.
# This script handles:
# - Regular deployments to gh-pages
# - PR preview deployments to gh-pages/previews/PR##/
# - Dual deployment to secondary repository (ai.damtp.cam.ac.uk)
#

using DocumenterVitepress

# Get deployment target from environment (for dual deployment)
deployment_target = get(ENV, "DEPLOYMENT_TARGET", "primary")

println("Starting DocumenterVitepress deployment...")
println("Deployment target: $deployment_target")
println("Event: $(get(ENV, "GITHUB_EVENT_NAME", "unknown"))")
println("Ref: $(get(ENV, "GITHUB_REF", "unknown"))")

# Get deployment decision from Documenter to determine correct subfolder
using Documenter
include("deploy_versions.jl")

# Custom DeployConfig that bypasses PR origin check for cross-repo deployments
# This allows deploying PR previews to ai.damtp.cam.ac.uk/pysr even though
# PRs exist in astroautomata/PySR (Documenter's security check would normally block this)
struct BypassPRCheckConfig <: Documenter.DeployConfig end

function Documenter.deploy_folder(
    ::BypassPRCheckConfig;
    repo,
    devbranch,
    devurl,
    push_preview,
    branch = "gh-pages",
    branch_previews = branch,
    kwargs...
)
    # Manually determine deployment subfolder from GitHub Actions environment
    github_event = get(ENV, "GITHUB_EVENT_NAME", "")
    github_ref = get(ENV, "GITHUB_REF", "")

    # Check for pull request
    if github_event == "pull_request" && push_preview
        # Security: Verify PR is from trusted repository
        pr_repo = get(ENV, "GITHUB_REPOSITORY", "")
        if pr_repo != "astroautomata/PySR"
            println("BypassPRCheckConfig: Rejecting PR from untrusted repo: $pr_repo")
            return Documenter.DeployDecision(; all_ok = false)
        end

        m = match(r"refs/pull/(\d+)/merge", github_ref)
        if m !== nothing
            pr_number = m.captures[1]
            subfolder = "previews/PR$(pr_number)"
            println("BypassPRCheckConfig: Detected PR preview deployment to $(subfolder)")
            return Documenter.DeployDecision(;
                all_ok = true,
                branch = branch_previews,
                is_preview = true,
                repo = repo,
                subfolder = subfolder
            )
        end
    end

    # Check for master/main branch push
    if github_event in ["push", "workflow_dispatch", "schedule"]
        m = match(r"^refs/heads/(.*)$", github_ref)
        if m !== nothing && String(m.captures[1]) == devbranch
            println("BypassPRCheckConfig: Detected $(devbranch) branch deployment to $(devurl)")
            return Documenter.DeployDecision(;
                all_ok = true,
                branch = branch,
                is_preview = false,
                repo = repo,
                subfolder = devurl
            )
        end
    end

    # Check for tag deployment
    if occursin(r"^refs/tags/", github_ref)
        m = match(r"^refs/tags/(.*)$", github_ref)
        if m !== nothing
            tag = m.captures[1]
            println("BypassPRCheckConfig: Detected tag deployment to $(tag)")
            return Documenter.DeployDecision(;
                all_ok = true,
                branch = branch,
                is_preview = false,
                repo = repo,
                subfolder = tag
            )
        end
    end

    # No deployment
    println("BypassPRCheckConfig: No deployment criteria met")
    return Documenter.DeployDecision(; all_ok = false)
end

Documenter.authentication_method(::BypassPRCheckConfig) = Documenter.SSH
Documenter.documenter_key(::BypassPRCheckConfig) = ENV["DOCUMENTER_KEY"]

# Configure deployment based on target
if deployment_target == "secondary"
    # Secondary: Use custom config to bypass PR origin check
    deploy_config = BypassPRCheckConfig()
    damtp_key = get(ENV, "DAMTP_DEPLOY_KEY", "")
    if isempty(damtp_key)
        error("DAMTP_DEPLOY_KEY environment variable is required for secondary deployment but is not set")
    end
    ENV["DOCUMENTER_KEY"] = damtp_key

    deploy_decision = Documenter.deploy_folder(
        deploy_config;
        repo="github.com/ai-damtp-cam-ac-uk/pysr",
        devbranch="master",
        devurl="dev",
        push_preview=true,
    )
else
    # Primary: Use normal Documenter flow with security checks
    deploy_config = Documenter.auto_detect_deploy_system()

    deploy_decision = Documenter.deploy_folder(
        deploy_config;
        repo="github.com/astroautomata/PySR",
        devbranch="master",
        devurl="dev",
        push_preview=true,
    )
end

println("Deploy decision: all_ok=$(deploy_decision.all_ok), is_preview=$(deploy_decision.is_preview), subfolder=$(deploy_decision.subfolder)")

if !deploy_decision.all_ok || isempty(deploy_decision.subfolder)
    println("Deployment skipped because no deployable subfolder was selected")
    exit(0)
end

# VitePress bakes the base path into every asset URL at build time, so each alias
# needs its own build rather than a redirect.
bases = deploy_bases(deploy_decision.subfolder)

base_prefix = deployment_target == "secondary" ? "/pysr/" : "/PySR/"
repo_url = deployment_target == "secondary" ?
    "github.com/ai-damtp-cam-ac-uk/pysr.git" : "github.com/astroautomata/PySR.git"

function build_base(base, target_dir)
    full_base = "$(base_prefix)$(base)/"
    println("Building VitePress with base: $full_base (deploy abspath: $base_prefix)")

    config_path = joinpath(@__DIR__, "src", ".vitepress", "config.mts")
    original_config = read(config_path, String)
    modified_config = replace(original_config, r"base:\s*'/[Pp]y[Ss][Rr]/'" => "base: '$full_base'")
    modified_config = replace(
        modified_config,
        r"__DEPLOY_ABSPATH__\s*:\s*JSON\.stringify\(getBaseRepository\([^)]+\)\)" =>
            "__DEPLOY_ABSPATH__: JSON.stringify('$base_prefix')",
    )
    canonical_domain = deployment_target == "primary" ? "https://ai.damtp.cam.ac.uk/pysr/" : ""
    modified_config = replace(
        modified_config,
        r"const canonicalDomain = '';" =>
            "const canonicalDomain = '$canonical_domain';",
    )
    write(config_path, modified_config)

    try
        cd(@__DIR__) do
            run(`npm run build:vitepress`)
        end
        println("VitePress build complete")
    finally
        write(config_path, original_config)
        println("Restored original config.mts")
    end

    dist_dir = joinpath(@__DIR__, "dist")
    ispath(target_dir) && rm(target_dir; recursive = true)
    mkpath(Base.dirname(target_dir))
    mv(dist_dir, target_dir)
    write(
        joinpath(target_dir, "siteinfo.js"),
        "var DOCUMENTER_CURRENT_VERSION = $(repr(deploy_decision.subfolder));\n",
    )
end

for (i, base) in enumerate(bases)
    target_dir = joinpath(@__DIR__, "dist_bases", string(i))
    build_base(base, target_dir)
    Documenter.deploydocs(;
        root = @__DIR__,
        repo = repo_url,
        deploy_config = deploy_config,
        push_preview = true,
        devbranch = "master",
        devurl = "dev",
        target = joinpath("dist_bases", string(i)),
        dirname = base,
        versions = PySRVersions(base),
    )
end

println("Deployment complete!")
