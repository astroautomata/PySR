#!/usr/bin/env julia
#
# Build and deploy VitePress documentation to versioned gh-pages directories.
# PR previews deploy to gh-pages/previews/PR##/.

using DocumenterVitepress

println("Starting DocumenterVitepress deployment...")
println("Event: $(get(ENV, "GITHUB_EVENT_NAME", "unknown"))")
println("Ref: $(get(ENV, "GITHUB_REF", "unknown"))")

# Get deployment decision from Documenter to determine correct subfolder
using Documenter
include("deploy_versions.jl")

struct StableRedirectVersion
    base_version::DocumenterVitepress.BaseVersion
    root_files::Dict{String,String}
end

function Documenter.determine_deploy_subfolder(deploy_decision, ::StableRedirectVersion)
    return nothing
end

function Documenter.postprocess_before_push(
    versions::StableRedirectVersion;
    subfolder,
    devurl,
    deploy_dir,
    dirname,
)
    Documenter.postprocess_before_push(
        versions.base_version; subfolder, devurl, deploy_dir, dirname
    )
    root = stable_deploy_root(deploy_dir)
    for (path, content) in versions.root_files
        destination = joinpath(root, path)
        mkpath(Base.dirname(destination))
        write(destination, content)
    end
    for path in stale_root_stubs(root, keys(versions.root_files))
        println("Removing stale redirect $path")
        rm(joinpath(root, path))
        directory = joinpath(root, Base.dirname(path))
        while directory != root && isempty(readdir(directory))
            rm(directory)
            directory = Base.dirname(directory)
        end
    end
    return
end

# Custom DeployConfig that bypasses PR origin checks for cross-repo previews.
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

deploy_config = BypassPRCheckConfig()
damtp_key = get(ENV, "DAMTP_DEPLOY_KEY", "")
isempty(damtp_key) && error("DAMTP_DEPLOY_KEY environment variable is required for deployment but is not set")
ENV["DOCUMENTER_KEY"] = damtp_key

deploy_decision = Documenter.deploy_folder(
    deploy_config;
    repo="github.com/ai-damtp-cam-ac-uk/pysr",
    devbranch="master",
    devurl="dev",
    push_preview=true,
)

println("Deploy decision: all_ok=$(deploy_decision.all_ok), is_preview=$(deploy_decision.is_preview), subfolder=$(deploy_decision.subfolder)")

if !deploy_decision.all_ok || isempty(deploy_decision.subfolder)
    println("Deployment skipped because no deployable subfolder was selected")
    exit(0)
end

subfolder = deploy_decision.subfolder

base_prefix = "/"
repo_url = "github.com/ai-damtp-cam-ac-uk/pysr.git"

# VitePress bakes the base path into every asset URL at build time
full_base = "$(base_prefix)$(subfolder)/"
println("Building VitePress with base: $full_base (deploy abspath: $base_prefix)")

config_path = joinpath(@__DIR__, "src", ".vitepress", "config.mts")
original_config = read(config_path, String)
modified_config = replace(original_config, r"base:\s*'/'" => "base: '$full_base'")
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

# The version picker resolves each folder's version through this file
write(
    joinpath(@__DIR__, "dist", "siteinfo.js"),
    "var DOCUMENTER_CURRENT_VERSION = $(repr(subfolder));\n",
)

deploy(folder, target; versions = DocumenterVitepress.BaseVersion(folder)) =
    Documenter.deploydocs(;
        root = @__DIR__,
        repo = repo_url,
        deploy_config = deploy_config,
        push_preview = true,
        devbranch = "master",
        devurl = "dev",
        target = target,
        dirname = folder,
        versions = versions,
    )

deploy(subfolder, "dist")

if claims_stable(subfolder)
    println("Pointing stable at $subfolder")
    dist_dir = joinpath(@__DIR__, "dist")
    redirect_files = stable_redirect_files(built_pages(dist_dir), subfolder)
    stable_dir = joinpath(@__DIR__, "dist_stable")
    mkpath(stable_dir)
    root_files = Dict{String,String}()
    for (path, content) in redirect_files
        if startswith(path, "stable/")
            destination = joinpath(stable_dir, path[(length("stable/") + 1):end])
            mkpath(Base.dirname(destination))
            write(destination, content)
        else
            root_files[path] = content
        end
    end
    for (path, content) in stable_llms_files(dist_dir)
        write(joinpath(stable_dir, path), content)
        root_files[path] = content
    end
    versions = StableRedirectVersion(DocumenterVitepress.BaseVersion("stable"), root_files)
    deploy("stable", "dist_stable"; versions)
end

println("Deployment complete!")
