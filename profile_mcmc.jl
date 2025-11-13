#!/usr/bin/env julia
"""
Profile the MCMC-iFFBS algorithm to identify performance bottlenecks.

Usage:
    julia profile_mcmc.jl

This will run a short MCMC chain (e.g., 500 iterations) and generate profiling data.
"""

using Profile
using ProfileView  # Install with: using Pkg; Pkg.add("ProfileView")
using PProf        # Install with: using Pkg; Pkg.add("PProf")

# Include your main script setup
include("runmodel_RDS.jl")

println("\n" * "="^60)
println("PROFILING MCMC ALGORITHM")
println("="^60)

# Modify N for profiling (shorter run)
N_profile = 500  # Run 500 iterations for profiling
blockSize_profile = 100

println("Running $(N_profile) iterations for profiling...")
println("This may take a few minutes...")

# Clear any previous profiling data
Profile.clear()

# Run with profiling
@profile begin
    out_profile = MCMCiFFBS_(
        N_profile,
        initParamValues,
        Xinit,
        TestMat,
        CaptHist,
        birthTimes,
        startSamplingPeriod,
        endSamplingPeriod,
        nuTimes,
        CaptEffort,
        capturesAfterMonit,
        numSeasons,
        seasonStart,
        maxt,
        hp_lambda,
        hp_beta,
        hp_q,
        hp_tau,
        hp_a2,
        hp_b2,
        hp_c1,
        hp_nu,
        hp_xi,
        hp_theta,
        hp_rho,
        hp_phi,
        hp_eta,
        k,
        K,
        sd_xi_min,
        method,
        epsilon,
        epsilonalphas,
        epsilonbq,
        epsilontau,
        epsilonc1,
        epsilonsens,
        L,
        path,
        blockSize_profile
    )
end

println("\n" * "="^60)
println("PROFILING COMPLETE")
println("="^60)

# Print profile statistics
println("\nTop 20 functions by time:")
Profile.print(format=:flat, sortedby=:count, maxdepth=20)

println("\n" * "="^60)
println("GENERATING PROFILE VISUALIZATIONS")
println("="^60)

# Generate flame graph (opens in browser)
try
    println("\nGenerating flame graph with PProf...")
    pprof()
    println("✓ Flame graph opened in browser")
catch e
    println("⚠ Could not generate PProf flame graph: $e")
end

# Generate ProfileView (interactive GUI)
try
    println("\nGenerating ProfileView visualization...")
    ProfileView.view()
    println("✓ ProfileView window opened")
    println("  (Close the ProfileView window to continue)")
catch e
    println("⚠ Could not generate ProfileView: $e")
end

println("\n" * "="^60)
println("PROFILING SUMMARY")
println("="^60)
println("Profile data collected from $(N_profile) MCMC iterations")
println("\nTo view the profile data again, run:")
println("  using Profile, ProfileView")
println("  ProfileView.view()  # Interactive GUI")
println("  Profile.print()     # Text output")
println("\nTo generate flame graph:")
println("  using PProf")
println("  pprof()")
println("="^60)
