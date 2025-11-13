#!/usr/bin/env julia
"""
Benchmark specific components of the MCMC algorithm.

This script provides detailed timing information for different parts of the algorithm.
"""

using BenchmarkTools
using Statistics

println("\n" * "="^60)
println("MCMC ALGORITHM BENCHMARKING")
println("="^60)

# Load all the required packages and functions
println("\nLoading packages and data...")
include("runmodel_RDS.jl")

println("\n" * "="^60)
println("TIMING FULL MCMC RUN")
println("="^60)

# Time a short run
N_test = 100
println("\nRunning $(N_test) iterations...")

time_start = time()
out_test = MCMCiFFBS_(
    N_test,
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
    blockSize
)
time_end = time()

elapsed = time_end - time_start
per_iter = elapsed / N_test

println("\n" * "="^60)
println("TIMING RESULTS")
println("="^60)
println("Total time for $(N_test) iterations: $(round(elapsed, digits=2)) seconds")
println("Time per iteration: $(round(per_iter * 1000, digits=2)) ms")
println("\nEstimated time for full run:")
println("  1,000 iterations: $(round(per_iter * 1000 / 60, digits=2)) minutes")
println("  10,000 iterations: $(round(per_iter * 10000 / 60, digits=2)) minutes")
println("  25,000 iterations: $(round(per_iter * 25000 / 60, digits=2)) minutes")
println("="^60)

# Memory allocation info
println("\n" * "="^60)
println("MEMORY ALLOCATION")
println("="^60)
println("\nRunning allocation analysis...")

# Run once more to get allocation info
alloc_result = @timed MCMCiFFBS_(
    10,  # Just 10 iterations for allocation check
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
    blockSize
)

println("Time: $(round(alloc_result.time, digits=3)) seconds")
println("Allocations: $(alloc_result.bytes ÷ 1024 ÷ 1024) MB")
println("GC time: $(round(alloc_result.gctime, digits=3)) seconds")
println("="^60)

println("\n✓ Benchmarking complete!")
println("\nFor detailed profiling with flame graphs, run:")
println("  julia profile_mcmc.jl")
