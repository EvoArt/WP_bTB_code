# ============================================================
# Julia vs R/Rcpp Function Comparison Tests
# ============================================================
# 
# HOW TO USE:
# 1. First, install BIID package in R:
#    library(devtools); install("BIID")
#
# 2. Run this script up to the test data generation section
#    (this loads packages and creates test data)
#
# 3. Then run each test block individually in the REPL
#    Each test shows Julia result, R result, and difference
#
# ============================================================

# Set R_HOME to use R 4.5.1 before loading RCall
ENV["R_HOME"] = "C:/Program Files/R/R-4.5.1"

# Add R directories to PATH - this must be done BEFORE loading RCall
r_home = ENV["R_HOME"]
r_bin_x64 = joinpath(r_home, "bin", "x64")
r_bin = joinpath(r_home, "bin")

# Update PATH with R directories
ENV["PATH"] = r_bin_x64 * ";" * r_bin * ";" * ENV["PATH"]

# Preload R.dll and other critical DLLs to ensure they're available for package DLLs
if Sys.iswindows()
    # Load R.dll
    r_dll_path = joinpath(r_bin_x64, "R.dll")
    try
        Libc.Libdl.dlopen(r_dll_path, Libc.Libdl.RTLD_GLOBAL)
        println("Preloaded R.dll successfully")
    catch e
        println("Warning: Could not preload R.dll: $e")
    end
    
    # Also preload Rblas.dll and Rlapack.dll which stats.dll depends on
    for dll_name in ["Rblas.dll", "Rlapack.dll"]
        dll_path = joinpath(r_bin_x64, dll_name)
        if isfile(dll_path)
            try
                Libc.Libdl.dlopen(dll_path, Libc.Libdl.RTLD_GLOBAL)
                println("Preloaded $dll_name successfully")
            catch e
                println("Warning: Could not preload $dll_name: $e")
            end
        end
    end
end

# Don't load stats package by default - it will be loaded when needed
ENV["R_DEFAULT_PACKAGES"] = "datasets,utils,grDevices,graphics,methods"

using Random
using Statistics
using RCall
using Test
using StatsFuns
using Distributions
using DataFrames
using RData
# Set seed for reproducible testing
Random.seed!(123)

# Check R version
println("Checking R version...")
R"""
r_version <- R.version.string
cat('Using:', r_version, '\n')
cat('R is working without stats package\n')
"""

# Note: Don't use @rlibrary for RcppArmadillo - it causes stats.dll loading issues
# We load the packages directly in R using library() calls below

# Reinstall BIID package to include modular functions
println("\n📦 Reinstalling BIID package to include modular functions...")

R"""
library(Rcpp)
suppressWarnings(library(RcppArmadillo))
library(devtools)
library(BIID)
cat('BIID package loaded\n')
"""

# Load Julia functions
println("📦 Loading Julia functions...")
include("julia/dimension_corrections.jl")
include("julia/iFFBS_modular.jl")

# ============================================================
# Test 7: iFFBS_ Comparison with Heatmap Visualization
# ============================================================
println("\n7️⃣ Testing iFFBS_ with heatmap visualization")

using GLMakie

# Load real data and parameters
println("Loading data and parameters from get_pars_and_data.jl...")
include("get_pars_and_data.jl")
# Select an individual for testing (e.g., individual 1)
id = 1
birthTime = birthTimes[id]
startTime = startSamplingPeriod[id]
endTime = endSamplingPeriod[id]

println("🔍 Starting iFFBS for individual ", id, " (birth: ", birthTime, ", start: ", startTime, ", end: ", endTime, ")")

# Helper function to compare matrices
function compare_matrices(name, julia_mat, r_mat, indices=nothing)
    if indices === nothing
        diff = maximum(abs.(julia_mat .- r_mat))
        println("  $name max diff: $diff")
        if diff > 1e-10
            println("    WARNING: Significant difference detected!")
        end
    else
        for idx in indices
            j_val = julia_mat[idx...]
            r_val = r_mat[idx...]
            diff = abs(j_val - r_val)
            println("  $name[$idx]: Julia=$j_val, R=$r_val, diff=$diff")
        end
    end
end

# 1. Initialize forward filtering
println("\n1️⃣ Initializing forward filtering...")
t0 = startTime
maxt_i = endTime - t0

# Call R/C++ version first
println("  Calling R/C++ version...")
predProb_R = copy(predProb)
result1_R = R"""
library(BIID)
iFFBS_initializeForwardFiltering(
    $birthTime,
    $startTime,
    $nuTimes,
    $nuEs,
    $nuIs,
    $predProb_R,
    $t0,
    $numStates
)
"""
println("  R: nuE_i = ", rcopy(result1_R[:nuE_i]), ", nuI_i = ", rcopy(result1_R[:nuI_i]))

# Call Julia version
println("  Calling Julia version...")
result1 = iFFBS_initializeForwardFiltering(
    birthTime,
    startTime,
    nuTimes,
    nuEs,
    nuIs,
    predProb,
    t0,
    numStates
)
predProb = result1.predProb
println("  Julia: nuE_i = ", result1.nuE_i, ", nuI_i = ", result1.nuI_i)

# Compare results
println("  Comparing predProb[t0, :]...")
predProb_R_mat = rcopy(result1_R[:predProb])
println("  R:     ", predProb_R_mat[t0, :])
println("  Julia: ", predProb[t0, :])
println("  Diff:  ", predProb[t0, :] .- predProb_R_mat[t0, :])

# 2. First step of forward filtering
println("\n2️⃣ First forward filtering step...")

# Call R/C++ version first
println("  Calling R/C++ version...")
corrector_R = copy(corrector)
predProb_R = copy(predProb)
filtProb_R = copy(filtProb)
logTransProbRest_R = copy(logTransProbRest')  # Transpose for R (states x time)
result2_R = R"""
library(BIID)
iFFBS_forwardFilteringFirstStep(
    $corrector_R,
    $predProb_R,
    $filtProb_R,
    $logTransProbRest_R,
    $t0 - 1,  # Convert to 0-based for C++
    $maxt - 1,
    $numStates
)
"""

# Call Julia version
println("  Calling Julia version...")
result2 = iFFBS_forwardFilteringFirstStep(
    corrector,
    predProb,
    filtProb,
    logTransProbRest,
    t0,
    maxt,
    numStates
)
filtProb = result2.filtProb

# Compare results
println("  Comparing filtProb[t0, :]...")
filtProb_R_mat = rcopy(result2_R[:filtProb])
println("  R:     ", filtProb_R_mat[t0, :])
println("  Julia: ", filtProb[t0, :])
println("  Diff:  ", filtProb[t0, :] .- filtProb_R_mat[t0, :])
println("  First filtering step completed")

# 3. Main forward filtering loop
println("\n3️⃣ Running forward filtering loop...")
result3 = iFFBS_forwardFilteringLoop(
    predProb,
    filtProb,
    corrector,
    logTransProbRest,
    probDyingMat,
    logProbStoSgivenSorE,
    logProbStoEgivenSorE,
    logProbEtoE,
    logProbEtoI,
    SocGroup,
    id,
    t0,
    maxt_i,
    numStates
)
predProb = result3.predProb
filtProb = result3.filtProb
println("  Forward filtering loop completed")

# 4. Final step of forward filtering
println("\n4️⃣ Final forward filtering step...")
result4 = iFFBS_forwardFilteringFinalStep(
    predProb,
    filtProb,
    corrector,
    logTransProbRest,
    probDyingMat,
    logProbStoSgivenSorE,
    logProbStoEgivenSorE,
    logProbEtoE,
    logProbEtoI,
    SocGroup,
    id,
    t0,
    maxt_i,
    maxt,
    numStates
)
predProb = result4.predProb
filtProb = result4.filtProb
println("  Final filtering step completed")

# 5. Backward sampling
println("\n5️⃣ Running backward sampling...")
result5 = iFFBS_backwardSampling(
    X,
    filtProb,
    predProb,
    probDyingMat,
    alpha_js,
    beta,
    q,
    tau,
    k,
    K,
    SocGroup,
    mPerGroup,
    numInfecMat,
    id,
    birthTime,
    startTime,
    endTime,
    t0,
    maxt_i
)
X = result5.X
println("  Backward sampling completed")
println("  Sampled state at endTime: ", X[id, endTime])

# 6. Calculate log corrector
println("\n6️⃣ Calculating log corrector...")
sumLogCorrector = iFFBS_calculateLogCorrector(
    X,
    corrector,
    id,
    t0,
    maxt_i
)
println("  Log corrector sum = ", sumLogCorrector)

# 7. Update group statistics
println("\n7️⃣ Updating group statistics...")
idNext = id + 1
if idNext <= m
    result7 = iFFBS_updateGroupStatistics(
        X,
        numInfecMat,
        mPerGroup,
        logProbStoSgivenSorE,
        logProbStoEgivenSorE,
        logProbStoSgivenI,
        logProbStoEgivenI,
        logProbStoSgivenD,
        logProbStoEgivenD,
        alpha_js,
        beta,
        q,
        K,
        SocGroup,
        id,
        idNext,
        m,
        maxt
    )
    numInfecMat = result7.numInfecMat
    mPerGroup = result7.mPerGroup
    logProbStoSgivenSorE = result7.logProbStoSgivenSorE
    logProbStoEgivenSorE = result7.logProbStoEgivenSorE
    logProbStoSgivenI = result7.logProbStoSgivenI
    logProbStoEgivenI = result7.logProbStoEgivenI
    logProbStoSgivenD = result7.logProbStoSgivenD
    logProbStoEgivenD = result7.logProbStoEgivenD
    println("  Group statistics updated")
else
    println("  Skipping (last individual)")
end

# 8. Update transition probabilities
println("\n8️⃣ Updating transition probabilities...")
if idNext <= m
    result8 = iFFBS_updateTransitionProbabilities(
        logProbRest,
        logTransProbRest,
        X,
        SocGroup,
        LogProbDyingMat,
        LogProbSurvMat,
        logProbStoSgivenSorE,
        logProbStoEgivenSorE,
        logProbStoSgivenI,
        logProbStoEgivenI,
        logProbStoSgivenD,
        logProbStoEgivenD,
        logProbEtoE,
        logProbEtoI,
        whichRequireUpdate,
        id,
        idNext,
        m,
        maxt,
        numStates
    )
    logProbRest = result8.logProbRest
    logTransProbRest = result8.logTransProbRest
    println("  Transition probabilities updated")
else
    println("  Skipping (last individual)")
end

println("\n✅ iFFBS completed successfully for individual ", id)

# Print some summary statistics
println("\n📊 Summary of results:")
println("- Final state at end time: ", X[id, endTime])
println("- Sum of log corrector: ", sumLogCorrector)
println("- Number of time points with updates: ", length([t for t in 1:maxt if !isempty(whichRequireUpdate[t])]))

# Visualize the results
using Plots

# Plot the state trajectory for this individual
plot(startTime:endTime, X[id, startTime:endTime], 
     xlabel="Time", ylabel="State", 
     title="State Trajectory for Individual $id",
     legend=false, marker=:circle, yticks=(0:3, ["S", "E", "I", "D"]))

# Save the plot
savefig("individual_$(id)_trajectory.png")
println("\n📈 Saved trajectory plot to individual_$(id)_trajectory.png")

