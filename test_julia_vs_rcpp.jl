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


# Initialize R and load required packages
println("\n📦 Loading R packages...")

# Check R version
println("Checking R version...")
R"""
r_version <- R.version.string
cat('Using:', r_version, '\n')
cat('R is working without stats package\n')
"""

# Note: Don't use @rlibrary for RcppArmadillo - it causes stats.dll loading issues
# We load the packages directly in R using library() calls below

# Load the BIID package (compile it first if needed)
println("\n📦 Loading BIID package...")

R"""
library(Rcpp)
suppressWarnings(library(RcppArmadillo))

# Load BIID package (must be installed first via: library(devtools); install('BIID'))
cat('Loading BIID package...\n')
library(BIID)
cat('BIID package loaded successfully\n')
"""

println("✅ BIID package loaded")

# Load Julia functions
println("📦 Loading Julia functions...")
include("julia/dimension_corrections.jl")

# Test utilities
function compare_arrays(julia_result, r_result, name::String; tolerance=1e-10)
    """Compare Julia and R results with tolerance"""
    try
        if length(julia_result) != length(r_result)
            println("❌ $name: Length mismatch - Julia: $(length(julia_result)), R: $(length(r_result))")
            return false
        end
        
        max_diff = maximum(abs.(julia_result .- r_result))
        if max_diff < tolerance
            println("✅ $name: PASS (max diff: $max_diff)")
            return true
        else
            println("❌ $name: FAIL (max diff: $max_diff exceeds tolerance $tolerance)")
            println("   Julia: $(julia_result[:min(5, end)])")
            println("   R: $(r_result[1:min(5, end)])")
            return false
        end
    catch e
        println("❌ $name: ERROR - $e")
        return false
    end
end

function compare_scalars(julia_result, r_result, name::String; tolerance=1e-10)
    """Compare scalar Julia and R results"""
    try
        diff = abs(julia_result - r_result)
        if diff < tolerance
            println("✅ $name: PASS (diff: $diff)")
            return true
        else
            println("❌ $name: FAIL (diff: $diff exceeds tolerance $tolerance)")
            println("   Julia: $julia_result")
            println("   R: $r_result")
            return false
        end
    catch e
        println("❌ $name: ERROR - $e")
        return false
    end
end

# Generate test data
println("\n🎲 Generating test data...")
Random.seed!(123)

# Test parameters
G = 3
m = 10
maxt = 20
numTests = 4
numSeasons = 4

# Generate realistic test data
birthTimes = rand(1:5, m)
startSamplingPeriod = rand(6:15, m)
endSamplingPeriod = rand(16:20, m)
X = rand([0, 1, 3, 9], m, maxt)
# Keep TestMat as Float64 with NaN for missing values in Julia
TestMat = Float64.(rand(1:m, 50, 3 + numTests))
TestMat[:, 4:end] = rand([0.0, 1.0, NaN], 50, numTests)
thetas = rand(0.1:0.1:0.9, numTests)
rhos = rand(0.1:0.1:0.9, numTests)
phis = rand(0.1:0.1:0.9, numTests)
hp_theta = [1.0, 1.0]
hp_rho = [1.0, 1.0]
hp_xi = [81.0, 60.0]

# Create TestField and TestTimes (use CORRECTED version)
TestField = TestMatAsField_CORRECTED(TestMat, m)
TestTimes = TestTimesField(TestMat, m)

println("\n🧪 Function Comparison Tests")
println("="^60)
println("Run each test block individually in the REPL")
println("="^60)

# ============================================================
# Test 1: TrProbDeath_
# ============================================================
println("\n1️⃣ Testing TrProbDeath_")
age = 5.0
a2 = 1.0
b2 = 1.0
c1 = 1.0
logar = false

# Julia result
julia_result = TrProbDeath_(age, a2, b2, c1, logar)
println("Julia result: ", julia_result)

# R result
@rput age a2 b2 c1 logar
r_result = R"TrProbDeath_(age, a2, b2, c1, logar)"[1]
println("R result:     ", r_result)

# Compare
diff = abs(julia_result - r_result)
println("Difference:   ", diff)
if diff < 1e-10
    println("✅ PASS")
else
    println("❌ FAIL")
end

# ============================================================
# Test 2: logisticD
# ============================================================
println("\n2️⃣ Testing logisticD")
x = 0.5

# Julia result
julia_result = logisticD(x)
println("Julia result: ", julia_result)

# R result
@rput x
r_result = R"logisticD(x)"[1]
println("R result:     ", r_result)

# Compare
diff = abs(julia_result - r_result)
println("Difference:   ", diff)
if diff < 1e-10
    println("✅ PASS")
else
    println("❌ FAIL")
end

# ============================================================
# Test 3: logPostThetasRhos
# ============================================================
println("\n3️⃣ Testing logPostThetasRhos")

# Julia result
julia_result = logPostThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                                TestField, TestTimes, hp_theta, hp_rho)
println("Julia result: ", julia_result)

# R result
@rput thetas rhos X startSamplingPeriod endSamplingPeriod TestField TestTimes hp_theta hp_rho
r_result = R"logPostThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho)"[1]
println("R result:     ", r_result)

# Compare
diff = abs(julia_result - r_result)
println("Difference:   ", diff)
if diff < 1e-10
    println("✅ PASS")
else
    println("❌ FAIL")
end

# ============================================================
# Test 4: gradThetasRhos
# ============================================================
println("\n4️⃣ Testing gradThetasRhos")

# Julia result
julia_result = gradThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                              TestField, TestTimes, hp_theta, hp_rho)
println("Julia result: ", julia_result)

# R result
@rput thetas rhos X startSamplingPeriod endSamplingPeriod TestField TestTimes hp_theta hp_rho
r_result = R"gradThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho)"
println("R result:     ", r_result)

# Compare (vector comparison)
max_diff = maximum(abs.(julia_result .- r_result))
println("Max difference: ", max_diff)
if max_diff < 1e-10
    println("✅ PASS")
else
    println("❌ FAIL")
end

# ============================================================
# Test 5: HMC_thetas_rhos (STOCHASTIC - requires distribution comparison)
# ============================================================
println("\n5️⃣ Testing HMC_thetas_rhos (Stochastic)")
println("This function has random components - comparing distributions...")

epsilonsens = 0.1
L = 5
n_samples = 10000

# Run Julia version n_samples times
println("Running Julia version $n_samples times...")
julia_results = zeros(n_samples, 2*numTests)
for i in 1:n_samples
    julia_results[i, :] = HMC_thetas_rhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                                          TestField, TestTimes, hp_theta, hp_rho, epsilonsens, L)
end

# Run R version n_samples times
println("Running R version $n_samples times...")
@rput thetas rhos X startSamplingPeriod endSamplingPeriod TestField TestTimes hp_theta hp_rho epsilonsens L n_samples
R"""
r_results <- matrix(0, nrow = n_samples, ncol = length(thetas) + length(rhos))
for (i in 1:n_samples) {
    r_results[i, ] <- HMC_thetas_rhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                                      TestField, TestTimes, hp_theta, hp_rho, epsilonsens, L)
}
"""
r_results = @rget r_results

# Compare distributions using statistical tests
println("\nDistribution comparison (mean ± std):")
println("-"^60)
n_params = size(julia_results, 2)
all_pass = true
for i in 1:n_params
    julia_mean = mean(julia_results[:, i])
    julia_std = std(julia_results[:, i])
    r_mean = mean(r_results[:, i])
    r_std = std(r_results[:, i])
    
    mean_diff = abs(julia_mean - r_mean)
    std_diff = abs(julia_std - r_std)
    
    # Check if means are within 3 standard errors of each other
    se = sqrt(julia_std^2 + r_std^2) / sqrt(n_samples)
    pass = mean_diff < 3 * se
    
    status = pass ? "✅" : "❌"
    println("$status Param $i:")
    println("   Julia: $(round(julia_mean, digits=4)) ± $(round(julia_std, digits=4))")
    println("   R:     $(round(r_mean, digits=4)) ± $(round(r_std, digits=4))")
    println("   Mean diff: $(round(mean_diff, digits=6)), SE: $(round(se, digits=6))")
    
    if !pass
        all_pass = false
    end
end

if all_pass
    println("\n✅ PASS - All distributions match statistically")
else
    println("\n❌ FAIL - Some distributions differ significantly")
end

# ============================================================
# Test 6: TestMatAsField
# ============================================================
println("\n6️⃣ Testing TestMatAsField")

# Julia result (use CORRECTED version)
julia_result = TestMatAsField_CORRECTED(TestMat, m)
println("Julia result: ", julia_result)

# R result
@rput TestMat m
R"""
r_field <- TestMatAsField(TestMat, m)
# Convert field to list of matrices for comparison
r_result_list <- lapply(r_field, function(x) matrix(x, ncol=ncol(x)))
"""
r_result = @rget r_result_list
println("R result:     ", r_result)

# Compare field arrays (need to compare each element)
println("Julia result length: ", length(julia_result))
println("R result length:     ", length(r_result))
field_match = true

if length(julia_result) != length(r_result)
    println("❌ FAIL - Different number of fields")
    field_match = false
else
    for i in 1:length(julia_result)
        j_field = julia_result[i]
        r_field = r_result[i]
        
        println("Field $i - Julia size: ", size(j_field), ", R size: ", size(r_field))
        
        if size(j_field) != size(r_field)
            println("  ❌ Size mismatch")
            field_match = false
        else
            # Compare element-wise, handling NaN and Missing
            # R's NA becomes Missing in Julia, our NaN stays as NaN
            max_diff = 0.0
            n_compared = 0
            
            for idx in CartesianIndices(j_field)
                j_val = j_field[idx]
                r_val = r_field[idx]
                
                # Check if both are missing/NaN
                j_missing = isnan(j_val)
                r_missing = ismissing(r_val) || (r_val isa Float64 && isnan(r_val))
                
                if j_missing && r_missing
                    # Both missing - OK
                    continue
                elseif j_missing || r_missing
                    # One missing, one not - FAIL
                    println("  ❌ Mismatch at $idx: Julia=$j_val, R=$r_val")
                    field_match = false
                    break
                else
                    # Both have values - compare
                    diff = abs(j_val - r_val)
                    max_diff = max(max_diff, diff)
                    n_compared += 1
                end
            end
            
            if n_compared > 0
                println("  Max diff: $max_diff (compared $n_compared values)")
                if max_diff > 1e-10
                    field_match = false
                end
            else
                println("  All values are missing/NaN")
            end
        end
    end
end

if field_match
    println("✅ PASS")
else
    println("❌ FAIL")
end

# ============================================================
# Test 7: logPostXi
# ============================================================
println("\n7️⃣ Testing logPostXi")
xi = 15
xiMin = 10
xiMax = 20

# Julia result
julia_result = logPostXi(xiMin, xiMax, xi, hp_xi, TestField, TestTimes, thetas, rhos, phis, 
                         X, startSamplingPeriod, endSamplingPeriod)
println("Julia result: ", julia_result)

# R result
@rput xiMin xiMax xi hp_xi TestField TestTimes thetas rhos phis X startSamplingPeriod endSamplingPeriod
r_result = R"logPostXi(xiMin, xiMax, xi, hp_xi, TestField, TestTimes, thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)"[1]
println("R result:     ", r_result)

# Compare
diff = abs(julia_result - r_result)
println("Difference:   ", diff)
if diff < 1e-10
    println("✅ PASS")
else
    println("❌ FAIL")
end

# ============================================================
# Test 8: RWMH_xi (STOCHASTIC)
# ============================================================
println("\n8️⃣ Testing RWMH_xi (Stochastic)")
println("This function has random components - comparing distributions...")

xi_cur = 15
xi_can = 18
n_samples = 1000

# Run Julia version n_samples times
println("Running Julia version $n_samples times...")
julia_results = zeros(n_samples)
for i in 1:n_samples
    TestField_copy = deepcopy(TestField)
    TestFieldProposal_copy = deepcopy(TestField)
    julia_results[i] = RWMH_xi(xi_can, xi_cur, hp_xi, TestFieldProposal_copy, TestField_copy, TestTimes, 
                               thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
end

# Run R version n_samples times
println("Running R version $n_samples times...")
@rput xi_can xi_cur hp_xi TestField TestTimes thetas rhos phis X startSamplingPeriod endSamplingPeriod n_samples
R"""
r_results <- numeric(n_samples)
for (i in 1:n_samples) {
    TestFieldProposal <- TestField
    r_results[i] <- RWMH_xi(xi_can, xi_cur, hp_xi, TestFieldProposal, TestField, TestTimes, 
                            thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
}
"""
r_results = @rget r_results

# Compare distributions
julia_mean = mean(julia_results)
julia_std = std(julia_results)
r_mean = mean(r_results)
r_std = std(r_results)

mean_diff = abs(julia_mean - r_mean)
se = sqrt(julia_std^2 + r_std^2) / sqrt(n_samples)
pass = mean_diff < 3 * se

println("\nDistribution comparison:")
println("Julia: $(round(julia_mean, digits=4)) ± $(round(julia_std, digits=4))")
println("R:     $(round(r_mean, digits=4)) ± $(round(r_std, digits=4))")
println("Mean diff: $(round(mean_diff, digits=6)), SE: $(round(se, digits=6))")

if pass
    println("✅ PASS")
else
    println("❌ FAIL")
end

# ============================================================
# Test 9: TestMatAsFieldProposal
# ============================================================
println("\n9️⃣ Testing TestMatAsFieldProposal")
xi = 15
xiCan = 18

# Julia version
TestFieldProposal_julia = deepcopy(TestField)
TestMatAsFieldProposal(TestFieldProposal_julia, TestField, TestTimes, xi, xiCan, m)
println("Julia result: ", TestFieldProposal_julia)

# R version
@rput TestField TestTimes xi xiCan m
R"""
TestFieldProposal_r <- TestField
TestMatAsFieldProposal(TestFieldProposal_r, TestField, TestTimes, xi, xiCan, m)
# Convert to list of matrices
TestFieldProposal_r <- lapply(TestFieldProposal_r, function(x) matrix(x, ncol=ncol(x)))
"""
TestFieldProposal_r = @rget TestFieldProposal_r
println("R result:     ", TestFieldProposal_r)

# Compare the modified TestFieldProposal
println("Julia result length: ", length(TestFieldProposal_julia))
println("R result length:     ", length(TestFieldProposal_r))
proposal_match = true

if length(TestFieldProposal_julia) != length(TestFieldProposal_r)
    println("❌ FAIL - Different number of fields")
    proposal_match = false
else
    for i in 1:length(TestFieldProposal_julia)
        j_field = TestFieldProposal_julia[i]
        r_field = TestFieldProposal_r[i]
        
        println("Field $i - Julia size: ", size(j_field), ", R size: ", size(r_field))
        
        if size(j_field) != size(r_field)
            println("  ❌ Size mismatch")
            proposal_match = false
        else
            # Compare element-wise, handling NaN and Missing
            max_diff = 0.0
            n_compared = 0
            
            for idx in CartesianIndices(j_field)
                j_val = j_field[idx]
                r_val = r_field[idx]
                
                j_missing = isnan(j_val)
                r_missing = ismissing(r_val) || (r_val isa Float64 && isnan(r_val))
                
                if j_missing && r_missing
                    continue
                elseif j_missing || r_missing
                    println("  ❌ Mismatch at $idx: Julia=$j_val, R=$r_val")
                    proposal_match = false
                    break
                else
                    diff = abs(j_val - r_val)
                    max_diff = max(max_diff, diff)
                    n_compared += 1
                end
            end
            
            if n_compared > 0
                println("  Max diff: $max_diff (compared $n_compared values)")
                if max_diff > 1e-10
                    proposal_match = false
                end
            else
                println("  All values are missing/NaN")
            end
        end
    end
end

if proposal_match
    println("✅ PASS")
else
    println("❌ FAIL")
end

println("\n🏁 All tests completed!")

# ============================================================
# Test 7: iFFBS_ Comparison with Heatmap Visualization
# ============================================================
println("\n7️⃣ Testing iFFBS_ with heatmap visualization")

using GLMakie

# Load real data and parameters
println("Loading data and parameters from get_pars_and_data.jl...")
include("get_pars_and_data.jl")

println("Data loaded successfully!")
println("m = $m, maxt = $maxt, G = $G, numTests = $numTests")

# Process data similar to MCMCiFFBS_.jl initialization
numStates = 4

# Initialize parameters from initParamValues
alphaStarInit = initParamValues[1]
lambdaInit = initParamValues[2]
betaInit = initParamValues[3]
qInit = initParamValues[4]
tauInit = initParamValues[5]
a2Init = initParamValues[6]
b2Init = initParamValues[7]
c1Init = initParamValues[8]

nuEInit = zeros(numNuTimes)
nuIInit = zeros(numNuTimes)
for i_nu in 1:numNuTimes
    nuEInit[i_nu] = initParamValues[7 + 1 + i_nu]
end
for i_nu in 1:numNuTimes
    nuIInit[i_nu] = initParamValues[7 + 1 + numNuTimes + i_nu]
end

xiInit = Int(initParamValues[7 + 2 + 2*numNuTimes])

thetasInit = zeros(numTests)
rhosInit = zeros(numTests)
phisInit = zeros(numTests)
etasInit = zeros(numSeasons)

nParsNotGibbs = G + 4 + 3 + 2*numNuTimes + 1
for iTest in 1:numTests
    thetasInit[iTest] = initParamValues[nParsNotGibbs-G+1+iTest]
end
for iTest in 1:numTests
    rhosInit[iTest] = initParamValues[nParsNotGibbs-G+1+numTests+iTest]
end
for iTest in 1:numTests
    phisInit[iTest] = initParamValues[nParsNotGibbs-G+1+2*numTests+iTest]
end
for s in 1:numSeasons
    etasInit[s] = initParamValues[nParsNotGibbs-G+1+3*numTests+s]
end

# Set up state variables
alpha_js = zeros(Float64, G)
for g in 1:G
    alpha_js[g] = alphaStarInit * lambdaInit
end

beta = betaInit
q = qInit
tau = tauInit
a2 = a2Init
b2 = b2Init
c1 = c1Init

nuEs = copy(nuEInit)
nuIs = copy(nuIInit)
xi = copy(xiInit)

thetas = copy(thetasInit)
rhos = copy(rhosInit)
phis = copy(phisInit)
etas = copy(etasInit)

seasonVec = MakeSeasonVec_(numSeasons, startingQuarter, maxt)

# Initialize X from Xinit
X = copy(Xinit)

# Calculate age matrix
ageMat = fill(-10, m, maxt)
for i in 1:m
    mint_i = max(1, birthTimes[i])
    for tt in mint_i:maxt
        ageMat[i, tt] = tt - birthTimes[i]
    end
end

# Get social group structure
SocGroup = LocateIndiv(TestMat, birthTimes)

# Calculate probability matrices
probDyingMat = zeros(Float64, m, maxt)
LogProbDyingMat = zeros(Float64, m, maxt)
LogProbSurvMat = zeros(Float64, m, maxt)

for i in 1:m
    for tt in 1:maxt
        age_i_tt = ageMat[i, tt]
        if age_i_tt > 0
            probDyingMat[i, tt] = TrProbDeath_(Float64(age_i_tt), a2, b2, c1, false)
            LogProbDyingMat[i, tt] = TrProbDeath_(Float64(age_i_tt), a2, b2, c1, true)
            LogProbSurvMat[i, tt] = log(1.0 - probDyingMat[i, tt])
        end
    end
end

# Initialize numInfecMat and mPerGroup
numInfecMat = zeros(Int, G, maxt-1)
for tt in 1:maxt-1
    for ii in 2:m
        if X[ii, tt] == 1
            g_i_tt = SocGroup[ii, tt]
            if g_i_tt != 0
                numInfecMat[g_i_tt, tt] += 1
            end
        end
    end
end

mPerGroup = zeros(Int, G, maxt)
for tt in 1:maxt
    for ii in 2:m
        if (X[ii, tt] == 0) || (X[ii, tt] == 1) || (X[ii, tt] == 3)
            g_i_tt = SocGroup[ii, tt]
            if g_i_tt != 0
                mPerGroup[g_i_tt, tt] += 1
            end
        end
    end
end

# Process test data
TestField = TestMatAsField_CORRECTED(TestMat, m)
TestTimes = TestTimesField(TestMat, m)

idVecAll = collect(1:m)

# Working matrices for iFFBS
corrector = zeros(Float64, maxt, numStates)
predProb = zeros(Float64, maxt, numStates)
filtProb = zeros(Float64, maxt, numStates)

# Calculate transition probabilities
logProbRest = zeros(Float64, maxt-1, numStates, m)
logTransProbRest = zeros(Float64, numStates, maxt-1)

# Transition probability matrices (simplified initialization)
logProbStoSgivenSorE = zeros(Float64, m, maxt)
logProbStoEgivenSorE = zeros(Float64, m, maxt)
logProbStoSgivenI = zeros(Float64, m, maxt)
logProbStoEgivenI = zeros(Float64, m, maxt)
logProbStoSgivenD = zeros(Float64, m, maxt)
logProbStoEgivenD = zeros(Float64, m, maxt)

# These will be calculated properly in the loop
for tt in 1:maxt
    for i in 1:m
        g_i = SocGroup[i, tt]
        if g_i > 0
            m_g_tt = mPerGroup[g_i, tt]
            if i >= 2
                m_g_tt += 1  # Add current individual
            end
            
            if m_g_tt > 0
                logProbStoSgivenSorE[i, tt] = log(1.0 - alpha_js[g_i])
                logProbStoEgivenSorE[i, tt] = log(alpha_js[g_i])
            end
        end
    end
end

logProbEtoE = log(1.0 - 1.0/tau)
logProbEtoI = log(1.0/tau)

# Update tracking
whichRequireUpdate = [Int[] for _ in 1:maxt]
sumLogCorrector = 0.0

# Test on first individual
id_test = 1
birthTime_test = birthTimes[id_test]
startTime_test = startSamplingPeriod[id_test]
endTime_test = endSamplingPeriod[id_test]

println("Running iFFBS_ 10 times in Julia and R...")

# Storage for results
n_runs = 10
maxt_range = endTime_test - startTime_test + 1
julia_results = zeros(n_runs, maxt_range, numStates)
r_results = zeros(n_runs, maxt_range, numStates)

# Run Julia version
println("Running Julia version...")
for run in 1:n_runs
    Random.seed!(1000 + run)
    
    # Reset state
    alpha_js_copy = copy(alpha_js)
    X_copy = copy(X)
    corrector_copy = zeros(Float64, maxt, numStates)
    predProb_copy = zeros(Float64, maxt, numStates)
    filtProb_copy = zeros(Float64, maxt, numStates)
    sumLogCorrector_copy = 0.0
    
    iFFBS_(alpha_js_copy, beta, q, tau, k, K,
           probDyingMat, LogProbDyingMat, LogProbSurvMat,
           logProbRest, nuTimes, nuEs, nuIs,
           thetas, rhos, phis, etas,
           id_test, birthTime_test, startTime_test, endTime_test,
           X_copy, seasonVec, TestField[id_test], TestTimes[id_test], CaptHist,
           corrector_copy, predProb_copy, filtProb_copy,
           logTransProbRest, numInfecMat, SocGroup, mPerGroup, idVecAll,
           logProbStoSgivenSorE, logProbStoEgivenSorE,
           logProbStoSgivenI, logProbStoEgivenI,
           logProbStoSgivenD, logProbStoEgivenD,
           logProbEtoE, logProbEtoI,
           whichRequireUpdate, sumLogCorrector_copy)
    
    # Extract relevant time range
    julia_results[run, :, :] = filtProb_copy[startTime_test:endTime_test, :]
end

# Run R version
println("Running R version...")

# Transfer data to R
@rput m maxt numStates G id_test birthTime_test startTime_test endTime_test maxt_range
@rput beta q tau k K
@rput probDyingMat LogProbDyingMat LogProbSurvMat logProbRest
@rput nuTimes nuEs nuIs thetas rhos phis etas
@rput X seasonVec CaptHist
@rput logTransProbRest numInfecMat SocGroup mPerGroup idVecAll
@rput logProbStoSgivenSorE logProbStoEgivenSorE
@rput logProbStoSgivenI logProbStoEgivenI
@rput logProbStoSgivenD logProbStoEgivenD
@rput logProbEtoE logProbEtoI n_runs alpha_js

# Transfer TestField and TestTimes for the specific individual
TestField_id = TestField[id_test]
TestTimes_id = TestTimes[id_test]
@rput TestField_id TestTimes_id

R"""
# Storage for results
r_results <- array(0, dim = c(n_runs, maxt_range, numStates))

# Convert whichRequireUpdate to R list
whichRequireUpdate_r <- vector("list", maxt)
for (i in 1:maxt) {
    whichRequireUpdate_r[[i]] <- integer(0)
}

for (run in 1:n_runs) {
    set.seed(1000 + run)
    
    # Reset state
    alpha_js_r <- alpha_js
    X_r <- X
    corrector_r <- matrix(0, maxt, numStates)
    predProb_r <- matrix(0, maxt, numStates)
    filtProb_r <- matrix(0, maxt, numStates)
    sumLogCorrector_r <- 0.0
    
    iFFBS_(alpha_js_r, beta, q, tau, k, K,
           probDyingMat, LogProbDyingMat, LogProbSurvMat,
           logProbRest, nuTimes, nuEs, nuIs,
           thetas, rhos, phis, etas,
           id_test, birthTime_test, startTime_test, endTime_test,
           X_r, seasonVec, TestField_id, TestTimes_id, CaptHist,
           corrector_r, predProb_r, filtProb_r,
           logTransProbRest, numInfecMat, SocGroup, mPerGroup, idVecAll,
           logProbStoSgivenSorE, logProbStoEgivenSorE,
           logProbStoSgivenI, logProbStoEgivenI,
           logProbStoSgivenD, logProbStoEgivenD,
           logProbEtoE, logProbEtoI,
           whichRequireUpdate_r, sumLogCorrector_r)
    
    # Extract relevant time range
    r_results[run, , ] <- filtProb_r[startTime_test:endTime_test, ]
}
"""

r_results = @rget r_results

println("Creating heatmap visualization...")

# Compute differences
diff_results = abs.(julia_results .- r_results)

# Create figure with 3 subplots
fig = Figure(resolution = (1800, 600))

# Julia results (average across runs)
ax1 = Axis(fig[1, 1], 
           title = "Julia iFFBS_ (avg filtProb)",
           xlabel = "Time",
           ylabel = "State")
julia_avg = mean(julia_results, dims=1)[1, :, :]
hm1 = heatmap!(ax1, 1:maxt_range, 1:numStates, julia_avg',
               colormap = :viridis)
Colorbar(fig[1, 2], hm1, label = "Probability")

# R results (average across runs)
ax2 = Axis(fig[1, 3],
           title = "R iFFBS_ (avg filtProb)",
           xlabel = "Time",
           ylabel = "State")
r_avg = mean(r_results, dims=1)[1, :, :]
hm2 = heatmap!(ax2, 1:maxt_range, 1:numStates, r_avg',
               colormap = :viridis)
Colorbar(fig[1, 4], hm2, label = "Probability")

# Difference heatmap
ax3 = Axis(fig[1, 5],
           title = "Absolute Difference |Julia - R|",
           xlabel = "Time",
           ylabel = "State")
diff_avg = mean(diff_results, dims=1)[1, :, :]
hm3 = heatmap!(ax3, 1:maxt_range, 1:numStates, diff_avg',
               colormap = :hot)
Colorbar(fig[1, 6], hm3, label = "Abs Difference")

display(fig)

# Print summary statistics
println("\n" * "="^60)
println("Summary Statistics:")
println("="^60)
println("Julia mean:  ", mean(julia_results))
println("R mean:      ", mean(r_results))
println("Max diff:    ", maximum(diff_results))
println("Mean diff:   ", mean(diff_results))
println("Median diff: ", median(diff_results))

if maximum(diff_results) < 1e-10
    println("\n✅ PASS - Results match within tolerance")
else
    println("\n⚠️  WARNING - Results differ by more than 1e-10")
    println("This may indicate implementation differences")
end

println("\n🏁 iFFBS_ comparison completed!")
