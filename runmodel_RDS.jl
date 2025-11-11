"""
    runmodel_RDS.jl

Julia version of the R script to run the MCMC-iFFBS model for badger TB transmission.
This script loads data from RDS files, sets up initial conditions, and runs the MCMC algorithm.
"""

## Load required packages
using OffsetArrays: Origin # to use 0-based indexing
using Distributions
using Random
using DataFrames
using RData
using Statistics
using CSV
# Print indexing warning
print_indexing_warning()
include("julia/iFFBS.jl")
include("julia/dimension_corrections.jl")
include("julia/MCMCiFFBS_.jl")


macro zero_based(x)
    return :( Origin(0)($x) )
end

## Set seed
seed = 2
Random.seed!(seed)

## Set folder name for results to be stored in
modelFileName = "outputs_seed$seed/"

## Set path to data
dataDirectory = "WPbadgerData/"

#########################################
#####      Set hyperparameters      #####
#########################################

method = 1  # "HMC"
# method = 2  # "RWMH"
epsilon = 0.01
epsilonalphas = 0.2
epsilonbq = 0.05
epsilontau = epsilon # 0.2
epsilonc1 = 0.05
epsilonsens = 0.1
lambda_b = 1
beta_b = 1
tau_b = 0.01
a2_b = 1
b2_b = 1
c1_b = 1
xi_mean = 81
xi_sd = 60
sd_xi_min = 8
k = 1

L = 30           # only used if method=="HMC"

#########################################
#####      Run MCMC-iFFBS code      #####
#########################################

N = 25000         # number of MCMC iterations
blockSize = 1000  # outputs will be saved every 'blockSize' iterations

## Create directory for the outputs (posterior samples and hidden states)
resultsDirectory = dataDirectory * modelFileName # where save Julia outputs
if !isdir(resultsDirectory)
    mkpath(resultsDirectory)
end
methodName = method == 1 ? "HMC" : "MH"
path = resultsDirectory * methodName * "/"
if !isdir(path)
    mkpath(path)
end

for i in 1:blockSize:N
    numFrom = i
    numTo = i+blockSize-1
    block = path * "Iters_from$numFrom" * "to$numTo" * "/"
    if !isdir(block)
        mkpath(block)
    end
end

## Load data (using RData.jl to read RDS files directly)
groupNames = RData.load(dataDirectory * "groupNames.rds")
TestMat_df = RData.load(dataDirectory * "TestMat.rds")
CaptHist = RData.load(dataDirectory * "CaptHist.rds")
CaptEffort = Array(RData.load(dataDirectory * "CaptEffort.rds"))
birthTimes = Int64.(RData.load(dataDirectory * "birthTimes.rds"))
startSamplingPeriod = Int64.(RData.load(dataDirectory * "startSamplingPeriod.rds"))
endSamplingPeriod = Int64.(RData.load(dataDirectory * "endSamplingPeriod.rds"))
capturesAfterMonit = Int64.(RData.load(dataDirectory * "capturesAfterMonit.rds"))
timeVec = RData.load(dataDirectory * "timeVec.rds")
dfTimes = RData.load(dataDirectory * "dfTimes.rds")

## Extract summaries from data
G = length(groupNames)
m = length(birthTimes)
maxt = size(CaptHist, 2)
# Extract test names before converting TestMat to matrix
testNames = names(TestMat_df)[4:end]
numTests = length(testNames)
numSeasons = 4

startingQuarter = 1  # because timeVec starts on Q1

## Creating Xinit (initial values for the hidden states)
Xinit = zeros(Float64, m, maxt)  # Use Float64 to allow NaN values

# Define tauInit early (needed for Xinit calculations)
tauInit = rand(Uniform(4, 16))  # Equivalent to R: tauInit <- runif(n = 1, 4, 16)

# Convert TestMat DataFrame to matrix upfront (like Rcpp does)
TestMat = Array(TestMat_df)  # Convert to matrix for all operations
# Replace missing values with NaN for numeric operations
TestMat = coalesce.(TestMat, NaN)
# TestMat = @zero_based TestMat  # Keep 1-based like R

# Apply 1-based indexing (keep everything like R)
Xinit = Float64.(Xinit)
# CaptHist = CaptHist  # Keep as is
# CaptEffort = Array(CaptEffort)  # Keep as is
# birthTimes = Int64.(birthTimes)  # Keep as is
# startSamplingPeriod = startSamplingPeriod  # Keep as is
# endSamplingPeriod = endSamplingPeriod  # Keep as is
# capturesAfterMonit = capturesAfterMonit  # Keep as is
# timeVec = timeVec  # Keep as is
# dfTimes = dfTimes  # Keep as is

# putting NAs before birth date 
for i in 1:m
    if birthTimes[i] > 1
        Xinit[i, 1:(birthTimes[i]-1)] .= NaN
    end
end

# putting infection whenever there's a positive result
for i in 1:m
    # Find rows for this individual (DataFrame indexing)
    individual_rows = findall(row -> row[1] == i, eachrow(TestMat))
    
    for tt in 1:maxt
        isInfec = zeros(Int, numTests)
        # Find rows for this time point by filtering TestMat directly
        time_rows = filter(row_idx -> TestMat[row_idx, 2] == tt, individual_rows)      
        
        if length(time_rows) > 0
            # Get test results for this time (columns 3:end are test results)
            test_results = hcat([TestMat[row, 3:end] for row in time_rows]...)
            
            if any(x -> x == 1, test_results)
                # infection starting at time of first positive test result
                earliest_pos_time = Int(minimum([TestMat[row, 2] for row in time_rows if !isnan(TestMat[row, 2]) && any(x -> x == 1, TestMat[row, 3:end])]))
                Xinit[i, 1:earliest_pos_time] .= 1.0  # Use Float64
                break
            end
        end
    end
end

# Assuming E becomes I some quarters later and forcing no E->S and I->E
for i in 1:m
    if any(x -> x == 3, Xinit[i, :])
        firstInfectedTime = findfirst(x -> x == 3, Xinit[i, :]) - 1
        if firstInfectedTime < maxt
            Xinit[i, firstInfectedTime+1:end] .= 3.0  # Use Float64
        end
        
        timefromEtoI = ceil(rand(Exponential(1/tauInit)))
        firstInfectiousTime = firstInfectedTime + Int(timefromEtoI)
        if firstInfectiousTime < maxt
            Xinit[i, firstInfectiousTime+1:end] .= 1.0  # Use Float64
        end
    end
end

# putting 9 after last capture dates
lastCaptureTimes = [findlast(x -> x == 1, CaptHist[i, :]) for i in 1:m]
for i in 1:m
    if lastCaptureTimes[i] !== nothing && lastCaptureTimes[i] < maxt
        Xinit[i, lastCaptureTimes[i]+1:end] .= 9.0  # Use Float64
    end
end

# putting NAs before the start of monitoring period
for i in 1:m
    if startSamplingPeriod[i] > 1
        Xinit[i, 1:(startSamplingPeriod[i]-1)] .= NaN
    end
end

# putting NAs after the end of monitoring period
for i in 1:m
    if endSamplingPeriod[i] < maxt
        Xinit[i, (endSamplingPeriod[i]+1):end] .= NaN
    end
end

for i in 1:m
    if all(isnan, Xinit[i, :])
        error("There are individuals with only NAs")
    end
    
    t0_i = startSamplingPeriod[i]
    valid_states = Xinit[i, t0_i:endSamplingPeriod[i]]
    if any(x -> !isnan(x) && x ∉ [0.0, 1.0, 3.0, 9.0], valid_states)
        error("individuals must have initial values 0,1,3,or 9 during their sampling time period")
    end
end

## Set Brock changepoint
# For matrices, we need to add a new column for Brock2
# Insert new column after column 3 (index 3 in 1-based)
new_col = fill(NaN, size(TestMat, 1))
TestMat = hcat(TestMat[:, 1:3], new_col, TestMat[:, 4:end])
testNames = ["Brock1", "Brock2", testNames[2:end]]
numTests = size(TestMat, 2) - 3  # Update numTests after adding Brock2 column

# initial Brock changepoint
hp_xi = [xi_mean, xi_sd]
xiInit = findfirst(x -> x >= 2000, timeVec) - 1
changePointBrock = xiInit

## This will be the initial TestMat
for irow in 1:size(TestMat, 1)
    if TestMat[irow, 2] >= changePointBrock  # column 2 is time in matrix (1-based)
        if !isnan(TestMat[irow, 4]) && TestMat[irow, 4] != 0  # column 4 is Brock1 (1-based)
            TestMat[irow, 5] = TestMat[irow, 4]  # Move Brock1 to Brock2
            TestMat[irow, 4] = NaN  # Set Brock1 to NaN
        end
    end
end

# put NAs in test results outside of the monitoring period
count = 0
for irow in 1:size(TestMat, 1)
    id_val = TestMat[irow, 1]  # column 1 is idNumber (1-based)
    time_val = TestMat[irow, 2]  # column 2 is time (1-based)
    
    # Handle NaN values - skip rows with missing id or time
    if isnan(id_val) || isnan(time_val)
        continue
    end
    
    id = Int(id_val)
    time = Int(time_val)
    
    if (time < startSamplingPeriod[id]) || (time > endSamplingPeriod[id])
        for col in 4:size(TestMat, 2)  # test result columns start at index 4 (1-based)
            TestMat[irow, col] = NaN
        end
        count += 1
    end
end

# checking which animals 
# -- were born before monitoring started in their groups
# -- after 1980

vec = DataFrame(id=Int[], firstGroup=Int[], birthTime=Float64[], startSamplingPeriod=Float64[])
ids = Int[]

for id in 1:m
    TestMat_i = TestMat_df[TestMat_df.idNumber .== id, :]

    time = TestMat_i.time[1]
    g    = TestMat_i.group[1]

    if birthTimes[id] < startSamplingPeriod[id] && birthTimes[id] > 1
        push!(ids, id)
        push!(vec, (id, g, birthTimes[id], startSamplingPeriod[id]))
    end
end

gs = sort(unique(vec.firstGroup))

starts = Int[]
starts_year = Any[]
for g_i in 1:length(gs)
    g = Int(gs[g_i])  # Convert group to integer
    effort_row = CaptEffort[g, :]
    ones_indices = findall(x -> x == 1, effort_row)
    first_one_idx = minimum(ones_indices)  # Equivalent to R's min(which(...)) 
    start_g = first_one_idx  # Keep 1-based for dfTimes lookup
    time_idx = findfirst(x -> x == start_g, dfTimes.idx)  # dfTimes is 1-based
    start_g_year = dfTimes.time[time_idx]
    push!(starts, start_g)
    push!(starts_year, start_g_year)
end

ord = sortperm(starts)
starts = starts[ord]
starts_year = starts_year[ord]
gs = gs[ord]
nuTimes = vcat([1], starts)  # Keep 1-based
numNuTimes = length(nuTimes)

TestMat_ = TestMat  # TestMat is already a matrix with 0-based indexing
CaptEffort_ = Matrix(CaptEffort)

thetaNames = ["theta$i" for i in 1:numTests]
rhoNames = ["rho$i" for i in 1:numTests]
phiNames = ["phi$i" for i in 1:numTests]
etaNames = ["eta$i" for i in 1:numSeasons]
parNames = vcat(
    ["alpha$i" for i in 1:G],
    "lambda", "beta", "q", "tau", 
    "a2", "b2", "c1", 
    ["nuE$i" for i in 0:numNuTimes-1],
    ["nuI$i" for i in 0:numNuTimes-1],
    "xi",
    thetaNames, rhoNames, phiNames, etaNames
)

socGroupSizes = fill(0, G)
for j in 1:G  # Use 1-based for groups
    # Find rows where group == j (column 3 is group)
    group_rows = findall(row -> row[3] == j, eachrow(TestMat))
    # Extract idNumber from those rows (column 1 is idNumber)
    ids_in_group = [TestMat[row, 1] for row in group_rows if !isnan(TestMat[row, 1])]
    socGroupSizes[j] = length(unique(ids_in_group))
end

K = median(socGroupSizes)

# hyperparameter values
hp_lambda = [1, lambda_b]
hp_beta = [1, beta_b]
hp_q = [1, 1]
hp_tau = [1, tau_b]
hp_a2 = [1, a2_b]
hp_b2 = [1, b2_b]
hp_c1 = [1, c1_b]
hp_nu = [1, 1, 1]
# hp_xi initialised above
hp_theta = [1, 1]
hp_rho = [1, 1]
hp_phi = [1, 1]
hp_eta = [1, 1]

hp = Dict(
    :hp_lambda => hp_lambda, 
    :hp_beta => hp_beta, 
    :hp_q => hp_q, 
    :hp_tau => hp_tau, 
    :hp_a2 => hp_a2, 
    :hp_b2 => hp_b2, 
    :hp_c1 => hp_c1, 
    :hp_nu => hp_nu, 
    :hp_xi => hp_xi,
    :hp_theta => hp_theta, 
    :hp_rho => hp_rho, 
    :hp_phi => hp_phi, 
    :hp_eta => hp_eta
)

############################################################################

# Using Julia code --------------------------------------

# Choosing initial parameter values from the prior -----------------------

# ## This is equivalent to using 
# initParamValues=Inf
# in which case Julia will generates initial values from the prior

lambdaInit = rand(Gamma(1, 1/100))
alphaInit = rand(Gamma(1, 1), G) # alphastar - needs G values, one per group
betaInit = rand(Gamma(1, 1/100))
qInit = rand(Beta(hp_q[1], hp_q[2]))
# tauInit  # now done above in order to be used in Xinit
a2Init = rand(Gamma(hp_a2[1], 1/100))
b2Init = rand(Gamma(hp_b2[1], 1/100))
c1Init = rand(Gamma(hp_c1[1], 1/100))

nuVecInit = rand(Dirichlet([8, 1, 1]), numNuTimes)
nuEInit = nuVecInit[2, :]  # Extract row 2 (should have length numNuTimes)
nuIInit = nuVecInit[3, :]  # Extract row 3 (should have length numNuTimes)
# xiInit was sampled above, and TestMat is constructed given xiInit
thetasInit = rand(Uniform(0.5, 1), numTests)
rhosInit = rand(Uniform(0.2, 0.8), numTests)
phisInit = rand(Uniform(0.7, 1), numTests)
etasInit = rand(Beta(hp_eta[1], hp_eta[2]), numSeasons)

initParamValues = vcat(
    alphaInit, lambdaInit, betaInit, qInit, tauInit, 
    a2Init, b2Init, c1Init, nuEInit, nuIInit, xiInit,
    thetasInit, rhosInit, phisInit, etasInit
)

# Debug parameter lengths
println("Debug: Parameter lengths:")
println("alphaInit: $(length(alphaInit)) (should be G=$G)")
println("lambdaInit: $(length(lambdaInit))")
println("betaInit: $(length(betaInit))")
println("qInit: $(length(qInit))")
println("tauInit: $(length(tauInit))")
println("a2Init: $(length(a2Init))")
println("b2Init: $(length(b2Init))")
println("c1Init: $(length(c1Init))")
println("nuEInit: $(length(nuEInit)) (should be numNuTimes=$numNuTimes)")
println("nuIInit: $(length(nuIInit)) (should be numNuTimes=$numNuTimes)")
println("xiInit: $(length(xiInit))")
println("thetasInit: $(length(thetasInit)) (should be numTests=$numTests)")
println("rhosInit: $(length(rhosInit)) (should be numTests=$numTests)")
println("phisInit: $(length(phisInit)) (should be numTests=$numTests)")
println("etasInit: $(length(etasInit)) (should be numSeasons=$numSeasons)")
println("Total initParamValues: $(length(initParamValues))")
expected_total = G + 1 + 1 + 1 + 1 + 1 + 1 + 1 + numNuTimes + numNuTimes + 1 + numTests + numTests + numTests + numSeasons
println("Expected total: $expected_total")
println("Match: $(length(initParamValues) == expected_total ? "✅" : "❌")")

# initParamValues = [Inf] # to generate initial values from the prior

############################################################################

## Include the MCMC-iFFBS function and helper functions

## Fit model
println("Starting MCMC-iFFBS algorithm...")

# Convert NaN to -1 for integer matrix (MCMC functions expect Int)
Xinit_int = copy(Xinit)
Xinit_int[isnan.(Xinit_int)] .= -1
initParamValues = Inf
out_ = MCMCiFFBS_(
    N, 
    initParamValues, 
    Matrix(Xinit_int), 
    Matrix(TestMat_),
    Matrix(CaptHist), 
    Vector(birthTimes),
    Vector(startSamplingPeriod),
    Vector(endSamplingPeriod),
    Vector(nuTimes),
    Matrix(CaptEffort_),
    Matrix(capturesAfterMonit),
    numSeasons, 
    startingQuarter,
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

# Convert to DataFrame with column names
results_df = DataFrame(out_, :auto)
rename!(results_df, parNames)

# Save results
println("Saving results...")
CSV.write(path * "mcmc_results.csv", results_df)

# Also save as RDS for compatibility with R analysis
RData.save(path * "mcmc_results.rds", Dict("results" => results_df, "parNames" => parNames))

println("MCMC-iFFBS completed successfully!")
println("Results saved to: $path")
println("Number of iterations: $N")
println("Number of parameters: $(length(parNames))")
println("Final parameter estimates:")
for (name, val) in zip(parNames, out_[end, :])
    println("  $name: $val")
end
