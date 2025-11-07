## Usage Instructions:

1. **Prerequisites**: Make sure you have the required packages installed:
   ```julia
   using Pkg
   Pkg.add(["OffsetArrays", "Distributions", "Random", "DataFrames", "RData", "Statistics", "CSV"])
   ```

2. **Data Files**: Ensure all data files are in the `WPbadgerData/` directory in RDS format (same as R version)

3. **Run the Model**:
   ```julia
   include("runmodel.jl")
   ```

4. **Results**: The model will save results to the specified output directory with both CSV and RDS formats for compatibility with R analysis

"""
    runmodel.jl

Julia version of the R script to run the MCMC-iFFBS model for badger TB transmission.
This script loads data, sets up initial conditions, and runs the MCMC algorithm.
"""

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

## Load data
groupNames = RData.load(dataDirectory * "groupNames.rds")
TestMat = RData.load(dataDirectory * "TestMat.rds")
CaptHist = RData.load(dataDirectory * "CaptHist.rds")
CaptEffort = RData.load(dataDirectory * "CaptEffort.rds")
birthTimes = RData.load(dataDirectory * "birthTimes.rds")
startSamplingPeriod = RData.load(dataDirectory * "startSamplingPeriod.rds")
endSamplingPeriod = RData.load(dataDirectory * "endSamplingPeriod.rds")
capturesAfterMonit = RData.load(dataDirectory * "capturesAfterMonit.rds")
timeVec = RData.load(dataDirectory * "timeVec.rds")
dfTimes = RData.load(dataDirectory * "dfTimes.rds")

## Extract summaries from data
G = length(groupNames)
m = length(birthTimes)
maxt = size(CaptHist, 2)
testNames = names(TestMat)[4:end]
numTests = length(testNames)
numSeasons = 4

startingQuarter = 1  # because timeVec starts on Q1

## Creating Xinit (initial values for the hidden states)
Xinit = zeros(Int, m, maxt)

# Apply 0-based indexing to all arrays
Xinit = @zero_based Xinit
TestMat = @zero_based TestMat
CaptHist = @zero_based CaptHist
CaptEffort = @zero_based CaptEffort
birthTimes = @zero_based birthTimes
startSamplingPeriod = @zero_based startSamplingPeriod
endSamplingPeriod = @zero_based endSamplingPeriod
capturesAfterMonit = @zero_based capturesAfterMonit
timeVec = @zero_based timeVec
dfTimes = @zero_based dfTimes

# putting NAs before birth date 
for i in 0:m-1
    if birthTimes[i+1] > 1
        Xinit[i+1, 1:birthTimes[i+1]] .= NaN
    end
end

# putting infection whenever there's a positive result
for i in 0:m-1
    df_i = filter(row -> row.idNumber == i+1, TestMat)
    for tt in 0:maxt-1
        isInfec = zeros(Int, numTests)
        if tt in df_i.time
            df_i_tt = filter(row -> row.time == tt, df_i)[:, 4:end]
            
            df_i_alltt = df_i[:, 4:end]
            
            if any(x -> x == 1, df_i_tt)
                # infection starting at time of first positive test result
                infecTimesEarlier = 0
                tStartInfec = max(birthTimes[i+1]+1, startSamplingPeriod[i+1], tt-infecTimesEarlier, 1)
                Xinit[i+1, tStartInfec+1] = 3 # infection starting at infecTimesEarlier
            end
        end
    end
end

# tauInit <- rgamma(n = 1, shape = 1, rate = 0.1)
tauInit = rand(Uniform(4, 16))

# Assuming E becomes I some quarters later and forcing no E->S and I->E
for i in 0:m-1
    if any(x -> x == 3, Xinit[i+1, :])
        firstInfectedTime = findfirst(x -> x == 3, Xinit[i+1, :]) - 1
        if firstInfectedTime < maxt-1
            Xinit[i+1, firstInfectedTime+1:end] .= 3
        end
        
        timefromEtoI = ceil(rand(Exponential(1/tauInit)))
        firstInfectiousTime = firstInfectedTime + Int(timefromEtoI)
        if firstInfectiousTime < maxt-1
            Xinit[i+1, firstInfectiousTime+1:end] .= 1
        end
    end
end

# putting 9 after last capture dates
lastCaptureTimes = [findlast(x -> x == 1, CaptHist[i+1, :]) - 1 for i in 0:m-1]
for i in 0:m-1
    if lastCaptureTimes[i+1] < maxt-1
        Xinit[i+1, lastCaptureTimes[i+1]+2:end] .= 9
    end
end

# putting NAs before the start of monitoring period
for i in 0:m-1
    if startSamplingPeriod[i+1] > 1
        Xinit[i+1, 1:startSamplingPeriod[i+1]] .= NaN
    end
end

# putting NAs after the end of monitoring period
for i in 0:m-1
    if endSamplingPeriod[i+1] < maxt
        Xinit[i+1, endSamplingPeriod[i+1]+2:end] .= NaN
    end
end

for i in 0:m-1
    if all(isnan, Xinit[i+1, :])
        error("There are individuals with only NAs")
    end
    
    t0_i = startSamplingPeriod[i+1]
    valid_states = Xinit[i+1, t0_i+1:endSamplingPeriod[i+1]+1]
    if any(x -> !isnan(x) && x ∉ [0, 1, 3, 9], valid_states)
        error("individuals must have initial values 0,1,3,or 9 during their sampling time period")
    end
end

## Set Brock changepoint
# Rename column 4 to "Brock1"
rename!(TestMat, names(TestMat)[4] => "Brock1")
TestMat.Brock2 = fill(NaN, nrow(TestMat)) # creating a column for Brock2
# Reorder columns
col_order = [names(TestMat)[1:4]; "Brock2"; names(TestMat)[5:end-1]]
TestMat = TestMat[:, col_order]
testNames = ["Brock1", "Brock2", testNames[2:end]]

numTests = size(TestMat, 2) - 3 # updating numTests

# initial Brock changepoint
hp_xi = [xi_mean, xi_sd]
xiInit = findfirst(x -> x >= 2000, timeVec) - 1
changePointBrock = xiInit

## This will be the initial TestMat
for irow in 1:nrow(TestMat)
    if TestMat[irow, :time] >= changePointBrock
        if !ismissing(TestMat[irow, :Brock1]) && !isnan(TestMat[irow, :Brock1])
            TestMat[irow, :Brock2] = TestMat[irow, :Brock1]
            TestMat[irow, :Brock1] = NaN
        end
    end
end

# put NAs in test results outside of the monitoring period
count = 0
for irow in 1:nrow(TestMat)
    id = TestMat[irow, :idNumber]
    time = TestMat[irow, :time]
    
    if (time < startSamplingPeriod[id+1]) || (time > endSamplingPeriod[id+1])
        for col in 4:size(TestMat, 2)
            TestMat[irow, col] = NaN
        end
        count += 1
    end
end

# checking which animals 
# -- were born before monitoring started in their groups
# -- after 1980

vec = Matrix{Float64}(undef, 0, 4)
ids = Int[]
for id in 0:m-1
    TestMat_i = filter(row -> row.idNumber == id+1, TestMat)
    time = TestMat_i[1, :time] # time of first captured
    g = TestMat_i[1, :group] 
    if (birthTimes[id+1] < startSamplingPeriod[id+1]) && (birthTimes[id+1] > 1)
        push!(ids, id+1)
        vec_id = [id+1, g, birthTimes[id+1], startSamplingPeriod[id+1]] 
        vec = [vec; vec_id']
    end
end

vec_df = DataFrame(
    id = ids,
    firstGroup = vec[:, 2],
    birthTime = vec[:, 3],
    startSamplingPeriod = vec[:, 4]
)

gs = sort(unique(vec_df.firstGroup))

starts = Int[]
starts_year = Int[]
for g_i in 1:length(gs)
    g = gs[g_i]
    start_g = findfirst(x -> x == 1, CaptEffort[g+1, :]) - 1
    start_g_year = dfTimes.time[findfirst(x -> x == start_g+1, dfTimes.idx)]
    push!(starts, start_g)
    push!(starts_year, start_g_year)
end

ord = sortperm(starts)
starts = starts[ord]
starts_year = starts_year[ord]
gs = gs[ord]
nuTimes = vcat([1], starts)
numNuTimes = length(nuTimes)

TestMat_ = Matrix(TestMat) # using a matrix to use in Julia code
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
for j in 0:G-1
    socGroupSizes[j+1] = length(unique(filter(row -> row.group == j+1, TestMat).idNumber))
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
alphaInit = rand(Gamma(1, 1)) # alphastar
betaInit = rand(Gamma(1, 1/100))
qInit = rand(Beta(hp_q[1], hp_q[2]))
# tauInit  # now done above in order to be used in Xinit
a2Init = rand(Gamma(hp_a2[1], 1/100))
b2Init = rand(Gamma(hp_b2[1], 1/100))
c1Init = rand(Gamma(hp_c1[1], 1/100))

# Dirichlet for nu parameters
nuVecInit = rand(Dirichlet([8, 1, 1]), numNuTimes)
nuEInit = nuVecInit[:, 2]
nuIInit = nuVecInit[:, 3]
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

# initParamValues = [Inf] # to generate initial values from the prior

############################################################################

## Include the MCMC-iFFBS function and helper functions
include("MCMCiFFBS_.jl")
include("helper_functions.jl")

## Fit model
println("Starting MCMC-iFFBS algorithm...")
out_ = MCMCiFFBS_(
    N=N, 
    initParamValues=initParamValues, 
    Xinit=Matrix{Int}(Xinit), 
    TestMat=Matrix{Int}(TestMat_),
    CaptHist=Matrix{Int}(CaptHist), 
    birthTimes=Vector{Int}(birthTimes),
    startSamplingPeriod=Vector{Int}(startSamplingPeriod),
    endSamplingPeriod=Vector{Int}(endSamplingPeriod),
    nuTimes=Vector{Int}(nuTimes),
    CaptEffort=Matrix{Int}(CaptEffort_),
    capturesAfterMonit=Matrix{Int}(capturesAfterMonit),
    numSeasons=numSeasons, 
    seasonStart=startingQuarter,
    maxt=maxt,
    hp_lambda=hp_lambda, 
    hp_beta=hp_beta, 
    hp_q=hp_q, 
    hp_tau=hp_tau, 
    hp_a2=hp_a2, 
    hp_b2=hp_b2, 
    hp_c1=hp_c1, 
    hp_nu=hp_nu, 
    hp_xi=hp_xi,
    hp_theta=hp_theta, 
    hp_rho=hp_rho,
    hp_phi=hp_phi, 
    hp_eta=hp_eta, 
    k=k, 
    K=K,
    sd_xi_min=sd_xi_min,
    method=method, 
    epsilon=epsilon, 
    epsilonalphas=epsilonalphas, 
    epsilonbq=epsilonbq, 
    epsilontau=epsilontau,
    epsilonc1=epsilonc1, 
    epsilonsens=epsilonsens,
    L=L, 
    path=path, 
    blockSize=blockSize
)

# Convert to DataFrame with column names
results_df = DataFrame(out_, :auto)
rename!(results_df, parNames)

# Save results
println("Saving results...")
CSV.write(path * "mcmc_results.csv", results_df)

# Also save as RDS for compatibility with R analysis
using RData
RData.save(path * "mcmc_results.rds", Dict("results" => results_df, "parNames" => parNames))

println("MCMC-iFFBS completed successfully!")
println("Results saved to: $path")
println("Number of iterations: $N")
println("Number of parameters: $(length(parNames))")
println("Final parameter estimates:")
for (name, val) in zip(parNames, out_[end, :])
    println("  $name: $val")
end
