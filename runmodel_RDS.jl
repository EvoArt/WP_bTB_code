
using JLD2 
using Distributions
using Random
using DataFrames
using RData
using Statistics
using CSV
using Logging
include("julia/iFFBS.jl")
include("julia/dimension_corrections.jl")
include("julia/MCMCiFFBS_.jl")

seed = 2
Random.seed!(seed)

modelFileName = "outputs_seed$seed/"

dataDirectory = "WPbadgerData/"

method = 1  
epsilon = 0.01
epsilonalphas = 0.2
epsilonbq = 0.05
epsilontau = epsilon 
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

L = 30       


N = 25000         
blockSize = 1000 

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

G = length(groupNames)
m = length(birthTimes)
maxt = size(CaptHist, 2)

testNames = names(TestMat_df)[4:end]
numTests = length(testNames)
numSeasons = 4

startingQuarter = 1 

Xinit = zeros(Float64, m, maxt) 


tauInit = rand(Uniform(4, 16))

TestMat = Array(TestMat_df)  
TestMat = coalesce.(TestMat, NaN)

Xinit = Float64.(Xinit)

for i in 1:m
    if birthTimes[i] > 1
        Xinit[i, 1:(birthTimes[i]-1)] .= NaN
    end
end

for i in 1:m
    individual_rows = findall(row -> row[1] == i, eachrow(TestMat))
    
    for tt in 1:maxt
        time_rows = filter(row_idx -> TestMat[row_idx, 2] == tt, individual_rows)      
        
        if length(time_rows) > 0
            test_results = hcat([TestMat[row, 3:end] for row in time_rows]...)
            
            if any(x -> x == 1, test_results)
                infecTimesEarlier = 0
                tStartInfec = max(birthTimes[i]+1, startSamplingPeriod[i], tt-infecTimesEarlier, 1)
                Xinit[i, Int(tStartInfec)] = 3.0  
            end
        end
    end
end

for i in 1:m
    if any(x -> x == 3, Xinit[i, :])
        firstInfectedTime = findfirst(x -> x == 3, Xinit[i, :])
        if firstInfectedTime < maxt
            Xinit[i, firstInfectedTime:end] .= 3.0  
        end
        
        timefromEtoI = ceil(rand(Exponential(1/tauInit)))
        firstInfectiousTime = firstInfectedTime + Int(timefromEtoI)
        if firstInfectiousTime < maxt
            Xinit[i, firstInfectiousTime:end] .= 1.0 
        end
    end
end

lastCaptureTimes = [findlast(x -> x == 1, CaptHist[i, :]) for i in 1:m]
for i in 1:m
    if lastCaptureTimes[i] !== nothing && lastCaptureTimes[i] < maxt
        Xinit[i, lastCaptureTimes[i]+1:end] .= 9.0  # Use Float64
    end
end

for i in 1:m
    if startSamplingPeriod[i] > 1
        Xinit[i, 1:(startSamplingPeriod[i]-1)] .= NaN
    end
end

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

TestMat = hcat(TestMat[:, 1:4], new_col, TestMat[:, 5:end])
testNames = ["Brock1", "Brock2", testNames[2:end]]
numTests = size(TestMat, 2) - 3  

hp_xi = [xi_mean, xi_sd]
xiInit = findfirst(x -> x >= 2000, timeVec) 
changePointBrock = xiInit

for irow in 1:size(TestMat, 1)
    if TestMat[irow, 1] >= changePointBrock  
        if !isnan(TestMat[irow, 4]) 
            TestMat[irow, 5] = TestMat[irow, 4]  
            TestMat[irow, 4] = NaN 
        end
    end
end

count = 0
for irow in 1:size(TestMat, 1)
    id_val = TestMat[irow, 2]  
    time_val = TestMat[irow, 1]
    
    if isnan(id_val) || isnan(time_val)
        continue
    end
    
    id = Int(id_val)
    time = Int(time_val)
    
    if (time < startSamplingPeriod[id]) || (time > endSamplingPeriod[id])
        for col in 4:size(TestMat, 2) 
            TestMat[irow, col] = NaN
        end
        count += 1
    end
end

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
    g = Int(gs[g_i]) 
    effort_row = CaptEffort[g, :]
    ones_indices = findall(x -> x == 1, effort_row)
    first_one_idx = minimum(ones_indices)  
    start_g = first_one_idx  
    time_idx = findfirst(x -> x == start_g, dfTimes.idx)  
    start_g_year = dfTimes.time[time_idx]
    push!(starts, start_g)
    push!(starts_year, start_g_year)
end

ord = sortperm(starts)
starts = starts[ord]
starts_year = starts_year[ord]
gs = gs[ord]
nuTimes = vcat([1], starts) 
numNuTimes = length(nuTimes)

TestMat_ = TestMat 
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
for j in 1:G  
    group_rows = findall(row -> row[3] == j, eachrow(TestMat))
    ids_in_group = [TestMat[row, 1] for row in group_rows if !isnan(TestMat[row, 1])]
    socGroupSizes[j] = length(unique(ids_in_group))
end

K = median(socGroupSizes)

# hyper params
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


lambdaInit = rand(Gamma(1, 1/100))
alphaInit = rand(Gamma(1, 1)) 
betaInit = rand(Gamma(1, 1/100))
qInit = rand(Beta(hp_q[1], hp_q[2]))

a2Init = rand(Gamma(hp_a2[1], 1/100))
b2Init = rand(Gamma(hp_b2[1], 1/100))
c1Init = rand(Gamma(hp_c1[1], 1/100))

nuVecInit = rand(Dirichlet([8, 1, 1]), numNuTimes)
nuEInit = nuVecInit[2, :]  
nuIInit = nuVecInit[3, :]  

thetasInit = rand(Uniform(0.5, 1), numTests)
rhosInit = rand(Uniform(0.2, 0.8), numTests)
phisInit = rand(Uniform(0.7, 1), numTests)
etasInit = rand(Beta(hp_eta[1], hp_eta[2]), numSeasons)

initParamValues = vcat(
    alphaInit, lambdaInit, betaInit, qInit, tauInit, 
    a2Init, b2Init, c1Init, nuEInit, nuIInit, xiInit,
    thetasInit, rhosInit, phisInit, etasInit
)


println("Starting MCMC-iFFBS algorithm...")

Xinit_int = copy(Xinit)
#Xinit_int[isnan.(Xinit_int)] .= -1


initParamValues = vcat(
    alphaInit, lambdaInit, betaInit, qInit, tauInit, 
    a2Init, b2Init, c1Init, nuEInit, nuIInit, xiInit,
    thetasInit, rhosInit, phisInit, etasInit
)

out_ = MCMCiFFBS_(
    1000, 
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
