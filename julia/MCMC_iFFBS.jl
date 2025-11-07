using Distributions
using Random
using LinearAlgebra
using OffsetArrays: Origin
using SpecialFunctions

# Macro for zero-based indexing
macro zero_based(x)
    return :(Origin(0)($x))
end

"""
Performs MCMC-iFFBS for CMR-Gompertz-SEID-rho model with group-specific alpha

# Arguments
- `N`: Number of MCMC iterations
- `Xinit`: Matrix of dimension (m × maxt) containing initial states
- `TestMat`: Matrix with all capture events (columns: time, id, group, test1, test2, ...)
- `CaptHist`: Matrix of dimension (m × maxt) containing capture history
- `birthTimes`: Integer vector of length m representing birth times
- `startSamplingPeriod`: Integer vector of length m with start sampling times
- `endSamplingPeriod`: Integer vector of length m with end sampling times
- `nuTimes`: Integer vector saying at which times nu parameters are applied
- `CaptEffort`: (G × maxt) matrix with 1 indicates monitoring, 0 otherwise
- `capturesAfterMonit`: Matrix containing last capture times for individuals captured after monitoring
- `numSeasons`: Number of seasons
- `seasonStart`: Starting season
- `maxt`: Number of time points
- `hp_lambda`: Hyperparameter values for Gamma prior on mean of alpha
- `hp_beta`: Hyperparameter values for Gamma prior on frequency-dependent transmission
- `hp_q`: Hyperparameter values for Gamma prior on q
- `hp_tau`: Hyperparameter values for Gamma prior on average latent period
- `hp_a2`: Hyperparameter values for Gamma prior on Gompertz parameter a2
- `hp_b2`: Hyperparameter values for Gamma prior on Gompertz parameter b2
- `hp_c1`: Hyperparameter values for Gamma prior on Gompertz parameter c1
- `hp_nu`: Hyperparameter values for Dirichlet prior on initial infection probabilities
- `hp_xi`: Hyperparameter values for prior on Brock changepoint
- `hp_theta`: Hyperparameter values for Beta prior on test sensitivities
- `hp_rho`: Hyperparameter values for Beta prior on scaling factor
- `hp_phi`: Hyperparameter values for Beta prior on test specificities
- `hp_eta`: Hyperparameter values for Beta prior on capture probabilities
- `k`: Positive integer shape parameter of Gamma distribution for latent period
- `K`: Rescaling parameter for population size
- `sd_xi_min`: Minimum value for proposal standard deviation
- `method`: Integer indicating update method (1=HMC, 2=RWMH)
- `epsilon`: Leapfrog stepsize in HMC
- `epsilonalphas`: Leapfrog stepsize in HMC for alphas
- `epsilonbq`: Leapfrog stepsize in HMC for beta and q
- `epsilontau`: Leapfrog stepsize in HMC for tau
- `epsilonc1`: Leapfrog stepsize in HMC for c1
- `epsilonsens`: Leapfrog stepsize in HMC for thetas and rhos
- `L`: Number of leapfrog steps in HMC
- `path`: Directory where results will be saved
- `blockSize`: Number of iterations saved in each folder
- `initParamValues`: Initial parameter values (or Inf for prior sampling)

# Returns
Matrix containing MCMC draws (N rows) from posterior distribution of all parameters

# Details
- Test results: 0=negative, 1=positive, NA=not observed
- Capture history: 0=not captured, 1=captured
- States: NA=not born, 0=susceptible, 3=exposed, 1=infectious, 9=dead
"""
function MCMCiFFBS(
    N::Int,
    initParamValues::Vector{Float64},
    Xinit::Matrix{Int},
    TestMat::Matrix{Int},
    CaptHist::Matrix{Int},
    birthTimes::Vector{Int},
    startSamplingPeriod::Vector{Int},
    endSamplingPeriod::Vector{Int},
    nuTimes::Vector{Int},
    CaptEffort::Matrix{Int},
    capturesAfterMonit::Matrix{Int},
    numSeasons::Int,
    seasonStart::Int,
    maxt::Int,
    hp_lambda::Vector{Float64},
    hp_beta::Vector{Float64},
    hp_q::Vector{Float64},
    hp_tau::Vector{Float64},
    hp_a2::Vector{Float64},
    hp_b2::Vector{Float64},
    hp_c1::Vector{Float64},
    hp_nu::Vector{Float64},
    hp_xi::Vector{Float64},
    hp_theta::Vector{Float64},
    hp_rho::Vector{Float64},
    hp_phi::Vector{Float64},
    hp_eta::Vector{Float64},
    k::Int,
    K::Float64,
    sd_xi_min::Float64,
    method::Int,
    epsilon::Float64,
    epsilonalphas::Float64,
    epsilonbq::Float64,
    epsilontau::Float64,
    epsilonc1::Float64,
    epsilonsens::Float64,
    L::Int,
    path::String,
    blockSize::Int
)
    
    # Check method
    if !(method == 1 || method == 2)
        error("Please use either method=1 ('HMC') or method=2 ('RWMH').")
    end
    
    m = size(CaptHist, 1)
    numTests = size(TestMat, 2) - 3
    G = maximum(TestMat[:, 3])
    
    # Input validation
    @assert size(Xinit, 1) == m "Xinit and CaptHist must include same number of individuals"
    @assert length(birthTimes) == m "birthTimes and CaptHist must include same number of individuals"
    @assert size(CaptHist, 2) == size(Xinit, 2) "CaptHist and Xinit must include same number of time points"
    
    numNuTimes = length(nuTimes)
    nParsNotGibbs = G + 4 + 3 + 2*numNuTimes + 1
    # G alphas, lambda, beta, q, tau, survival rates, init probs, Brock changepoint
    
    nParsThetasRhos = 2 * numTests
    
    # Validate hyperparameters
    @assert length(hp_lambda) == 2 "hp_lambda must be vector of 2 hyperparameters"
    @assert length(hp_beta) == 2 "hp_beta must be vector of 2 hyperparameters"
    @assert length(hp_q) == 2 "hp_q must be vector of 2 hyperparameters"
    @assert length(hp_tau) == 2 "hp_tau must be vector of 2 hyperparameters"
    @assert length(hp_a2) == 2 "hp_a2 must be vector of 2 hyperparameters"
    @assert length(hp_b2) == 2 "hp_b2 must be vector of 2 hyperparameters"
    @assert length(hp_c1) == 2 "hp_c1 must be vector of 2 hyperparameters"
    @assert length(hp_nu) == 3 "hp_nu must be vector of 3 hyperparameters"
    @assert length(hp_xi) == 2 "hp_xi must be vector of 2 hyperparameters"
    @assert length(hp_theta) == 2 "hp_theta must be vector of 2 hyperparameters"
    @assert length(hp_rho) == 2 "hp_rho must be vector of 2 hyperparameters"
    @assert length(hp_phi) == 2 "hp_phi must be vector of 2 hyperparameters"
    @assert length(hp_eta) == 2 "hp_eta must be vector of 2 hyperparameters"
    
    # Initialize parameters
    if all(isfinite.(initParamValues))
        expected_length = (nParsNotGibbs - G + 1) + 3*numTests + numSeasons
        @assert length(initParamValues) == expected_length "initParamValues has incorrect length"
        
        println("Initial parameter values supplied by user:")
        alphaStarInit = initParamValues[1]
        lambdaInit = initParamValues[2]
        betaInit = initParamValues[3]
        qInit = initParamValues[4]
        tauInit = initParamValues[5]
        a2Init = initParamValues[6]
        b2Init = initParamValues[7]
        c1Init = initParamValues[8]
        
        nuEInit = initParamValues[9:(8+numNuTimes)]
        nuIInit = initParamValues[(9+numNuTimes):(8+2*numNuTimes)]
        xiInit = Int(initParamValues[9+2*numNuTimes])
        
        @assert 1 <= xiInit <= maxt-1 "Initial value for Brock changepoint outside study period"
        
        thetasInit = initParamValues[(nParsNotGibbs-G+2):(nParsNotGibbs-G+1+numTests)]
        rhosInit = initParamValues[(nParsNotGibbs-G+2+numTests):(nParsNotGibbs-G+1+2*numTests)]
        phisInit = initParamValues[(nParsNotGibbs-G+2+2*numTests):(nParsNotGibbs-G+1+3*numTests)]
        etasInit = initParamValues[(nParsNotGibbs-G+2+3*numTests):end]
        
    else
        println("Initial parameter values generated from prior:")
        
        lambdaInit = rand(Gamma(hp_lambda[1], 1/hp_lambda[2]))
        alphaStarInit = rand(Gamma(1.0, 1.0))
        betaInit = rand(Gamma(hp_beta[1], 1/hp_beta[2]))
        qInit = rand(Beta(hp_q[1], hp_q[2]))
        tauInit = rand(Gamma(hp_tau[1], 1/hp_tau[2]))
        a2Init = rand(Gamma(hp_a2[1], 1/hp_a2[2]))
        b2Init = rand(Gamma(hp_b2[1], 1/hp_b2[2]))
        c1Init = rand(Gamma(hp_c1[1], 1/hp_c1[2]))
        
        nuEInit = zeros(numNuTimes)
        nuIInit = zeros(numNuTimes)
        for i_nu in 1:numNuTimes
            samp_nuInit = rand(Dirichlet(hp_nu))
            nuEInit[i_nu] = samp_nuInit[2]
            nuIInit[i_nu] = samp_nuInit[3]
        end
        
        xiInit = round(Int, rand(Normal(hp_xi[1], hp_xi[2])))
        count = 0
        while xiInit < 1 || xiInit > maxt-1
            count += 1
            xiInit = round(Int, rand(Normal(hp_xi[1], hp_xi[2])))
            if count > 100
                error("Use better prior for xi. More than 100 initial values drawn, none in study period")
            end
        end
        
        thetasInit = [rand(Beta(hp_theta[1], hp_theta[2])) for _ in 1:numTests]
        rhosInit = [rand(Beta(hp_rho[1], hp_rho[2])) for _ in 1:numTests]
        phisInit = [rand(Beta(hp_phi[1], hp_phi[2])) for _ in 1:numTests]
        etasInit = [rand(Beta(hp_eta[1], hp_eta[2])) for _ in 1:numSeasons]
    end
    
    # Print initial values
    println("alphaStar = $alphaStarInit")
    println("lambda = $lambdaInit")
    println("alpha = $(alphaStarInit * lambdaInit)")
    println("beta = $betaInit")
    println("q = $qInit")
    println("tau = $tauInit")
    println("a2 = $a2Init")
    println("b2 = $b2Init")
    println("c1 = $c1Init")
    println("nuEs = $nuEInit")
    println("nuIs = $nuIInit")
    println("xi = $xiInit")
    println("thetas = $thetasInit")
    println("rhos = $rhosInit")
    println("phis = $phisInit")
    println("etas = $etasInit")
    
    # Initialize storage matrices
    nPars = nParsNotGibbs + 3*numTests + numSeasons
    out = zeros(N, nPars)
    
    logPostPerIter = zeros(N)
    logLikPerIter = zeros(N)
    
    nSus = zeros(Int, maxt, N)
    nExp = zeros(Int, maxt, N)
    nInf = zeros(Int, maxt, N)
    
    nSusTested = zeros(Int, maxt, blockSize, numTests)
    nExpTested = zeros(Int, maxt, blockSize, numTests)
    nInfTested = zeros(Int, maxt, blockSize, numTests)
    
    # Per social group storage (using arrays of arrays)
    nExpTestedPerGroup = [zeros(Int, maxt, blockSize, numTests) for _ in 1:G]
    nInfTestedPerGroup = [zeros(Int, maxt, blockSize, numTests) for _ in 1:G]
    nSusTestedPerGroup = [zeros(Int, maxt, blockSize, numTests) for _ in 1:G]
    
    nSusByGroup = zeros(Int, G, maxt, blockSize)
    nExpByGroup = zeros(Int, G, maxt, blockSize)
    nInfByGroup = zeros(Int, G, maxt, blockSize)
    
    infTimes = fill(-10, m, blockSize)
    infectivityTimes = fill(-10, m, blockSize)
    deathTimes = fill(-10, m, blockSize)
    
    AcontribPopTime = zeros(maxt, blockSize)
    AcontribPop = zeros(blockSize)
    AcontribGroupTime = zeros(G, maxt, blockSize)
    AcontribGroup = zeros(G, blockSize)
    AcontribIndivGroupTime = zeros(m, G, maxt)
    
    # Initialize parameters on transformed scale
    nParsBlock1 = nParsNotGibbs - 2*numNuTimes - 1
    pars = zeros(nParsBlock1)
    alpha_js = zeros(G)
    
    # Helper functions for logit/logistic transforms
    logitD(x) = log(x / (1 - x))
    logisticD(x) = 1 / (1 + exp(-x))
    
    for g in 1:G
        pars[g] = log(alphaStarInit)
        alpha_js[g] = alphaStarInit * lambdaInit
    end
    pars[G+1] = log(lambdaInit)
    pars[G+2] = log(betaInit)
    pars[G+3] = logitD(qInit)
    pars[G+4] = log(tauInit)
    pars[G+5] = log(a2Init)
    pars[G+6] = log(b2Init)
    pars[G+7] = log(c1Init)
    
    lambda = lambdaInit
    beta = betaInit
    q = qInit
    ql = logitD(qInit)
    tau = tauInit
    a2 = a2Init
    b2 = b2Init
    c1 = c1Init
    
    nuEs = copy(nuEInit)
    nuIs = copy(nuIInit)
    xi = xiInit
    thetas = copy(thetasInit)
    rhos = copy(rhosInit)
    phis = copy(phisInit)
    etas = copy(etasInit)
    
    # Make season vector
    seasonVec = MakeSeasonVec(numSeasons, seasonStart, maxt)
    
    # Initialize X
    X = copy(Xinit)
    
    # Calculate last capture times
    lastCaptureTimes = zeros(Int, m)
    for i in 1:m
        capt_hist_i = CaptHist[i, :]
        whichCapt = findall(capt_hist_i .== 1)
        if !isempty(whichCapt)
            lastCaptureTimes[i] = maximum(whichCapt)
        else
            lastCaptureTimes[i] = birthTimes[i]
        end
    end
    
    # Covariance matrices for RWMH
    Sigma = Matrix{Float64}(I, nParsBlock1, nParsBlock1) * 0.1
    Sigma2 = Matrix{Float64}(I, 2*numTests, 2*numTests) * 0.01
    
    sd_xi = 2.0
    count_accept_sd_xi = 0
    
    thetas_rhos = zeros(2 * numTests)
    
    # Calculate age matrix
    ageMat = fill(-10, m, maxt)
    for i in 1:m
        mint_i = max(1, birthTimes[i])
        for tt in mint_i:maxt
            ageMat[i, tt] = tt - birthTimes[i] + 1
        end
    end
    
    # Locate individuals in social groups
    SocGroup = LocateIndiv(TestMat, birthTimes)
    
    # Adjust capture history for monitoring effort
    CaptHistUsed = copy(CaptHist)
    for ii in 1:m
        for tt in 1:maxt
            g = SocGroup[ii, tt]
            if g == 0 || g == -1
                CaptHistUsed[ii, tt] = 0
            else
                CaptHistUsed[ii, tt] = CaptHistUsed[ii, tt] * CaptEffort[g, tt]
            end
        end
    end
    
    # Initialize numInfecMat (infection counts per group, excluding first individual)
    numInfecMat = @zero_based zeros(Int, G, maxt-1)
    for tt in 0:(maxt-2)
        for ii in 1:(m-1)  # Excluding first individual
            if X[ii+1, tt+1] == 1
                g_i_tt = SocGroup[ii+1, tt+1]
                if g_i_tt != 0
                    numInfecMat[g_i_tt-1, tt] += 1
                end
            end
        end
    end
    
    # Initialize mPerGroup (number of individuals per group, excluding first)
    mPerGroup = @zero_based zeros(Int, G, maxt)
    for tt in 0:(maxt-1)
        for ii in 1:(m-1)
            if X[ii+1, tt+1] in [0, 1, 3]
                g_i_tt = SocGroup[ii+1, tt+1]
                if g_i_tt != 0
                    mPerGroup[g_i_tt-1, tt] += 1
                end
            end
        end
    end
    
    # Create test field structures
    TestField = TestMatAsField(TestMat, m)
    TestFieldProposal = copy(TestField)
    TestTimes = TestTimesField(TestMat, m)
    
    idVecAll = collect(0:(m-1))
    corrector = zeros(maxt, 4)
    predProb = zeros(maxt, 4)
    filtProb = zeros(maxt, 4)
    
    # Calculate which individuals require update
    whichRequireUpdate = Vector{Vector{Int}}(undef, (m-1)*(maxt-1))
    count = 1
    for i in 0:(m-2)
        id = i
        idNext = i + 1
        for tt in 0:(maxt-2)
            idx = Int[]
            for jj in 1:(m-1)
                if ((jj != id) && (jj != idNext)) &&
                   (SocGroup[jj+1, tt+1] != 0) &&
                   (SocGroup[jj+1, tt+1] == SocGroup[id+1, tt+1] ||
                    SocGroup[jj+1, tt+1] == SocGroup[idNext+1, tt+1])
                    push!(idx, jj)
                end
            end
            whichRequireUpdate[count] = idx
            count += 1
        end
    end
    
    iterSub = 1
    
    # Main MCMC loop
    for iter in 1:N
        
        if iter > 1 && (iter % 1000 == 0)
            println("iter: $iter out of N=$N")
        end
        
        # Update parameters from transformed scale
        lambda = exp(pars[G+1])
        for g in 1:G
            alpha_js[g] = exp(pars[g]) * lambda
        end
        beta = exp(pars[G+2])
        q = logisticD(pars[G+3])
        ql = pars[G+3]
        tau = exp(pars[G+4])
        a2 = exp(pars[G+5])
        b2 = exp(pars[G+6])
        c1 = exp(pars[G+7])
        
        sumLogCorrector = 0.0
        
        # Update death probabilities using Gompertz parameters
        probDyingMat = fill(-10.0, m, maxt)
        LogProbDyingMat = similar(probDyingMat)
        LogProbSurvMat = similar(probDyingMat)
        
        for i in 1:m
            for tt in 1:maxt
                if ageMat[i, tt] > 0
                    if tt > lastCaptureTimes[i]
                        age_i_tt = Float64(ageMat[i, tt])
                        condProbDeath = TrProbDeath(age_i_tt, a2, b2, c1, false)
                        probDyingMat[i, tt] = condProbDeath
                        LogProbDyingMat[i, tt] = log(condProbDeath)
                        LogProbSurvMat[i, tt] = TrProbSurvive(age_i_tt, a2, b2, c1, true)
                    else
                        probDyingMat[i, tt] = 0.0
                        LogProbDyingMat[i, tt] = log(0.0)
                        LogProbSurvMat[i, tt] = log(1.0)
                    end
                end
            end
        end
        
        # Calculate E->E and E->I probabilities
        logProbEtoE = log(1.0 - ErlangCDF(1, k, tau/k))
        logProbEtoI = log(ErlangCDF(1, k, tau/k))
        
        # Update transition probabilities using new infection rates
        logProbStoSgivenSorE = @zero_based zeros(G, maxt-1)
        logProbStoEgivenSorE = @zero_based zeros(G, maxt-1)
        logProbStoSgivenI = @zero_based zeros(G, maxt-1)
        logProbStoEgivenI = @zero_based zeros(G, maxt-1)
        logProbStoSgivenD = @zero_based zeros(G, maxt-1)
        logProbStoEgivenD = @zero_based zeros(G, maxt-1)
        
        for tt in 0:(maxt-2)
            for g in 0:(G-1)
                mgt = mPerGroup[g, tt]
                
                if SocGroup[1, tt+1] == g+1
                    # If 1st individual is in this group and alive
                    inf_mgt = numInfecMat[g, tt] / ((mgt+1.0)/K)^q
                    logProbStoSgivenSorE[g, tt] = -alpha_js[g+1] - beta*inf_mgt
                    logProbStoEgivenSorE[g, tt] = log1mexp(alpha_js[g+1] + beta*inf_mgt)
                    
                    # If 1st individual is infectious
                    inf_mgt = (numInfecMat[g, tt]+1.0) / ((mgt+1.0)/K)^q
                    logProbStoSgivenI[g, tt] = -alpha_js[g+1] - beta*inf_mgt
                    logProbStoEgivenI[g, tt] = log1mexp(alpha_js[g+1] + beta*inf_mgt)
                    
                    # If 1st individual is dead
                    inf_mgt = numInfecMat[g, tt] / (mgt/K)^q
                    logProbStoSgivenD[g, tt] = -alpha_js[g+1] - beta*inf_mgt
                    logProbStoEgivenD[g, tt] = log1mexp(alpha_js[g+1] + beta*inf_mgt)
                else
                    # 1st individual not in this group
                    inf_mgt = numInfecMat[g, tt] / (mgt/K)^q
                    FOI = alpha_js[g+1] + beta*inf_mgt
                    log1mexpFOI = log1mexp(FOI)
                    
                    logProbStoSgivenSorE[g, tt] = -FOI
                    logProbStoEgivenSorE[g, tt] = log1mexpFOI
                    logProbStoSgivenI[g, tt] = -FOI
                    logProbStoEgivenI[g, tt] = log1mexpFOI
                    logProbStoSgivenD[g, tt] = -FOI
                    logProbStoEgivenD[g, tt] = log1mexpFOI
                end
            end
        end
        
        # Calculate logProbRest for all individuals except first
        logProbRest = @zero_based zeros(maxt-1, 4, m)
        for jj in 1:(m-1)
            for tt in 0:(maxt-2)
                if X[jj+1, tt+1] in [0, 1, 3]
                    iFFBScalcLogProbRest!(
                        jj, tt, logProbRest, X, SocGroup,
                        LogProbDyingMat, LogProbSurvMat,
                        logProbStoSgivenSorE, logProbStoEgivenSorE,
                        logProbStoSgivenI, logProbStoEgivenI,
                        logProbStoSgivenD, logProbStoEgivenD,
                        logProbEtoE, logProbEtoI
                    )
                end
            end
        end
        
        # Sum up logTransProbRest
        logTransProbRest = @zero_based zeros(maxt-1, 4)
        for jj in 1:(m-1)
            for tt in 0:(maxt-2)
                for s in 0:3
                    logTransProbRest[tt, s] += logProbRest[tt, s, jj]
                end
            end
        end
        
        # Run iFFBS for each individual
        for jj in 0:(m-1)
            iFFBS!(
                alpha_js, beta, q, tau, k, K,
                probDyingMat, LogProbDyingMat, LogProbSurvMat,
                logProbRest, nuTimes, nuEs, nuIs,
                thetas, rhos, phis, etas,
                jj+1, birthTimes[jj+1], startSamplingPeriod[jj+1], endSamplingPeriod[jj+1],
                X, seasonVec, TestField[jj+1], TestTimes[jj+1],
                CaptHist, corrector, predProb, filtProb,
                logTransProbRest, numInfecMat, SocGroup, mPerGroup,
                idVecAll, logProbStoSgivenSorE, logProbStoEgivenSorE,
                logProbStoSgivenI, logProbStoEgivenI,
                logProbStoSgivenD, logProbStoEgivenD,
                logProbEtoE, logProbEtoI,
                whichRequireUpdate, sumLogCorrector
            )
        end
        
        # Calculate last observed alive times
        lastObsAliveTimes = zeros(Int, m)
        for jj in 1:m
            which_deadTimes = findall(X[jj, :] .== 9)
            if !isempty(which_deadTimes)
                lastObsAliveTimes[jj] = minimum(which_deadTimes)
            else
                lastObsAliveTimes[jj] = endSamplingPeriod[jj]
            end
        end
        
        # Update total infection counts (including first individual)
        totalNumInfec = copy(numInfecMat)
        for tt in 0:(maxt-2)
            if X[1, tt+1] == 1
                g = SocGroup[1, tt+1]
                totalNumInfec[g-1, tt] += 1
            end
        end
        
        # Similarly for totalmPerGroup
        totalmPerGroup = copy(mPerGroup)
        for tt in 0:(maxt-1)
            if X[1, tt+1] in [0, 1, 3]
                g = SocGroup[1, tt+1]
                totalmPerGroup[g-1, tt] += 1
            end
        end
        
        # Update parameters using HMC or RWMH
        if method == 1
            pars = HMC_2!(
                pars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
                ageMat, epsilon, epsilonalphas, epsilonbq, epsilontau, epsilonc1,
                nParsBlock1, L, hp_lambda, hp_beta, hp_q, hp_tau,
                hp_a2, hp_b2, hp_c1, k, K
            )
        elseif method == 2
            # Adapt covariance every 100 iterations
            if iter > 1 && (iter % 100 == 0)
                ir0 = floor(Int, iter * 0.1)
                histLogFirstPars = zeros(iter - ir0, n