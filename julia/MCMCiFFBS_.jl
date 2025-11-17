using LinearAlgebra, StatsFuns, StatsBase, Distributions, RData, JLD2

"""
    MCMCiFFBS_(N, initParamValues, Xinit, TestMat, CaptHist, birthTimes, 
               startSamplingPeriod, endSamplingPeriod, nuTimes, CaptEffort,
               capturesAfterMonit, numSeasons, seasonStart, maxt, hp_lambda,
               hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, hp_nu, hp_xi,
               hp_theta, hp_rho, hp_phi, hp_eta, k, K, sd_xi_min, method,
               epsilon, epsilonalphas, epsilonbq, epsilontau, epsilonc1,
               epsilonsens, L, path, blockSize)

Performs MCMC-iFFBS for CMR-Gompertz-SEID-rho model with group-specific alpha

# Arguments
- `N`: Number of MCMC iterations
- `Xinit`: Matrix of dimension (m x maxt) containing the initial states
- `TestMat`: Matrix with all capture events. The columns are (time, id, group, test1, test2, test3, ...)
- `CaptHist`: Matrix of dimension (m x maxt) containing the capture history
- `birthTimes`: Integer vector of length m representing the birth times
- `startSamplingPeriod`: Integer vector of length m with the start sampling times
- `endSamplingPeriod`: Integer vector of length m with the end sampling times
- `nuTimes`: Integer vector saying at which times nu parameters are applied
- `CaptEffort`: (G x maxt) matrix with 1 indicates monitoring and 0 otherwise
- `capturesAfterMonit`: matrix containing the last capture times for individuals captured after the monitoring period
- `numSeasons`: Number of seasons
- `seasonStart`: Starting season
- `maxt`: Number of time points
- `hp_lambda`: Vector of hyperparameter values for the Gamma prior on mean of alpha (1/lambda)
- `hp_beta`: Vector of hyperparameter values for the Gamma prior on frequency-dependent transmission rate
- `hp_q`: Vector of hyperparameter values for the Gamma prior on q
- `hp_tau`: Vector of hyperparameter values for the Gamma prior on average latent period
- `hp_a2`: Vector of hyperparameter values for the Gamma prior on Gompertz parameter a2
- `hp_b2`: Vector of hyperparameter values for the Gamma prior on Gompertz parameter b2
- `hp_c1`: Vector of hyperparameter values for the Gamma prior on Gompertz parameter c1
- `hp_nu`: Vector of hyperparameter values for Dirichlet prior on the initial probability of infection of being susceptible, exposed, infectious
- `hp_xi`: Vector of hyperparameter values (mean, std deviation) for the prior on the Brock changepoint
- `hp_theta`: Vector of hyperparameter values for the Beta prior on test sensitivities
- `hp_rho`: Vector of hyperparameter values for the Beta prior on scaling factor of test sensitivities in the latent period
- `hp_phi`: Vector of hyperparameter values for the Beta prior on test specificities
- `hp_eta`: Vector of hyperparameter values for the Beta prior on capture probabilities
- `k`: Positive integer (shape) parameter of Gamma distribution for the latent period
- `K`: Rescaling parameter for the population size (in order to have beta independent of q)
- `sd_xi_min`: Minimum value for the proposal standard deviation
- `method`: Integer indicating which method is used for updating infection rates and Gompertz parameters (method=1 for "HMC" and method=2 for "RWMH")
- `epsilon`: Leapfrog stepsize in HMC
- `epsilonalphas`: Leapfrog stepsize in HMC for alphas
- `epsilonbq`: Leapfrog stepsize in HMC for beta and q
- `epsilontau`: Leapfrog stepsize in HMC for tau
- `epsilonc1`: Leapfrog stepsize in HMC for c1
- `epsilonsens`: Leapfrog stepsize in HMC for thetas and rhos
- `L`: Number of leapfrog steps in HMC
- `path`: Directory's name where results will be saved on
- `blockSize`: number of iterations saved in each folder
- `initParamValues`: Initial parameter values for the two infection rates; rate at latent period; three Gompertz parameters; initial probability of being susceptible, exposed, and infectious; test sensitivities; rho scaling parameters; test specificities; and seasonal capture probabilities. If set to Inf, then the initial parameter values will be generated from their priors.

# Returns
A matrix containing MCMC draws (N rows) from the posterior distribution of all parameters.

# Details
- Columns having test results in TestMat should have values:
  - 0: negative test
  - 1: positive test
  - NA: test not observed
- Capture history should have values:
  - 0: not captured
  - 1: captured
- Xinit should have values:
  - NA: not born yet
  - 0: susceptible
  - 3: exposed (infected but not infectious)
  - 1: infectious
  - 9: dead
"""
function MCMCiFFBS_(N, 
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
                    blockSize)
    
    m = size(CaptHist, 1) #same
    # No conversion needed - using 1-based indexing throughout
    
    if !((method != 1) | (method != 2))
        println("Please use either method=1 ('HMC') or method=2 ('RWMH').")
    end
    
    numTests = size(TestMat, 2) - 3 #same
    G = Int64(maximum(TestMat[:, 3][.!isnan.(TestMat[:, 3])])) #same
    
    if size(Xinit, 1) != m
        error("Xinit and CaptHist must include the same number of individuals.")
    end
    if length(birthTimes) != m
        error("birthTimes and CaptHist must include the same number of individuals.")
    end
    if size(CaptHist, 2) != size(Xinit, 2)
        error("CaptHist and Xinit must include the same number of time points.")
    end
    
    numNuTimes = length(nuTimes)
    nParsNotGibbs = G + 4 + 3 + 2*numNuTimes + 1 
    # G alphas, lambda, beta, q, tau, survival rates, init probs, Brock changepoint
    
    nParsThetasRhos = 2*numTests # thetas, rhos
    
    # Validate hyperparameter vectors
    if length(hp_lambda) != 2
        error("hp_lambda must be a vector of 2 hyperparameters")
    end
    if length(hp_beta) != 2
        error("hp_beta must be a vector of 2 hyperparameters")
    end
    if length(hp_q) != 2
        error("hp_q must be a vector of 2 hyperparameters")
    end
    if length(hp_tau) != 2
        error("hp_tau must be a vector of 2 hyperparameters")
    end
    if length(hp_a2) != 2
        error("hp_a2 must be a vector of 2 hyperparameters")
    end
    if length(hp_b2) != 2
        error("hp_b2 must be a vector of 2 hyperparameters")
    end
    if length(hp_c1) != 2
        error("hp_c1 must be a vector of 2 hyperparameters")
    end
    if length(hp_nu) != 3
        error("hp_nu must be a vector of 3 hyperparameters")
    end
    if length(hp_xi) != 2
        error("hp_xi must be a vector of 2 hyperparameters")
    end
    if length(hp_theta) != 2
        error("hp_theta must be a vector of 2 hyperparameters")
    end
    if length(hp_rho) != 2
        error("hp_rho must be a vector of 2 hyperparameters")
    end
    if length(hp_phi) != 2
        error("hp_phi must be a vector of 2 hyperparameters")
    end
    if length(hp_eta) != 2
        error("hp_eta must be a vector of 2 hyperparameters")
    end
    
    hp_nu_NumVec = hp_nu
    
    # Initialize parameter values
    lambdaInit = 0.0
    alphaStarInit = 0.0
    betaInit = 0.0
    qInit = 0.0
    tauInit = 0.0
    a2Init = 0.0
    b2Init = 0.0
    c1Init = 0.0
    nuEInit = zeros(Float64, numNuTimes)
    nuIInit = zeros(Float64, numNuTimes)
    xiInit = 0
    thetasInit = zeros(Float64, numTests)
    rhosInit = zeros(Float64, numTests)
    phisInit = zeros(Float64, numTests)
    etasInit = zeros(Float64, numSeasons)
    
    if all(isfinite, initParamValues)
        if length(initParamValues) != ((nParsNotGibbs-G+1)+3*numTests+numSeasons)
            error("If supplied by the user, initParamValues must have correct length. It must be for: alpha, lambda, beta, q, average latent period, 3 Gompertz parameters, probs of being at E and I compartments at nuTimes, sensitivities, rhos, specificities, and capture probabilities.")
        end
        
        println("Initial parameter values supplied by the user: ")
        
        alphaStarInit = initParamValues[1]
        lambdaInit = initParamValues[2]
        betaInit = initParamValues[3]
        qInit = initParamValues[4]
        tauInit = initParamValues[5]
        a2Init = initParamValues[6]
        b2Init = initParamValues[7]
        c1Init = initParamValues[8]
        
        # Print scalar parameters immediately (matching C++ format)
        println("alphaStar = $alphaStarInit")
        println("lambda = $lambdaInit")
        println("alpha = $(alphaStarInit*lambdaInit)")
        println("beta = $betaInit")
        println("q = $qInit")
        println("tau = $tauInit")
        println("a2 = $a2Init")
        println("b2 = $b2Init")
        println("c1 = $c1Init")
        
        for i_nu in 1:numNuTimes
            nuEInit[i_nu] = initParamValues[7 + 1 + i_nu]
        end
        for i_nu in 1:numNuTimes
            nuIInit[i_nu] = initParamValues[7 + 1 + numNuTimes + i_nu]
        end
        
        xiInit = Int(initParamValues[7 + 2 + 2*numNuTimes])
        
        if xiInit < 1 || xiInit > maxt-1
            error("Initial value for the Brock changepoint is outside the study period.")
        end
        
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
        
        # Print vector parameters (matching C++ format with .t() transpose)
        println("nuEs = ", nuEInit')
        println("nuIs = ", nuIInit')
        println("xi = $xiInit")
        print("thetas = ", thetasInit')
        print("rhos = ", rhosInit')
        print("phis = ", phisInit')
        print("etas = ", etasInit')
        
    else
        println("Initial parameter values generated from the prior: ")
        # need to check parametisation
        lambdaInit = rand(Gamma(hp_lambda[1], hp_lambda[2]))  # C++: Rf_rgamma(hp_lambda[0], 1/hp_lambda[1])
        alphaStarInit = rand(Gamma(1.0, 1.0))
        betaInit = rand(Gamma(hp_beta[1], hp_beta[2]))  # C++: Rf_rgamma(hp_beta[0], 1/hp_beta[1])
        qInit = rand(Beta(hp_q[1], hp_q[2]))
        tauInit = rand(Gamma(hp_tau[1], hp_tau[2]))
        a2Init = rand(Gamma(hp_a2[1], hp_a2[2]))  # C++: Rf_rgamma(hp_a2[0], 1/hp_a2[1])
        b2Init = rand(Gamma(hp_b2[1], hp_b2[2]))  # C++: Rf_rgamma(hp_b2[0], 1/hp_b2[1])
        c1Init = rand(Gamma(hp_c1[1], hp_c1[2]))  # C++: Rf_rgamma(hp_c1[0], 1/hp_c1[1])
        
        # Match script: generate all nu parameters at once
        nuVecInit = rand(Dirichlet([8, 1, 1]), numNuTimes)
        nuEInit = nuVecInit[2, :]  # Extract row 2
        nuIInit = nuVecInit[3, :]  # Extract row 3
        
        xiInit = Int(round(rand(Normal(hp_xi[1], hp_xi[2]))))
        
        count = 0
        while xiInit < 1 || xiInit > maxt-1
            count += 1
            xiInit = Int(round(rand(Normal(hp_xi[1], hp_xi[2]))))
            if count > 100
                error("Use a better prior for xi. More than 100 initial values were drawn from the prior, and none of them was during the study period.")
            end
        end
        
        # Match script: use Uniform distributions for test parameters
        thetasInit = rand(Uniform(0.5, 1), numTests)
        rhosInit = rand(Uniform(0.2, 0.8), numTests)
        phisInit = rand(Uniform(0.7, 1), numTests)
        etasInit = rand(Beta(hp_eta[1], hp_eta[2]), numSeasons)
    end
    
    nPars = Int64(nParsNotGibbs + 3*numTests + numSeasons)
    out = zeros(Float64, N, nPars)
    
    # Full log-posterior
    logPostPerIter = zeros(Float64, N)
    logLikPerIter = zeros(Float64, N)
    
    # Number of infectives (I), exposed (E), susceptibles (S) and S+E+I at each time point
    nSus = zeros(Int, maxt, N)
    nExp = zeros(Int, maxt, N)
    nInf = zeros(Int, maxt, N)
    
    # Number of susceptibles, exposed and infectives which were tested
    nSusTested = zeros(Int, maxt, blockSize, numTests)
    nExpTested = zeros(Int, maxt, blockSize, numTests)
    nInfTested = zeros(Int, maxt, blockSize, numTests)
    
    # Now saving these per social group in a field of length G
    nExpTestedPerGroup = [zeros(Int, maxt, blockSize, numTests) for _ in 1:G]
    nInfTestedPerGroup = [zeros(Int, maxt, blockSize, numTests) for _ in 1:G]
    nSusTestedPerGroup = [zeros(Int, maxt, blockSize, numTests) for _ in 1:G]
    
    # Number of exposed, infectives and total number of individuals per social group at each time
    nSusByGroup = zeros(Int, G, maxt, blockSize)
    nExpByGroup = zeros(Int, G, maxt, blockSize)
    nInfByGroup = zeros(Int, G, maxt, blockSize)
    
    # Infection, infectivity and death times
    infTimes = fill(-10, m, blockSize)
    infectivityTimes = fill(-10, m, blockSize)
    deathTimes = fill(-10, m, blockSize)
    
    # Contributions of background rate of infection at times 1,...,T 
    # (i.e. 0,...,maxt-1 indexes) (although at the moment no infection exist in t=1 (t=0))
    AcontribPopTime = zeros(Float64, maxt, blockSize)
    AcontribPop = zeros(Float64, blockSize)
    
    AcontribGroupTime = zeros(Float64, G, maxt, blockSize)
    AcontribGroup = zeros(Float64, G, blockSize)
    
    AcontribIndivGroupTime = zeros(Float64, m, G, maxt)
    
    # Infection rates, latent period and Gompertz parameters
    nParsBlock1 = nParsNotGibbs - 2*numNuTimes - 1  # nuEs, nuIs, and xi are updated separately
    pars = zeros(Float64, nParsBlock1)
    
    alpha_js = zeros(Float64, G)
    
    for g in 1:G
        pars[g] = log(alphaStarInit)
        alpha_js[g] = alphaStarInit*lambdaInit
    end
    pars[G+1] = log(lambdaInit)
    pars[G+1+1] = log(betaInit)
    pars[G+2+1] = logit(qInit)
    pars[G+3+1] = log(tauInit)
    pars[G+4+1] = log(a2Init)
    pars[G+5+1] = log(b2Init)
    pars[G+6+1] = log(c1Init)
    
    lambda = lambdaInit
    beta = betaInit
    q = qInit
    ql = logit(qInit)
    
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
    
    seasonVec = MakeSeasonVec_(numSeasons, seasonStart, maxt)
    
    X = copy(Xinit)
    
    # Check last time each individual was captured
    lastCaptureTimes = zeros(Int, m)
    for i in 1:m
        capt_hist_i = CaptHist[i, :]
        whichCapt = findall(x -> x == 1, capt_hist_i)
        if length(whichCapt) > 0
            lastCaptureTimes[i] = maximum(whichCapt)
        else
            lastCaptureTimes[i] = birthTimes[i] #+ 1
        end
    end
    
    # Covariance matrix of posteriors used in RWMH only
    Sigma = Matrix{Float64}(I, nParsBlock1, nParsBlock1) * 0.1
    Sigma2 = Matrix{Float64}(I, 2*numTests, 2*numTests) * 0.01
    can = zeros(Float64, nParsBlock1)
    
    sd_xi = 2.0 # starting with a small proposal variance in the first 100 iterations
    count_accept_sd_xi = 0
    
    thetas_rhos = zeros(Float64, 2*numTests)
    
    # Calculate matrix of ages
    # This section of c++ was v confusing due to 0-based indexing arithmatic.
    # Need to double check I've translated properly.
    ageMat = fill(-10, m, maxt)
    for i in 1:m
        mint_i = max(1, birthTimes[i])# + 1)
        for tt in mint_i:maxt
            ageMat[i, tt] = tt - birthTimes[i] # think this is right
        end
    end
    
    SocGroup = LocateIndiv(TestMat, birthTimes) # (m x maxt) matrix
    
    CaptHistUsed = copy(CaptHist)
    for ii in 1:m
        for tt in 1:maxt
            g = SocGroup[ii, tt]
            
            if (g == 0) || (g == -1)
                CaptHistUsed[ii, tt] = 0
            else
                CaptHistUsed[ii, tt] = CaptHistUsed[ii, tt] * CaptEffort[g, tt]
            end
        end
    end
    
    # numInfecMat is a (G x (maxt-1)) matrix, where the (g,t)-th element
    # is the number of infectious at group g at time t (except the individual 
    # that is being updated).
    # This is for the coming individual to be updated, i.e. the 1st individual here.
    # Inside the function iFFBS, numInfecMat will be updated.
    numInfecMat = zeros(Int, G, maxt-1)
    for tt in 1:maxt-1
        for ii in 2:m # without 1st indiv
            if X[ii, tt] == 1
                g_i_tt = SocGroup[ii, tt]
                if g_i_tt != 0
                    numInfecMat[g_i_tt, tt] += 1
                end
            end
        end
    end
    
    # mPerGroup is a (G x (maxt)) matrix, where the (g,t)-th element
    # is the number of individuals at group g at time t (except the individual 
    # that is being updated)
    # This is for the coming individual to be updated, i.e. the 1st individual here.
    mPerGroup = zeros(Int, G, maxt)
    for tt in 1:maxt
        for ii in 2:m # without 1st indiv
            if (X[ii, tt] == 0) || (X[ii, tt] == 1) || (X[ii, tt] == 3)
                g_i_tt = SocGroup[ii, tt]
                if g_i_tt != 0
                    mPerGroup[g_i_tt, tt] += 1
                end
            end
        end
    end

    TestField = TestMatAsField_CORRECTED(TestMat, m)
    TestFieldProposal = copy(TestField)
    TestTimes = TestTimesField(TestMat, m)
    
    idVecAll = 1:m
    corrector = zeros(Float64, maxt, 4)
    predProb = zeros(Float64, maxt, 4)
    filtProb = zeros(Float64, maxt, 4)
    
    corrector_theta_rho = zeros(Float64, maxt, 4)
    
    # When updating a new individual jj using iFFBS, some transition probabilities of
    # the remaining m-1 individuals have to be updated
    # Below we calculate which individuals have to be updated at the end of the update
    # of individual jj (that is, probs to be used when updating jj+1)
    whichRequireUpdate = Vector{Vector{Int}}(undef, (m-1)*(maxt-1))
    count = 0
    for i in 1:m-1
        id = i
        idNext = i+1
        
        for tt in 1:maxt-1
            idx = Int[]
            for jj in 2:m
                if ((jj != id) && (jj != idNext)) &&
                   (SocGroup[jj, tt] != 0) &&
                   (SocGroup[jj, tt] == SocGroup[id, tt] ||
                    SocGroup[jj, tt] == SocGroup[idNext, tt])
                    push!(idx, jj)
                end
            end
            count += 1
            whichRequireUpdate[count] = idx
        end
    end
    
    iterSub = 0
    
    # Start MCMC iterations -------------------------------------------
    for iter in 1:N
        
        lambda = exp(pars[G+1])
        for g in 1:G
            alpha_js[g] = exp(pars[g]) * lambda
        end
        beta = exp(pars[G+2])
        q = logistic(pars[G+3])
        ql = pars[G+3]
        tau = exp(pars[G+4])
        a2 = exp(pars[G+5])
        b2 = exp(pars[G+6])
        c1 = exp(pars[G+7])
        
        ql = logit(q)
        
        sumLogCorrector = 0.0
        
        # Update death probabilities conditional on age using new Gompertz parameters
        probDyingMat = fill(-10.0, m, maxt)
        LogProbDyingMat = zeros(Float64, m, maxt)
        LogProbSurvMat = zeros(Float64, m, maxt)
        
        for i in 1:m
            for tt in 1:maxt
                if ageMat[i, tt] > 0
                    if tt > lastCaptureTimes[i]
                        age_i_tt = Float64(ageMat[i, tt])
                        condProbDeath = TrProbDeath_(age_i_tt, a2, b2, c1, false)
                        probDyingMat[i, tt] = condProbDeath
                        LogProbDyingMat[i, tt] = log(condProbDeath)
                        LogProbSurvMat[i, tt] = TrProbSurvive_(age_i_tt, a2, b2, c1, true)
                    else
                        probDyingMat[i, tt] = 0.0
                        LogProbDyingMat[i, tt] = log(0.0)
                        LogProbSurvMat[i, tt] = log(1.0)
                    end
                end
            end
        end
        
        logProbEtoE = log(1.0 - cdf(Erlang(k, k/tau), 1))
        logProbEtoI = log(cdf(Erlang(k, k/tau), 1))
        
        # Update probs from tt to tt+1 using new infection rates
        logProbStoSgivenSorE = zeros(Float64, G, maxt-1)
        logProbStoEgivenSorE = zeros(Float64, G, maxt-1)
        logProbStoSgivenI = zeros(Float64, G, maxt-1)
        logProbStoEgivenI = zeros(Float64, G, maxt-1)
        logProbStoSgivenD = zeros(Float64, G, maxt-1)
        logProbStoEgivenD = zeros(Float64, G, maxt-1)
        
        for tt in 1:maxt-1
            for g in 1:G
                mgt = mPerGroup[g, tt] # without the 1st individual
                
                if SocGroup[1, tt] == g
                    # if 1st individual is alive and S or E
                    inf_mgt = numInfecMat[g, tt] / ((Float64(mgt+1.0)/K)^q)
                    logProbStoSgivenSorE[g, tt] = -alpha_js[g] - beta*inf_mgt                                  
                    logProbStoEgivenSorE[g, tt] = safe_log1mexp(alpha_js[g] + beta*inf_mgt)
                    
                    # if 1st individual is alive and I
                    inf_mgt = (numInfecMat[g, tt] + 1.0) / ((Float64(mgt+1.0)/K)^q)
                    logProbStoSgivenI[g, tt] = -alpha_js[g] - beta*inf_mgt
                    logProbStoEgivenI[g, tt] = safe_log1mexp(alpha_js[g] + beta*inf_mgt)
                    
                    # if 1st individual is dead
                    inf_mgt = numInfecMat[g, tt] / ((Float64(mgt)/K)^q)
                    logProbStoSgivenD[g, tt] = -alpha_js[g] - beta*inf_mgt
                    logProbStoEgivenD[g, tt] = safe_log1mexp(alpha_js[g] + beta*inf_mgt)
                else
                    # if 1st individual is alive and S or E
                    inf_mgt = numInfecMat[g, tt] / ((Float64(mgt)/K)^q)
                    FOI = alpha_js[g] + beta*inf_mgt
                    log1mexpFOI = safe_log1mexp(FOI)
                    
                    logProbStoSgivenSorE[g, tt] = -FOI
                    logProbStoEgivenSorE[g, tt] = log1mexpFOI
                    
                    # if 1st individual is alive and I
                    logProbStoSgivenI[g, tt] = -FOI
                    logProbStoEgivenI[g, tt] = log1mexpFOI
                    
                    # if 1st individual is dead
                    logProbStoSgivenD[g, tt] = -FOI
                    logProbStoEgivenD[g, tt] = log1mexpFOI
                #isnan(logProbStoEgivenSorE[g, tt]) ? println([g, tt, alpha_js[g], beta, numInfecMat[g, tt],m, mgt, q, K]) : nothing

                end
                #isnan(logProbStoEgivenSorE[g, tt]) ? println([g, tt, alpha_js[g], beta, numInfecMat[g, tt],m, mgt, q, K]) : nothing
            end
        end
        
        logProbRest = zeros(Float64, maxt-1, 4, m)
        for jj in 2:m
            for tt in 1:maxt-1
                # update  logProbRest(tt,_,jj) except 1st individual
                if (X[jj, tt] == 0) || (X[jj, tt] == 1) || (X[jj, tt] == 3)
                    iFFBScalcLogProbRest(jj, tt, logProbRest, X, SocGroup,
                                         LogProbDyingMat, LogProbSurvMat,
                                         logProbStoSgivenSorE, logProbStoEgivenSorE,
                                         logProbStoSgivenI, logProbStoEgivenI,
                                         logProbStoSgivenD, logProbStoEgivenD,
                                         logProbEtoE, logProbEtoI)
                end
            end
        end
        
        logTransProbRest = zeros(Float64, maxt-1, 4)
        for jj in 2:m
            logTransProbRest += logProbRest[:, :, jj]
        end

        for jj in 1:m
            iFFBS_(alpha_js, beta, q, tau, k, K,
                   probDyingMat,
                   LogProbDyingMat, 
                   LogProbSurvMat,
                   logProbRest,
                   nuTimes,
                   nuEs, 
                   nuIs,
                   thetas, 
                   rhos,
                   phis,
                   etas,
                   jj, birthTimes[jj], startSamplingPeriod[jj], endSamplingPeriod[jj],
                   X, seasonVec, TestField[jj], TestTimes[jj], CaptHist, corrector, predProb, filtProb, 
                   logTransProbRest, numInfecMat, SocGroup, mPerGroup, idVecAll,
                   logProbStoSgivenSorE, logProbStoEgivenSorE, 
                   logProbStoSgivenI, logProbStoEgivenI, 
                   logProbStoSgivenD, logProbStoEgivenD, 
                   logProbEtoE, logProbEtoI, 
                   whichRequireUpdate, 
                   sumLogCorrector)
  
        end
        
        lastObsAliveTimes = zeros(Int, m)
        for jj in 1:m
            which_deadTimes = findall(x -> x == 9, X[jj, :])
            if length(which_deadTimes) > 0
                lastObsAliveTimes[jj] = minimum(which_deadTimes)
            else
                lastObsAliveTimes[jj] = endSamplingPeriod[jj]
            end
        end
        
        # numInfec here is for all except the 1-st individual 
        # (that is, the vector updated at the iFFBS for the m-th individual).
        # Thus, we need to take into account the 1st individual
        totalNumInfec = copy(numInfecMat)
        for tt in 1:maxt-1
            if X[1, tt] == 1
                g = SocGroup[1, tt]
                totalNumInfec[g, tt] += 1
            end
        end
        # similarly for totalmPerGroup:
        totalmPerGroup = copy(mPerGroup)
        for tt in 1:maxt
            if X[1, tt] == 0 || X[1, tt] == 1 || X[1, tt] == 3
                g = SocGroup[1, tt]
                totalmPerGroup[g, tt] += 1
            end
        end
        
        # Updating (a, b, tau) and Gompertz parameters using HMC or RWMH
        if method == 1
            pars = HMC_2_CORRECTED(pars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                        birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
                        ageMat, epsilon, epsilonalphas, epsilonbq, epsilontau, epsilonc1, nParsBlock1, L, 
                        hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
            
        elseif method == 2
            if iter > 0 && (iter) % 100 == 0
                ir0 = floor(iter * 0.1)
                histLogFirstPars = zeros(Float64, iter-ir0, nParsBlock1) 
                
                for ir in 1:iter-ir0
                    for ic in 1:nParsBlock1
                        if ic <= G
                            histLogFirstPars[ir, ic] = log(out[ir+ir0, ic] / out[ir+ir0, G+1])
                        elseif ic == G+3
                            histLogFirstPars[ir, ic] = logit(out[ir+ir0, ic])
                        else
                            histLogFirstPars[ir, ic] = log(out[ir+ir0, ic])
                        end
                    end
                end
                postVar = cov(histLogFirstPars)
                
                Sigma = (2.38^2 / size(postVar, 2)) * postVar
            end
            
            can = multrnorm(pars, Sigma)
            
            pars = RWMH_(can, pars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                        birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
                        ageMat,
                        hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
        end
        
        # Updating nuEs and nuIs using Gibbs Sampling
        numS_atnuTimes = zeros(Int, numNuTimes)
        numE_atnuTimes = zeros(Int, numNuTimes)
        numI_atnuTimes = zeros(Int, numNuTimes)
        
        startTime = 0
        i_nu = 0
        for i in 1:m
            startTime = startSamplingPeriod[i]
            
            if birthTimes[i] < startTime  # born before monitoring started
                i_nu = findfirst(x -> x == startTime, nuTimes)
                
                if X[i, startTime] == 0
                    numS_atnuTimes[i_nu] += 1
                elseif X[i, startTime] == 3
                    numE_atnuTimes[i_nu] += 1
                elseif X[i, startTime] == 1
                    numI_atnuTimes[i_nu] += 1
                end
            end
        end
        
        nuDirichParamsMat = zeros(Float64, numNuTimes, 3)
        nuSEIMat = zeros(Float64, numNuTimes, 3)
        for i_nu in 1:numNuTimes
            nuDirichParams = [numS_atnuTimes[i_nu] + hp_nu[1], 
                             numE_atnuTimes[i_nu] + hp_nu[2], 
                             numI_atnuTimes[i_nu] + hp_nu[3]]
            nuSEI = rand(Dirichlet(nuDirichParams))
            nuEs[i_nu] = nuSEI[2]
            nuIs[i_nu] = nuSEI[3]
            
            nuDirichParamsMat[i_nu, :] = nuDirichParams
            nuSEIMat[i_nu, :] = nuSEI
        end
        
        if method == 1
            # HMC step for thetas and rhos
            thetas_rhos = HMC_thetas_rhos(thetas, rhos, X, startSamplingPeriod, 
                                          endSamplingPeriod, TestField,
                                          TestTimes, hp_theta, hp_rho, epsilonsens, L)
            
        elseif method == 2
            if iter > 0 && (iter) % 100 == 0
                ir0 = floor(iter * 0.1)
                histThetasRhos = zeros(Float64, iter-ir0, nParsThetasRhos)
                for ir in 1:iter-ir0
                    for ic in 1:nParsThetasRhos
                        natScaleValue = out[ir+ir0, nParsNotGibbs+ic]
                        histThetasRhos[ir, ic] = log(natScaleValue / (1-natScaleValue))
                    end
                end
                postVar = cov(histThetasRhos)
                Sigma2 = (2.38^2 / size(postVar, 2)) * postVar
            end
            
            # MH step for thetas and rhos
            thetas_rhos = RWMH_thetas_rhos(thetas, rhos, X, startSamplingPeriod, 
                                           endSamplingPeriod, TestField,
                                           TestTimes, hp_theta, hp_rho, Sigma2)
        end
        
        for iTest in 1:numTests
            thetas[iTest] = thetas_rhos[iTest]
            rhos[iTest] = thetas_rhos[iTest+numTests]
        end
        
        # Updating test specificities using Gibbs Sampling
        sensSpecMatrix = CheckSensSpec__CORRECTED(numTests, TestField, TestTimes, X)
        
        for iTest in 1:numTests
            phis[iTest] = rand(Beta(sensSpecMatrix[3, iTest] + hp_phi[1],
                                    sensSpecMatrix[4, iTest] + hp_phi[2]))
        end
        
        # Updating Brock changepoint -----------
        # Adapting proposal variance (0.44 acceptance rate target approach)
        if (iter > 0) && (iter % 100 == 0) && (count_accept_sd_xi < 3) && (iter < 5000)
            out0 = out[:, G+7+1+2*numNuTimes]
            vecSub = out0[iter-99:iter]
            d = diff(vecSub)
            
            ccc = Base.count(x -> x != 0, d)
            
            acc = ccc / 99  # Match C++: fixed denominator
            
            if acc < 0.39
                sd_xi = 0.9 * sd_xi  # Match C++: decrease if acceptance too low
                count_accept_sd_xi = 0
            elseif acc > 0.49
                sd_xi = 1.1 * sd_xi  # Match C++: increase if acceptance too high
                count_accept_sd_xi = 0
            else
                count_accept_sd_xi += 1
            end
            
            if sd_xi < sd_xi_min
                sd_xi = sd_xi_min
            end
            
            println("iter: $(iter)")
            println("sd_xi: $sd_xi")
        end
        
        xiCan = Int(round(rand(Normal(xi, sd_xi))))
        
        # if xiCan==xi, nothing has do be done
        # if xiCan is outside of the studyperiod, then reject it
        if (xiCan != xi) && (xiCan >= 1) && (xiCan <= maxt-1)
            # Function 'TestMatAsFieldProposal' corrects TestFieldProposal given the 
            # current TestField
            TestMatAsFieldProposal(TestFieldProposal, TestField, TestTimes, xi, xiCan, m)
             println("iter: $(iter); sd_xi: $sd_xi; xi: $xi; xiCan: $xiCan")
            # depending on the accept-reject step, either TestField or 
            # TestFieldProposal is updated accordingly in the function RWMH_xi:
            xi = RWMH_xi(xiCan, xi, hp_xi, TestFieldProposal, TestField, TestTimes, 
                        thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
                  
            println("iter: $(iter); sd_xi: $sd_xi; xi: $xi")
        end
        
        # Updating eta using Gibbs Sampling considering irregular trapping
        for s in 1:numSeasons
            sumCaptHist_s = 0
            for tt in 1:maxt
                if seasonVec[tt] == s
                    sumCaptHist_s += sum(CaptHistUsed[:, tt])
                end
            end
            
            g = 0
            sumXAlive_s = 0
            for ir in 1:size(X, 1)
                for ic in 1:size(X, 2)
                    if seasonVec[ic] == s
                        g = SocGroup[ir, ic]
                        if (X[ir, ic] == 0 || X[ir, ic] == 3 || X[ir, ic] == 1) && CaptEffort[g, ic] == 1
                            sumXAlive_s += 1
                        end
                    end
                end
            end
            
            sh1_eta = sumCaptHist_s + hp_eta[1]
            sh2_eta = sumXAlive_s - sumCaptHist_s + hp_eta[2]
            eta_s = rand(Beta(sh1_eta, sh2_eta))
            etas[s] = eta_s
        end
        
        lambda = exp(pars[G+1])
        for g in 1:G
            alpha_js[g] = exp(pars[g]) * lambda
        end
        beta = exp(pars[G+2])
        q = logistic(pars[G+3])
        ql = pars[G+3]
        tau = exp(pars[G+4])
        a2 = exp(pars[G+5])
        b2 = exp(pars[G+6])
        c1 = exp(pars[G+7])
        
        # Store results
        for g in 1:G
            out[iter, g] = exp(pars[g]) * lambda
        end
        out[iter, G+1] = lambda
        out[iter, G+2] = exp(pars[G+2])
        out[iter, G+3] = logistic(pars[G+3])
        out[iter, G+4] = exp(pars[G+4])
        out[iter, G+5] = exp(pars[G+5])
        out[iter, G+6] = exp(pars[G+6])
        
        for i_nu in 1:numNuTimes
            out[iter, G+7+1+i_nu] = nuEs[i_nu]
        end
        for i_nu in 1:numNuTimes
            out[iter, G+7+1+numNuTimes+i_nu] = nuIs[i_nu]
        end
        out[iter, G+7+1+2*numNuTimes] = xi
        
        for iTest in 1:numTests
            out[iter, nParsNotGibbs+iTest] = thetas[iTest]
        end
        for iTest in 1:numTests
            out[iter, nParsNotGibbs+numTests+iTest] = rhos[iTest]
        end
        for iTest in 1:numTests
            out[iter, nParsNotGibbs+2*numTests+iTest] = phis[iTest]
        end
        for s in 1:numSeasons
            out[iter, nParsNotGibbs+3*numTests+s] = etas[s]
        end
        
        # Calculating full log-posterior
        logPostPerIter[iter] = logPost_(pars, G, X, totalNumInfec, 
                                         SocGroup, totalmPerGroup,
                                         birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
                                         ageMat, 
                                         hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
        
        # Calculate logLikPerIter using logPost - logPriors:
        a_prior = 0.0
        for g in 1:G
            a = exp(pars[g])
            a_prior += logpdf(Gamma(1.0, 1.0), a) + log(a)
        end
        lambda = exp(pars[G+1])
        beta = exp(pars[G+2])
        q = logistic(pars[G+3])
        ql = pars[G+3]
        tau = exp(pars[G+4])
        a2 = exp(pars[G+5])
        b2 = exp(pars[G+6])
        c1 = exp(pars[G+7])
        
        # Removing Jacobian terms
        logPostPerIter[iter] -= (sum(pars) - pars[G+3]) + (ql - 2*log(1+exp(ql)))
        
        lambda_prior = logpdf(Gamma(hp_lambda[1], hp_lambda[2]), lambda)
        b_prior = logpdf(Gamma(hp_beta[1], hp_beta[2]), beta)
        q_prior = logpdf(Beta(hp_q[1], hp_q[2]), q)
        tau_prior = logpdf(Gamma(hp_tau[1], hp_tau[2]), tau)
        a2_prior = logpdf(Gamma(hp_a2[1], hp_a2[2]), a2)
        b2_prior = logpdf(Gamma(hp_b2[1], hp_b2[2]), b2)
        c1_prior = logpdf(Gamma(hp_c1[1], hp_c1[2]), c1)
        
        logprior = a_prior + lambda_prior + b_prior + q_prior + tau_prior +
                   a2_prior + b2_prior + c1_prior
        
        logLikPerIter[iter] = logPostPerIter[iter] - logprior
        
        # nuEs, nuIs
        nuDirichParamsPrior = hp_nu
        
        for i_nu in 1:numNuTimes
            dens_nu_post = logpdf(Dirichlet(nuDirichParamsMat[i_nu, :]), nuSEIMat[i_nu, :])
            logPostPerIter[iter] += dens_nu_post
            
            dens_nu_prior = logpdf(Dirichlet(nuDirichParamsPrior), nuSEIMat[i_nu, :])
            logLikPerIter[iter] += dens_nu_post - dens_nu_prior
        end
        
        # Adding the observation process part
        logPostPerIter[iter] += sumLogCorrector
        logLikPerIter[iter] += sumLogCorrector
        
        # Adding remaining log-priors to the log-posterior
        # Brock changepoint prior
        xiLogPrior = logpdf(Normal(hp_xi[1], hp_xi[2]), xi)
        # thetas, rhos, phis
        thetasLogPrior = 0.0
        rhosLogPrior = 0.0
        phisLogPrior = 0.0
        for iTest in 1:numTests
            thetasLogPrior += logpdf(Beta(hp_theta[1], hp_theta[2]), thetas[iTest])
            rhosLogPrior += logpdf(Beta(hp_rho[1], hp_rho[2]), rhos[iTest])
            phisLogPrior += logpdf(Beta(hp_phi[1], hp_phi[2]), phis[iTest])
        end
        # etas
        etasLogPrior = 0.0
        for s in 1:numSeasons
            etasLogPrior += logpdf(Beta(hp_eta[1], hp_eta[2]), etas[s])
        end
        
        logPostPerIter[iter] += xiLogPrior + thetasLogPrior + rhosLogPrior + phisLogPrior + etasLogPrior
        
        # Saving nInf, nSus, nTot, nSusTested, nExpTested, nInfTested
        g_i_tt = 0
        for tt in 1:maxt
            nSus[tt, iter] = Base.count(x -> x == 0, X[:, tt])
            nExp[tt, iter] = Base.count(x -> x == 3, X[:, tt])
            nInf[tt, iter] = Base.count(x -> x == 1, X[:, tt])
            
            for i in 1:m
                g_i_tt = SocGroup[i, tt]
                if X[i, tt] == 0
                    nSusByGroup[g_i_tt, tt, iterSub+1] += 1
                end
                if X[i, tt] == 3
                    nExpByGroup[g_i_tt, tt, iterSub+1] += 1
                end
                if X[i, tt] == 1
                    nInfByGroup[g_i_tt, tt, iterSub+1] += 1
                end
            end
        end
        
        tt = 0
        tested = false
        for i in 1:m
            Tests_i = TestField[i]
            testTimes_i = TestTimes[i]
            if length(testTimes_i) > 0
                for tt_i in 1:length(testTimes_i)
                    tt = testTimes_i[tt_i]
                    for iTest in 1:numTests
                        tested = (Tests_i[tt_i, iTest] == 0 || Tests_i[tt_i, iTest] == 1)
                        if tested
                            g_i_tt = SocGroup[i, tt]
                            if X[i, tt] == 1
                                nInfTested[tt, iterSub+1, iTest] += 1
                                nInfTestedPerGroup[g_i_tt][tt, iterSub+1, iTest] += 1
                            elseif X[i, tt] == 3
                                nExpTested[tt, iterSub+1, iTest] += 1
                                nExpTestedPerGroup[g_i_tt][tt, iterSub+1, iTest] += 1
                            elseif X[i, tt] == 0
                                nSusTested[tt, iterSub+1, iTest] += 1
                                nSusTestedPerGroup[g_i_tt][tt, iterSub+1, iTest] += 1
                            end
                        end
                    end
                end
            end
        end
        
        AcontribIndivGroupTime .= 0.0
        
        g = 0
        totalFOI = 0.0
        for i in 1:m
            if any(x -> x == 3, X[i, :])
                tt = findfirst(x -> x == 3, X[i, :])
                infTimes[i, iterSub+1] = tt
                
                if tt > 1 # at time interval (t-1, t), we do not know infection rates
                    g = SocGroup[i, tt]
                    totalFOI = alpha_js[g] + beta * totalNumInfec[g, tt-1] / ((Float64(totalmPerGroup[g, tt-1])/K)^q)
                    AcontribIndivGroupTime[i, g, tt] = alpha_js[g] / totalFOI
                end
            end
            if any(x -> x == 1, X[i, :])
                infectivityTimes[i, iterSub+1] = findfirst(x -> x == 1, X[i, :])
            end
            if any(x -> x == 9, X[i, :])
                deathTimes[i, iterSub+1] = findfirst(x -> x == 9, X[i, :])
            end
        end
        
        # Calculate Acontrib contributions
        AcontribPop_sum = 0.0
        AcontribPop_count = 0.0
        for i in 1:m
            for g in 1:G
                for tt in 1:maxt
                    if AcontribIndivGroupTime[i, g, tt] > 0.0
                        AcontribPop_sum += AcontribIndivGroupTime[i, g, tt]
                        AcontribPop_count += 1.0
                    end
                end
            end
        end
        AcontribPop[iterSub+1] = AcontribPop_sum / AcontribPop_count
        
        for tt in 1:maxt
            AcontribPopTime_sum = 0.0
            AcontribPopTime_count = 0.0
            for i in 1:m
                for g in 1:G
                    if AcontribIndivGroupTime[i, g, tt] > 0.0
                        AcontribPopTime_sum += AcontribIndivGroupTime[i, g, tt]
                        AcontribPopTime_count += 1.0
                    end
                end
            end
            AcontribPopTime[tt, iterSub+1] = AcontribPopTime_sum / AcontribPopTime_count
        end
        
        for g in 1:G
            AcontribGroup_sum = 0.0
            AcontribGroup_count = 0.0
            for tt in 1:maxt
                for i in 1:m
                    if AcontribIndivGroupTime[i, g, tt] > 0.0
                        AcontribGroup_sum += AcontribIndivGroupTime[i, g, tt]
                        AcontribGroup_count += 1.0
                    end
                end
            end
            AcontribGroup[g, iterSub+1] = AcontribGroup_sum / AcontribGroup_count
        end
        
        for g in 1:G
            for tt in 1:maxt
                AcontribGroupTime_sum = 0.0
                AcontribGroupTime_count = 0.0
                for i in 1:m
                    if AcontribIndivGroupTime[i, g, tt] > 0.0
                        AcontribGroupTime_sum += AcontribIndivGroupTime[i, g, tt]
                        AcontribGroupTime_count += 1.0
                    end
                end
                AcontribGroupTime[g, tt, iterSub+1] = AcontribGroupTime_sum / AcontribGroupTime_count
            end
        end
        
        # Saving temporary MCMC results in files
        # Save every 100 iterations (overwrite) for quick recovery
        if (iter) % 100 == 0
            println("Saving checkpoint at iteration $(iter)...")
            
            # Save as JLD2 (native Julia, faster)
            jldsave(joinpath(path, "checkpoint.jld2");
                    out = out[1:iter, :],
                    logPostPerIter = logPostPerIter[1:iter],
                    iter = iter)
        end
        
        # Save every blockSize iterations (permanent blocks)
        if (iter) % blockSize == 0
            println("File with the first $(iter) iterations has been saved.")
            
            numFrom = string(iter-blockSize+1)
            numTo = string(iter)
            block_ = "Iters_from" * numFrom * "to" * numTo * "/"
            
            # Create directory if it doesn't exist
            block_dir = joinpath(path, block_)
            mkpath(block_dir)
            
            # Save main outputs (cumulative) - JLD2 format
            jldsave(joinpath(path, "results.jld2");
                    out = out[1:iter, :],
                    logPostPerIter = logPostPerIter[1:iter])
            # save(joinpath(path, "postPars.jld2"), "out", out)
            # save(joinpath(path, "logPost.jld2"), "logPostPerIter", logPostPerIter)
            # etc.
        end
        
        iterSub += 1
        
        if (iter) % blockSize == 0
            iterSub = 0
            
            nSusTested .= 0
            nExpTested .= 0
            nInfTested .= 0
            
            for g in 1:G
                nExpTestedPerGroup[g] .= 0
                nInfTestedPerGroup[g] .= 0
                nSusTestedPerGroup[g] .= 0
            end
            
            nSusByGroup .= 0
            nExpByGroup .= 0
            nInfByGroup .= 0
            
            infTimes .= -10
            infectivityTimes .= -10
            deathTimes .= -10
        end
        
    end # end MCMC iterations
    
    # Save final results
    
    println("Saving final results...")
  #  jldsave(joinpath(path, "results_final.jld2");
   #         out = out,
    #        logPostPerIter = logPostPerIter)
    println("MCMC completed. Final results saved to: $(joinpath(path, "results_final.jld2"))")
    
    return out
end

