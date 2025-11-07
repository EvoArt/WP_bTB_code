using OffsetArrays: Origin # to use 0-based indexing

macro zero_based(x)
    return :( Origin(0)($x) )
end

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
function MCMCiFFBS_(N::Int, 
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
                    blockSize::Int)
    
    # Apply 0-based indexing to all arrays
    Xinit = @zero_based Xinit
    TestMat = @zero_based TestMat
    CaptHist = @zero_based CaptHist
    birthTimes = @zero_based birthTimes
    startSamplingPeriod = @zero_based startSamplingPeriod
    endSamplingPeriod = @zero_based endSamplingPeriod
    nuTimes = @zero_based nuTimes
    CaptEffort = @zero_based CaptEffort
    capturesAfterMonit = @zero_based capturesAfterMonit
    
    if !((method != 1) | (method != 2))
        println("Please use either method=1 ('HMC') or method=2 ('RWMH').")
    end
    
    m = size(CaptHist, 1)
    numTests = size(TestMat, 2) - 3
    G = maximum(TestMat[:, 3])
    
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
    length(hp_lambda) == 2 || error("hp_lambda must be a vector of 2 hyperparameters")
    length(hp_beta) == 2 || error("hp_beta must be a vector of 2 hyperparameters")
    length(hp_q) == 2 || error("hp_q must be a vector of 2 hyperparameters")
    length(hp_tau) == 2 || error("hp_tau must be a vector of 2 hyperparameters")
    length(hp_a2) == 2 || error("hp_a2 must be a vector of 2 hyperparameters")
    length(hp_b2) == 2 || error("hp_b2 must be a vector of 2 hyperparameters")
    length(hp_c1) == 2 || error("hp_c1 must be a vector of 2 hyperparameters")
    length(hp_nu) == 3 || error("hp_nu must be a vector of 3 hyperparameters")
    length(hp_xi) == 2 || error("hp_xi must be a vector of 2 hyperparameters")
    length(hp_theta) == 2 || error("hp_theta must be a vector of 2 hyperparameters")
    length(hp_rho) == 2 || error("hp_rho must be a vector of 2 hyperparameters")
    length(hp_phi) == 2 || error("hp_phi must be a vector of 2 hyperparameters")
    length(hp_eta) == 2 || error("hp_eta must be a vector of 2 hyperparameters")
    
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
        
        for i_nu in 1:numNuTimes
            nuEInit[i_nu] = initParamValues[8 + 1 + i_nu]
        end
        for i_nu in 1:numNuTimes
            nuIInit[i_nu] = initParamValues[8 + 1 + numNuTimes + i_nu]
        end
        
        xiInit = Int(initParamValues[8 + 1 + 2*numNuTimes])
        
        if xiInit < 1 || xiInit > maxt-1
            error("Initial value for the Brock changepoint is outside the study period.")
        end
        
        for iTest in 1:numTests
            thetasInit[iTest] = initParamValues[nParsNotGibbs-G+2+iTest]
        end
        for iTest in 1:numTests
            rhosInit[iTest] = initParamValues[nParsNotGibbs-G+2+numTests+iTest]
        end
        for iTest in 1:numTests
            phisInit[iTest] = initParamValues[nParsNotGibbs-G+2+2*numTests+iTest]
        end
        for s in 1:numSeasons
            etasInit[s] = initParamValues[nParsNotGibbs-G+2+3*numTests+s]
        end
        
    else
        println("Initial parameter values generated from the prior: ")
        
        lambdaInit = rand(Gamma(hp_lambda[1], 1/hp_lambda[2]))
        alphaStarInit = rand(Gamma(1.0, 1.0))
        betaInit = rand(Gamma(hp_beta[1], 1/hp_beta[2]))
        qInit = rand(Beta(hp_q[1], hp_q[2]))
        tauInit = rand(Gamma(hp_tau[1], 1/hp_tau[2]))
        a2Init = rand(Gamma(hp_a2[1], 1/hp_a2[2]))
        b2Init = rand(Gamma(hp_b2[1], 1/hp_b2[2]))
        c1Init = rand(Gamma(hp_c1[1], 1/hp_c1[2]))
        
        for i_nu in 1:numNuTimes
            samp_nuInit = rand(Dirichlet(hp_nu_NumVec))
            nuEInit[i_nu] = samp_nuInit[2]
            nuIInit[i_nu] = samp_nuInit[3]
        end
        
        xiInit = Int(round(rand(Normal(hp_xi[1], hp_xi[2]))))
        
        count = 0
        while xiInit < 1 || xiInit > maxt-1
            count += 1
            xiInit = Int(round(rand(Normal(hp_xi[1], hp_xi[2]))))
            if count > 100
                error("Use a better prior for xi. More than 100 initial values were drawn from the prior, and none of them was during the study period.")
            end
        end
        
        for iTest in 1:numTests
            thetasInit[iTest] = rand(Beta(hp_theta[1], hp_theta[2]))
        end
        for iTest in 1:numTests
            rhosInit[iTest] = rand(Beta(hp_rho[1], hp_rho[2]))
        end
        for iTest in 1:numTests
            phisInit[iTest] = rand(Beta(hp_phi[1], hp_phi[2]))
        end
        
        for s in 1:numSeasons
            etasInit[s] = rand(Beta(hp_eta[1], hp_eta[2]))
        end
    end
    
    println("alphaStar = $alphaStarInit")
    println("lambda = $lambdaInit")
    println("alpha = $(alphaStarInit*lambdaInit)")
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
    
    # Parameter estimates
    nPars = nParsNotGibbs + 3*numTests + numSeasons
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
    zerosCube = zeros(Int, maxt, blockSize, numTests)
    nExpTestedPerGroup = [zerosCube for _ in 1:G]
    nInfTestedPerGroup = [zerosCube for _ in 1:G]
    nSusTestedPerGroup = [zerosCube for _ in 1:G]
    
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
    pars[G+2] = log(betaInit)
    pars[G+3] = logit(qInit)
    pars[G+4] = log(tauInit)
    pars[G+5] = log(a2Init)
    pars[G+6] = log(b2Init)
    pars[G+7] = log(c1Init)
    
    lambda = lambdaInit
    beta = betaInit
    q = qInit
    ql = logit(qInit)
    
    tau = tauInit
    a2 = a2Init
    b2 = b2Init
    c1 = c1Init
    
    nuEs = nuEInit
    nuIs = nuIInit
    
    xi = xiInit
    
    thetas = thetasInit
    rhos = rhosInit
    
    phis = phisInit
    etas = etasInit
    
    seasonVec = MakeSeasonVec_(numSeasons, seasonStart, maxt)
    
    X = copy(Xinit)
    
    # Check last time each individual was captured
    lastCaptureTimes = zeros(Int, m)
    for i in 0:m-1
        capt_hist_i = CaptHist[i+1, :]
        whichCapt = findall(x -> x == 1, capt_hist_i) .- 1
        if length(whichCapt) > 0
            lastCaptureTimes[i+1] = maximum(whichCapt) + 1
        else
            lastCaptureTimes[i+1] = birthTimes[i+1] + 1
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
    ageMat = fill(-10, m, maxt)
    for i in 0:m-1
        mint_i = max(1, birthTimes[i+1] + 1)
        for tt in mint_i-1:maxt-1
            ageMat[i+1, tt+1] = tt + 2 - (birthTimes[i+1] + 1)
        end
    end
    
    SocGroup = LocateIndiv(TestMat, birthTimes) # (m x maxt) matrix
    
    CaptHistUsed = copy(CaptHist)
    for ii in 0:m-1
        for tt in 0:maxt-1
            g = SocGroup[ii+1, tt+1]
            
            if g == 0 || g == -1
                CaptHistUsed[ii+1, tt+1] = 0
            else
                CaptHistUsed[ii+1, tt+1] = CaptHistUsed[ii+1, tt+1] * CaptEffort[g, tt+1]
            end
        end
    end
    
    # numInfecMat is a (G x (maxt-1)) matrix, where the (g,t)-th element
    # is the number of infectious at group g at time t (except the individual 
    # that is being updated).
    # This is for the coming individual to be updated, i.e. the 1st individual here.
    # Inside the function iFFBS, numInfecMat will be updated.
    numInfecMat = zeros(Int, G, maxt-1)
    for tt in 0:maxt-2
        for ii in 1:m-1 # without 1st indiv
            if X[ii+1, tt+1] == 1
                g_i_tt = SocGroup[ii+1, tt+1]
                if g_i_tt != 0
                    numInfecMat[g_i_tt, tt+1] += 1
                end
            end
        end
    end
    
    # mPerGroup is a (G x (maxt)) matrix, where the (g,t)-th element
    # is the number of individuals at group g at time t (except the individual 
    # that is being updated)
    # This is for the coming individual to be updated, i.e. the 1st individual here.
    mPerGroup = zeros(Int, G, maxt)
    for tt in 0:maxt-1
        for ii in 1:m-1 # without 1st indiv
            if X[ii+1, tt+1] == 0 || X[ii+1, tt+1] == 1 || X[ii+1, tt+1] == 3
                g_i_tt = SocGroup[ii+1, tt+1]
                if g_i_tt != 0
                    mPerGroup[g_i_tt, tt+1] += 1
                end
            end
        end
    end
    
    TestField = TestMatAsField(TestMat, m)
    TestFieldProposal = copy(TestField)
    TestTimes = TestTimesField(TestMat, m)
    
    idVecAll = 0:m-1
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
    for i in 0:m-2
        id = i
        idNext = i+1
        
        for tt in 0:maxt-2
            idx = Int[]
            for jj in 1:m-1
                if ((jj != id+1) && (jj != idNext+1)) &&
                   (SocGroup[jj, tt+1] != 0) &&
                   (SocGroup[jj, tt+1] == SocGroup[id+1, tt+1] ||
                    SocGroup[jj, tt+1] == SocGroup[idNext+1, tt+1])
                    push!(idx, jj-1)
                end
            end
            count += 1
            whichRequireUpdate[count] = idx
        end
    end
    
    iterSub = 0
    
    # Start MCMC iterations -------------------------------------------
    for iter in 0:N-1
        
        if iter > 0 && (iter+1) % 1000 == 0
            println("iter: $(iter+1) out of N=$N")
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
        
        ql = logit(q)
        
        sumLogCorrector = 0.0
        
        # Update death probabilities conditional on age using new Gompertz parameters
        probDyingMat = fill(-10.0, m, maxt)
        LogProbDyingMat = zeros(Float64, m, maxt)
        LogProbSurvMat = zeros(Float64, m, maxt)
        
        for i in 0:m-1
            for tt in 0:maxt-1
                if ageMat[i+1, tt+1] > 0
                    if tt+1 > lastCaptureTimes[i+1]
                        age_i_tt = Float64(ageMat[i+1, tt+1])
                        condProbDeath = TrProbDeath_(age_i_tt, a2, b2, c1, false)
                        probDyingMat[i+1, tt+1] = condProbDeath
                        LogProbDyingMat[i+1, tt+1] = log(condProbDeath)
                        LogProbSurvMat[i+1, tt+1] = TrProbSurvive_(age_i_tt, a2, b2, c1, true)
                    else
                        probDyingMat[i+1, tt+1] = 0.0
                        LogProbDyingMat[i+1, tt+1] = log(0.0)
                        LogProbSurvMat[i+1, tt+1] = log(1.0)
                    end
                end
            end
        end
        
        logProbEtoE = log(1.0 - ErlangCDF(1, k, tau/Float64(k)))
        logProbEtoI = log(ErlangCDF(1, k, tau/Float64(k)))
        
        # Update probs from tt to tt+1 using new infection rates
        logProbStoSgivenSorE = zeros(Float64, G, maxt-1)
        logProbStoEgivenSorE = zeros(Float64, G, maxt-1)
        logProbStoSgivenI = zeros(Float64, G, maxt-1)
        logProbStoEgivenI = zeros(Float64, G, maxt-1)
        logProbStoSgivenD = zeros(Float64, G, maxt-1)
        logProbStoEgivenD = zeros(Float64, G, maxt-1)
        
        for tt in 0:maxt-2
            for g in 1:G
                mgt = mPerGroup[g, tt+1] # without the 1st individual
                
                if SocGroup[1, tt+1] == g
                    # if 1st individual is alive and S or E
                    inf_mgt = numInfecMat[g, tt+1] / ((Float64(mgt+1.0)/K)^q)
                    logProbStoSgivenSorE[g, tt+1] = -alpha_js[g] - beta*inf_mgt
                    logProbStoEgivenSorE[g, tt+1] = log1mexp(alpha_js[g] + beta*inf_mgt)
                    
                    # if 1st individual is alive and I
                    inf_mgt = (numInfecMat[g, tt+1] + 1.0) / ((Float64(mgt+1.0)/K)^q)
                    logProbStoSgivenI[g, tt+1] = -alpha_js[g] - beta*inf_mgt
                    logProbStoEgivenI[g, tt+1] = log1mexp(alpha_js[g] + beta*inf_mgt)
                    
                    # if 1st individual is dead
                    inf_mgt = numInfecMat[g, tt+1] / ((Float64(mgt)/K)^q)
                    logProbStoSgivenD[g, tt+1] = -alpha_js[g] - beta*inf_mgt
                    logProbStoEgivenD[g, tt+1] = log1mexp(alpha_js[g] + beta*inf_mgt)
                    
                else
                    # if 1st individual is alive and S or E
                    inf_mgt = numInfecMat[g, tt+1] / ((Float64(mgt)/K)^q)
                    FOI = alpha_js[g] + beta*inf_mgt
                    log1mexpFOI = log1mexp(FOI)
                    
                    logProbStoSgivenSorE[g, tt+1] = -FOI
                    logProbStoEgivenSorE[g, tt+1] = log1mexpFOI
                    
                    # if 1st individual is alive and I
                    logProbStoSgivenI[g, tt+1] = -FOI
                    logProbStoEgivenI[g, tt+1] = log1mexpFOI
                    
                    # if 1st individual is dead
                    logProbStoSgivenD[g, tt+1] = -FOI
                    logProbStoEgivenD[g, tt+1] = log1mexpFOI
                end
            end
        end
        
        logProbRest = zeros(Float64, maxt-1, 4, m)
        for jj in 1:m-1
            for tt in 0:maxt-2
                # update logProbRest(tt+1,_,jj+1) except 1st individual
                if X[jj+1, tt+1] == 0 || X[jj+1, tt+1] == 1 || X[jj+1, tt+1] == 3
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
        for jj in 1:m-1
            logTransProbRest += logProbRest[:, :, jj+1]
        end
        
        for jj in 0:m-1
            # updating X(jj, _)
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
                   jj, 
                   birthTimes[jj+1],
                   startSamplingPeriod[jj+1],
                   endSamplingPeriod[jj+1],
                   X,
                   seasonVec,
                   TestField[jj+1],
                   TestTimes[jj+1],
                   CaptHist,
                   corrector,
                   predProb,
                   filtProb,
                   logTransProbRest,
                   numInfecMat, 
                   SocGroup,
                   mPerGroup,
                   idVecAll,
                   logProbStoSgivenSorE, logProbStoEgivenSorE, 
                   logProbStoSgivenI, logProbStoEgivenI, 
                   logProbStoSgivenD, logProbStoEgivenD, 
                   logProbEtoE, logProbEtoI, 
                   whichRequireUpdate, 
                   sumLogCorrector)
        end
        
        lastObsAliveTimes = zeros(Int, m)
        for jj in 0:m-1
            which_deadTimes = findall(x -> x == 9, X[jj+1, :])
            if length(which_deadTimes) > 0
                lastObsAliveTimes[jj+1] = minimum(which_deadTimes)
            else
                lastObsAliveTimes[jj+1] = endSamplingPeriod[jj+1]
            end
        end
        
        # numInfec here is for all except the 1-st individual 
        # (that is, the vector updated at the iFFBS for the m-th individual).
        # Thus, we need to take into account the 1st individual
        totalNumInfec = copy(numInfecMat)
        for tt in 0:maxt-2
            if X[1, tt+1] == 1
                g = SocGroup[1, tt+1]
                totalNumInfec[g, tt+1] += 1
            end
        end
        # similarly for totalmPerGroup:
        totalmPerGroup = copy(mPerGroup)
        for tt in 0:maxt-1
            if X[1, tt+1] == 0 || X[1, tt+1] == 1 || X[1, tt+1] == 3
                g = SocGroup[1, tt+1]
                totalmPerGroup[g, tt+1] += 1
            end
        end
        
        # Updating (a, b, tau) and Gompertz parameters using HMC or RWMH
        if method == 1
            pars = HMC_2(pars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                        birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
                        ageMat, epsilon, epsilonalphas, epsilonbq, epsilontau, epsilonc1, nParsBlock1, L, 
                        hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
            
        elseif method == 2
            if iter > 0 && (iter+1) % 100 == 0
                ir0 = floor(iter * 0.1)
                histLogFirstPars = zeros(Float64, iter-ir0, nParsBlock1) 
                
                for ir in 0:iter-ir0-1
                    for ic in 1:nParsBlock1
                        if ic <= G
                            histLogFirstPars[ir+1, ic] = log(out[ir+ir0+1, ic] / out[ir+ir0+1, G+1])
                        elseif ic == G+3
                            histLogFirstPars[ir+1, ic] = logit(out[ir+ir0+1, ic])
                        else
                            histLogFirstPars[ir+1, ic] = log(out[ir+ir0+1, ic])
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
        for i in 0:m-1
            startTime = startSamplingPeriod[i+1]
            
            if birthTimes[i+1] < startTime  # born before monitoring started
                i_nu = findfirst(x -> x == startTime, nuTimes)
                
                if X[i+1, startTime] == 0
                    numS_atnuTimes[i_nu] += 1
                elseif X[i+1, startTime] == 3
                    numE_atnuTimes[i_nu] += 1
                elseif X[i+1, startTime] == 1
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
            if iter > 0 && (iter+1) % 100 == 0
                ir0 = floor(iter * 0.1)
                histThetasRhos = zeros(Float64, iter-ir0, nParsThetasRhos)
                for ir in 0:iter-ir0-1
                    for ic in 1:nParsThetasRhos
                        natScaleValue = out[ir+ir0+1, nParsNotGibbs+ic]
                        histThetasRhos[ir+1, ic] = log(natScaleValue / (1-natScaleValue))
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
        sensSpecMatrix = CheckSensSpec_(numTests, TestField, TestTimes, X)
        
        for iTest in 1:numTests
            phis[iTest] = rand(Beta(sensSpecMatrix[3, iTest] + hp_phi[1],
                                    sensSpecMatrix[4, iTest] + hp_phi[2]))
        end
        
        # Updating Brock changepoint -----------
        # Adapting proposal variance (0.44 acceptance rate target approach)
        if iter > 0 && (iter+1) % 100 == 0 && count_accept_sd_xi < 3 && iter < 5000
            out0 = out[:, G+7+1+2*numNuTimes]
            vecSub = out0[iter-98:iter]
            d = diff(vecSub)
            
            ccc = count(x -> x != 0, d)
            
            acc = ccc / 99
            
            if acc < 0.39
                sd_xi = 0.9 * sd_xi
                count_accept_sd_xi = 0
            elseif acc > 0.49
                sd_xi = 1.1 * sd_xi
                count_accept_sd_xi = 0
            else
                count_accept_sd_xi += 1
            end
            
            if sd_xi < sd_xi_min
                sd_xi = sd_xi_min
            end
            
            println("iter: $(iter+1)")
            println("sd_xi: $sd_xi")
        end
        
        xiCan = Int(round(rand(Normal(xi, sd_xi))))
        
        # if xiCan==xi, nothing has do be done
        # if xiCan is outside of the studyperiod, then reject it
        if xiCan != xi && xiCan >= 1 && xiCan <= maxt-1
            # Function 'TestMatAsFieldProposal' corrects TestFieldProposal given the 
            # current TestField
            TestMatAsFieldProposal(TestFieldProposal, TestField, TestTimes, xi, xiCan, m)
            
            # depending on the accept-reject step, either TestField or 
            # TestFieldProposal is updated accordingly in the function RWMH_xi:
            xi = RWMH_xi(xiCan, xi, hp_xi, TestFieldProposal, TestField, TestTimes, 
                        thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
        end
        
        # Updating eta using Gibbs Sampling considering irregular trapping
        for s in 1:numSeasons
            sumCaptHist_s = 0
            for tt in 0:maxt-1
                if seasonVec[tt+1] == s
                    sumCaptHist_s += sum(CaptHistUsed[:, tt+1])
                end
            end
            
            g = 0
            sumXAlive_s = 0
            for ir in 0:size(X, 1)-1
                for ic in 0:size(X, 2)-1
                    if seasonVec[ic+1] == s
                        g = SocGroup[ir+1, ic+1]
                        if (X[ir+1, ic+1] == 0 || X[ir+1, ic+1] == 3 || X[ir+1, ic+1] == 1) && CaptEffort[g, ic+1] == 1
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
            out[iter+1, g] = exp(pars[g]) * lambda
        end
        out[iter+1, G+1] = lambda
        out[iter+1, G+2] = exp(pars[G+2])
        out[iter+1, G+3] = logistic(pars[G+3])
        out[iter+1, G+4] = exp(pars[G+4])
        out[iter+1, G+5] = exp(pars[G+5])
        out[iter+1, G+6] = exp(pars[G+6])
        
        for i_nu in 1:numNuTimes
            out[iter+1, G+7+1+i_nu] = nuEs[i_nu]
        end
        for i_nu in 1:numNuTimes
            out[iter+1, G+7+1+numNuTimes+i_nu] = nuIs[i_nu]
        end
        out[iter+1, G+7+1+2*numNuTimes] = xi
        
        for iTest in 1:numTests
            out[iter+1, nParsNotGibbs+iTest] = thetas[iTest]
        end
        for iTest in 1:numTests
            out[iter+1, nParsNotGibbs+numTests+iTest] = rhos[iTest]
        end
        for iTest in 1:numTests
            out[iter+1, nParsNotGibbs+2*numTests+iTest] = phis[iTest]
        end
        for s in 1:numSeasons
            out[iter+1, nParsNotGibbs+3*numTests+s] = etas[s]
        end
        
        # Calculating full log-posterior
        logPostPerIter[iter+1] = logPost_(pars, G, X, totalNumInfec, 
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
        logPostPerIter[iter+1] -= (sum(pars) - pars[G+3]) + (ql - 2*log(1+exp(ql)))
        
        lambda_prior = logpdf(Gamma(hp_lambda[1], 1/hp_lambda[2]), lambda)
        b_prior = logpdf(Gamma(hp_beta[1], 1/hp_beta[2]), beta)
        q_prior = logpdf(Beta(hp_q[1], hp_q[2]), q)
        tau_prior = logpdf(Gamma(hp_tau[1], 1/hp_tau[2]), tau)
        a2_prior = logpdf(Gamma(hp_a2[1], 1/hp_a2[2]), a2)
        b2_prior = logpdf(Gamma(hp_b2[1], 1/hp_b2[2]), b2)
        c1_prior = logpdf(Gamma(hp_c1[1], 1/hp_c1[2]), c1)
        
        logprior = a_prior + lambda_prior + b_prior + q_prior + tau_prior +
                   a2_prior + b2_prior + c1_prior
        
        logLikPerIter[iter+1] = logPostPerIter[iter+1] - logprior
        
        # nuEs, nuIs
        nuDirichParamsPrior = hp_nu
        
        for i_nu in 1:numNuTimes
            dens_nu_post = logpdf(Dirichlet(nuDirichParamsMat[i_nu, :]), nuSEIMat[i_nu, :])
            logPostPerIter[iter+1] += dens_nu_post
            
            dens_nu_prior = logpdf(Dirichlet(nuDirichParamsPrior), nuSEIMat[i_nu, :])
            logLikPerIter[iter+1] += dens_nu_post - dens_nu_prior
        end
        
        # Adding the observation process part
        logPostPerIter[iter+1] += sumLogCorrector
        logLikPerIter[iter+1] += sumLogCorrector
        
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
        
        logPostPerIter[iter+1] += xiLogPrior + thetasLogPrior + rhosLogPrior + phisLogPrior + etasLogPrior
        
        # Saving nInf, nSus, nTot, nSusTested, nExpTested, nInfTested
        g_i_tt = 0
        for tt in 0:maxt-1
            nSus[tt+1, iter+1] = count(x -> x == 0, X[:, tt+1])
            nExp[tt+1, iter+1] = count(x -> x == 3, X[:, tt+1])
            nInf[tt+1, iter+1] = count(x -> x == 1, X[:, tt+1])
            
            for i in 0:m-1
                g_i_tt = SocGroup[i+1, tt+1]
                if X[i+1, tt+1] == 0
                    nSusByGroup[g_i_tt, tt+1, iterSub+1] += 1
                end
                if X[i+1, tt+1] == 3
                    nExpByGroup[g_i_tt, tt+1, iterSub+1] += 1
                end
                if X[i+1, tt+1] == 1
                    nInfByGroup[g_i_tt, tt+1, iterSub+1] += 1
                end
            end
        end
        
        tt = 0
        tested = false
        for i in 0:m-1
            Tests_i = TestField[i+1]
            testTimes_i = TestTimes[i+1] .- 1
            if length(testTimes_i) > 0
                for tt_i in 1:length(testTimes_i)
                    tt = testTimes_i[tt_i]
                    for iTest in 1:numTests
                        tested = (Tests_i[tt_i, iTest] == 0 || Tests_i[tt_i, iTest] == 1)
                        if tested
                            g_i_tt = SocGroup[i+1, tt+1]
                            if X[i+1, tt+1] == 1
                                nInfTested[tt+1, iterSub+1, iTest] += 1
                                nInfTestedPerGroup[g_i_tt][tt+1, iterSub+1, iTest] += 1
                            elseif X[i+1, tt+1] == 3
                                nExpTested[tt+1, iterSub+1, iTest] += 1
                                nExpTestedPerGroup[g_i_tt][tt+1, iterSub+1, iTest] += 1
                            elseif X[i+1, tt+1] == 0
                                nSusTested[tt+1, iterSub+1, iTest] += 1
                                nSusTestedPerGroup[g_i_tt][tt+1, iterSub+1, iTest] += 1
                            end
                        end
                    end
                end
            end
        end
        
        AcontribIndivGroupTime .= 0.0
        
        g = 0
        totalFOI = 0.0
        for i in 0:m-1
            if any(x -> x == 3, X[i+1, :])
                tt = findfirst(x -> x == 3, X[i+1, :]) - 1
                infTimes[i+1, iterSub+1] = tt + 1
                
                if tt > 0 # at time interval (t-1, t), we do not know infection rates
                    g = SocGroup[i+1, tt+1] - 1
                    totalFOI = alpha_js[g+1] + beta * totalNumInfec[g+1, tt] / ((Float64(totalmPerGroup[g+1, tt])/K)^q)
                    AcontribIndivGroupTime[i+1, g+1, tt+1] = alpha_js[g+1] / totalFOI
                end
            end
            if any(x -> x == 1, X[i+1, :])
                infectivityTimes[i+1, iterSub+1] = findfirst(x -> x == 1, X[i+1, :])
            end
            if any(x -> x == 9, X[i+1, :])
                deathTimes[i+1, iterSub+1] = findfirst(x -> x == 9, X[i+1, :])
            end
        end
        
        # Calculate Acontrib contributions
        AcontribPop_sum = 0.0
        AcontribPop_count = 0.0
        for i in 0:m-1
            for g in 1:G
                for tt in 0:maxt-1
                    if AcontribIndivGroupTime[i+1, g, tt+1] > 0.0
                        AcontribPop_sum += AcontribIndivGroupTime[i+1, g, tt+1]
                        AcontribPop_count += 1.0
                    end
                end
            end
        end
        AcontribPop[iterSub+1] = AcontribPop_sum / AcontribPop_count
        
        for tt in 0:maxt-1
            AcontribPopTime_sum = 0.0
            AcontribPopTime_count = 0.0
            for i in 0:m-1
                for g in 1:G
                    if AcontribIndivGroupTime[i+1, g, tt+1] > 0.0
                        AcontribPopTime_sum += AcontribIndivGroupTime[i+1, g, tt+1]
                        AcontribPopTime_count += 1.0
                    end
                end
            end
            AcontribPopTime[tt+1, iterSub+1] = AcontribPopTime_sum / AcontribPopTime_count
        end
        
        for g in 1:G
            AcontribGroup_sum = 0.0
            AcontribGroup_count = 0.0
            for tt in 0:maxt-1
                for i in 0:m-1
                    if AcontribIndivGroupTime[i+1, g, tt+1] > 0.0
                        AcontribGroup_sum += AcontribIndivGroupTime[i+1, g, tt+1]
                        AcontribGroup_count += 1.0
                    end
                end
            end
            AcontribGroup[g, iterSub+1] = AcontribGroup_sum / AcontribGroup_count
        end
        
        for g in 1:G
            for tt in 0:maxt-1
                AcontribGroupTime_sum = 0.0
                AcontribGroupTime_count = 0.0
                for i in 0:m-1
                    if AcontribIndivGroupTime[i+1, g, tt+1] > 0.0
                        AcontribGroupTime_sum += AcontribIndivGroupTime[i+1, g, tt+1]
                        AcontribGroupTime_count += 1.0
                    end
                end
                AcontribGroupTime[g, tt+1, iterSub+1] = AcontribGroupTime_sum / AcontribGroupTime_count
            end
        end
        
        # Saving temporary MCMC results in files
        if (iter+1) % blockSize == 0
            println("File with the first $(iter+1) iterations has been saved.")
            
            # Save results using JLD2 or similar serialization
            # Note: In Julia, you might want to use JLD2.jl or BSON.jl for saving
            # This is a placeholder for the actual saving logic
            
            numFrom = string(iter+1-blockSize+1)
            numTo = string(iter+1)
            block_ = "Iters_from" * numFrom * "to" * numTo * "/"
            
            # Create directory if it doesn't exist
            block_dir = joinpath(path, block_)
            mkpath(block_dir)
            
            # Save various results (placeholder - implement with actual serialization)
            # save(joinpath(path, "postPars.jld2"), "out", out)
            # save(joinpath(path, "logPost.jld2"), "logPostPerIter", logPostPerIter)
            # etc.
        end
        
        iterSub += 1
        
        if (iter+1) % blockSize == 0
            iterSub = 0
            
            nSusTested .= 0
            nExpTested .= 0
            nInfTested .= 0
            
            for g in 1:G
                nExpTestedPerGroup[g] .= zerosCube
                nInfTestedPerGroup[g] .= zerosCube
                nSusTestedPerGroup[g] .= zerosCube
            end
            
            nSusByGroup .= 0
            nExpByGroup .= 0
            nInfByGroup .= 0
            
            infTimes .= -10
            infectivityTimes .= -10
            deathTimes .= -10
        end
        
    end # end MCMC iterations
    
    return out
end

# Helper functions that need to be implemented or imported:
# These would typically be in separate files or modules

# Mathematical helper functions
logit(x) = log(x / (1 - x))
logistic(x) = 1 / (1 + exp(-x))
log1mexp(x) = log1p(-exp(x))

# Statistical distributions
using Distributions

# Placeholder functions that need to be implemented:
function MakeSeasonVec_(numSeasons, seasonStart, maxt)
    # Implementation needed
    return repeat(1:numSeasons, outer=ceil(Int, maxt/numSeasons))[1:maxt]
end

function LocateIndiv(TestMat, birthTimes)
    # Implementation needed
    m = length(birthTimes)
    maxt = size(TestMat, 1)
    return zeros(Int, m, maxt)
end

function TestMatAsField(TestMat, m)
    # Implementation needed
    return [zeros(Int, 0, 0) for _ in 1:m]
end

function TestTimesField(TestMat, m)
    # Implementation needed
    return [zeros(Int, 0) for _ in 1:m]
end

function iFFBScalcLogProbRest(jj, tt, logProbRest, X, SocGroup, 
                             LogProbDyingMat, LogProbSurvMat, 
                             logProbStoSgivenSorE, logProbStoEgivenSorE, 
                             logProbStoSgivenI, logProbStoEgivenI, 
                             logProbStoSgivenD, logProbStoEgivenD, 
                             logProbEtoE, logProbEtoI)
    # Implementation needed
end

function iFFBS_(alpha_js, beta, q, tau, k, K,
               probDyingMat, LogProbDyingMat, LogProbSurvMat,
               logProbRest, nuTimes, nuEs, nuIs, thetas, rhos, phis, etas, 
               jj, birthTime, startSamplingPeriod, endSamplingPeriod,
               X, seasonVec, TestField_jj, TestTimes_jj, CaptHist,
               corrector, predProb, filtProb, logTransProbRest,
               numInfecMat, SocGroup, mPerGroup, idVecAll,
               logProbStoSgivenSorE, logProbStoEgivenSorE, 
               logProbStoSgivenI, logProbStoEgivenI, 
               logProbStoSgivenD, logProbStoEgivenD, 
               logProbEtoE, logProbEtoI, whichRequireUpdate, sumLogCorrector)
    # Implementation needed
end

function HMC_2(pars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
              birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
              ageMat, epsilon, epsilonalphas, epsilonbq, epsilontau, epsilonc1, nParsBlock1, L, 
              hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
    # Implementation needed
    return pars
end

function RWMH_(can, pars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
             birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
             ageMat, hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
    # Implementation needed
    return pars
end

function HMC_thetas_rhos(thetas, rhos, X, startSamplingPeriod, 
                        endSamplingPeriod, TestField, TestTimes, 
                        hp_theta, hp_rho, epsilonsens, L)
    # Implementation needed
    return vcat(thetas, rhos)
end

function RWMH_thetas_rhos(thetas, rhos, X, startSamplingPeriod, 
                         endSamplingPeriod, TestField, TestTimes, 
                         hp_theta, hp_rho, Sigma2)
    # Implementation needed
    return vcat(thetas, rhos)
end

function TestMatAsFieldProposal(TestFieldProposal, TestField, TestTimes, xi, xiCan, m)
    # Implementation needed
end

function RWMH_xi(xiCan, xi, hp_xi, TestFieldProposal, TestField, TestTimes, 
                thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
    # Implementation needed
    return xi
end

function CheckSensSpec_(numTests, TestField, TestTimes, X)
    # Implementation needed
    return zeros(Int, 4, numTests)
end

function logPost_(pars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                 birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
                 ageMat, hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
    # Implementation needed
    return 0.0
end

function TrProbDeath_(age, a2, b2, c1, log_scale)
    # Implementation needed
    return 0.0
end

function TrProbSurvive_(age, a2, b2, c1, log_scale)
    # Implementation needed
    return 0.0
end

function ErlangCDF(x, k, tau)
    # Implementation needed
    return 0.0
end

function multrnorm(mean, cov)
    # Implementation needed - multivariate normal random draw
    return rand(MvNormal(mean, cov))
end
