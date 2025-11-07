using OffsetArrays: Origin # to use 0-based indexing
using Distributions
using LinearAlgebra

macro zero_based(x)
    return :( Origin(0)($x) )
end

# Mathematical helper functions
logit(x) = log(x / (1 - x))
logistic(x) = 1 / (1 + exp(-x))
log1mexp(x) = log1p(-exp(x))

"""
    MakeSeasonVec_(numSeasons, seasonStart, maxt)

Create a vector indicating season for each time point with 0-based indexing.
"""
function MakeSeasonVec_(numSeasons, seasonStart, maxt)
    seasonsVec = 0:numSeasons-1
    rows = findall(x -> x == seasonStart-1, seasonsVec)
    if length(rows) == 0
        error("seasonStart must be an integer from {1, ..., numSeasons}.")
    end
    
    seasonVec = ones(Int, maxt)
    seasonVec[1] = seasonStart
    for tt in 2:maxt
        if seasonVec[tt-1] < numSeasons
            seasonVec[tt] = seasonVec[tt-1] + 1
        end
    end
    
    return seasonVec
end

"""
    LocateIndiv(TestMat, birthTimes)

Locate individuals in social groups over time with 0-based indexing.
"""
function LocateIndiv(TestMat, birthTimes)
    maxt = maximum(TestMat[:, 1])
    m = maximum(TestMat[:, 2])
    
    SocGroup = zeros(Int, m, maxt)
    
    # Apply 0-based indexing
    TestMat = @zero_based TestMat
    birthTimes = @zero_based birthTimes
    SocGroup = @zero_based SocGroup
    
    id_i = TestMat[:, 2]
    for i in 0:m-1
        which_i = findall(x -> x == i, id_i)
        Tests_i = TestMat[which_i, :]
        times_i = Tests_i[:, 1]
        groups_i = Tests_i[:, 3]
        
        tt0 = max(0, birthTimes[i+1])
        
        firstcapttime = minimum(times_i)
        which_row = findall(x -> x == firstcapttime, times_i)
        g = groups_i[which_row[1]]  # first group it belongs to
        
        for tt in tt0:maxt-1
            # check if moved to another group
            tt_capt = findall(x -> x == tt, times_i)
            if length(tt_capt) > 0
                newGroup = groups_i[tt_capt[1]]
                if newGroup != g
                    g = newGroup
                end
            end
            SocGroup[i+1, tt+1] = g
        end
    end
    
    return SocGroup
end

"""
    TestMatAsField(TestMat, m)

Convert test matrix to field format with 0-based indexing.
"""
function TestMatAsField(TestMat, m)
    numTests = size(TestMat, 2) - 3
    cols = 3:2+numTests
    
    F = Vector{Matrix{Int}}(undef, m)
    
    # Apply 0-based indexing
    TestMat = @zero_based TestMat
    
    id_i = TestMat[:, 2]
    for i in 0:m-1
        which_i = findall(x -> x == i, id_i)
        Tests_i = TestMat[which_i, :]
        F[i+1] = @zero_based Tests_i[:, cols]
    end
    
    return F
end

"""
    TestMatAsFieldProposal(TestFieldProposal, TestField, TestTimes, xi, xiCan, m)

Update test field proposal for Brock changepoint with 0-based indexing.
"""
function TestMatAsFieldProposal(TestFieldProposal, 
                                TestField,
                                TestTimes,
                                xi, xiCan, m)
    
    for i in 0:m-1
        TestTimes_i = TestTimes[i+1]
        Tests_i = copy(TestFieldProposal[i+1])
        numCapt = size(Tests_i, 1)
        
        if xiCan < xi  # proposing an earlier changepoint
            for irow in 0:numCapt-1
                t = TestTimes_i[irow+1]
                if t >= xiCan && t < xi
                    brock1 = Tests_i[irow+1, 1]
                    Tests_i[irow+1, 1] = Tests_i[irow+1, 2]
                    Tests_i[irow+1, 2] = brock1
                end
            end
        else  # proposing a later changepoint
            for irow in 0:numCapt-1
                t = TestTimes_i[irow+1]
                if t >= xi && t < xiCan
                    brock1 = Tests_i[irow+1, 1]
                    Tests_i[irow+1, 1] = Tests_i[irow+1, 2]
                    Tests_i[irow+1, 2] = brock1
                end
            end
        end
        
        TestFieldProposal[i+1] = Tests_i
    end
end

"""
    TestTimesField(TestMat, m)

Extract test times field with 0-based indexing.
"""
function TestTimesField(TestMat, m)
    F = Vector{Vector{Int}}(undef, m)
    
    # Apply 0-based indexing
    TestMat = @zero_based TestMat
    
    id_i = TestMat[:, 2]
    for i in 0:m-1
        which_i = findall(x -> x == i, id_i)
        Tests_i = TestMat[which_i, :]
        F[i+1] = Tests_i[:, 1]
    end
    
    return F
end

"""
    CheckSensSpec_(numTests, TestField, TestTimes, X)

Check sensitivity and specificity with 0-based indexing.
"""
function CheckSensSpec_(numTests, 
                        TestField, 
                        TestTimes,
                        X)
    
    m = size(X, 1)
    
    # Apply 0-based indexing
    X = @zero_based X
    
    out = zeros(Int, 4, numTests)
    
    for iTest in 0:numTests-1
        numInfecTested = 0
        numInfecPositives = 0
        numSuscepTested = 0
        numSuscepNegatives = 0
        
        for i in 0:m-1
            Tests_i = TestField[i+1]
            testTimes_i = TestTimes[i+1] .- 1
            X_i = X[i+1, :]
            status = X_i[testTimes_i .+ 1]
            
            tests_i = Tests_i[:, iTest+1]
            
            # exposed and infectious individuals
            which_ExpInfec = findall(x -> x == 3 || x == 1, status)
            tests_i_inf = tests_i[which_ExpInfec]
            
            newInfTes = count(x -> x == 0 || x == 1, tests_i_inf)
            newInfPos = count(x -> (status[x] == 3 || status[x] == 1) && (tests_i[x] == 1), eachindex(status))
            
            numInfecTested += newInfTes
            numInfecPositives += newInfPos
            
            # susceptible individuals
            which_suscep = findall(x -> x == 0, status)
            tests_i_suscep = tests_i[which_suscep]
            
            newSuscepTes = count(x -> x == 0 || x == 1, tests_i_suscep)
            newSuscepPos = count(x -> (status[x] == 0) && (tests_i[x] == 0), eachindex(status))
            
            numSuscepTested += newSuscepTes
            numSuscepNegatives += newSuscepPos
        end
        
        out[1, iTest+1] = numInfecPositives
        out[2, iTest+1] = numInfecTested - numInfecPositives
        out[3, iTest+1] = numSuscepNegatives
        out[4, iTest+1] = numSuscepTested - numSuscepNegatives
    end
    
    return out
end

"""
    TrProbDeath_(age, a2, b2, c1, logar)

Calculate transition probability of death with 0-based indexing.
"""
function TrProbDeath_(age, a2, b2, 
                      c1, logar)
    
    # calculating diffExpsLateLife = exp(b2*(age-1)) - exp(b2*age)
    y1 = b2*(age-1)
    y2 = b2*age
    diffExpsLateLife = -exp(y1 + log(exp(y2-y1)-1))
    log_pt = -c1 + (a2/b2)*(diffExpsLateLife)
    out = 1 - exp(log_pt)
    if logar
        out = log(out)
    end
    return out
end

"""
    TrProbSurvive_(age, a2, b2, c1, logar)

Calculate transition probability of survival with 0-based indexing.
"""
function TrProbSurvive_(age, a2, b2, 
                        c1, logar)
    
    # calculating diffExpsLateLife = exp(b2*(age-1)) - exp(b2*age)
    y1 = b2*(age-1)
    y2 = b2*age
    diffExpsLateLife = -exp(y1 + log(exp(y2-y1)-1))
    out = -c1 + (a2/b2)*(diffExpsLateLife)
    if !logar
        out = exp(out)
    end
    return out
end

"""
    fact(k)

Calculate factorial function.
"""
function fact(k)
    return gamma(Float64(k) + 1.0)
end

"""
    ErlangCDF(x_, k, tau)

Calculate Erlang CDF with 0-based indexing.
"""
function ErlangCDF(x_, k, tau)
    x = Float64(x_)
    Q = 1.0
    if k > 1
        for n in 1:k-1
            Q += (1/fact(n))*pow(x/tau, n)
        end
    end
    out = 1.0 - exp(-x/tau)*Q
    return out
end

"""
    DerivErlangCDF(x_, k, tau)

Calculate derivative of Erlang CDF with 0-based indexing.
"""
function DerivErlangCDF(x_, k, tau)
    x = Float64(x_)
    out = -(1/fact(k-1))*pow(x/tau, Float64(k))*exp(-x/tau)
    return out
end

"""
    logitD(x)

Logit function (scalar version).
"""
function logitD(x)
    return log(x / (1 - x))
end

"""
    logisticD(x)

Logistic function (scalar version).
"""
function logisticD(x)
    return exp(x) / (1 + exp(x))
end

"""
    logit(x)

Logit function (vector version).
"""
function logit(x)
    n = length(x)
    logit_vec = zeros(Float64, n)
    for i in 1:n
        logit_vec[i] = log(x[i] / (1 - x[i]))
    end
    return logit_vec
end

"""
    logistic(x)

Logistic function (vector version).
"""
function logistic(x)
    n = length(x)
    logistic_vec = zeros(Float64, n)
    for i in 1:n
        logistic_vec[i] = exp(x[i]) / (1 + exp(x[i]))
    end
    return logistic_vec
end

"""
    sumLogJacobian(xtilde)

Sum log Jacobian for probability transformations.
"""
function sumLogJacobian(xtilde)
    n = length(xtilde)
    sumLogsJac = 0.0
    for i in 1:n
        sumLogsJac += xtilde[i] - 2*log(1+exp(xtilde[i]))
    end
    return sumLogsJac
end

"""
    multrnorm(mu, Sigma)

Generate multivariate normal random vector with 0-based indexing.
"""
function multrnorm(mu, Sigma)
    return rand(MvNormal(mu, Sigma))
end

"""
    logdmultrnorm(x, mu, Sigma)

Calculate multivariate normal density with 0-based indexing.
"""
function logdmultrnorm(x, mu, Sigma)
    return logpdf(MvNormal(mu, Sigma), x)
end

"""
    randu(n)

Generate uniform random vector.
"""
function randu(n)
    out = zeros(Float64, n)
    out .= rand(n)
    return out
end

"""
    logS(age, a2, b2, c1)

Log survival function with 0-based indexing.
"""
function logS(age, a2, b2, c1)
    return -c1*age + (a2/b2)*(1-exp(b2*age))
end

"""
    DlogS_a2(age, a2, b2)

Derivative of log survival wrt a2 with 0-based indexing.
"""
function DlogS_a2(age, a2, b2)
    return (a2/b2)*(1-exp(b2*age))
end

"""
    DlogS_b2(age, a2, b2)

Derivative of log survival wrt b2 with 0-based indexing.
"""
function DlogS_b2(age, a2, b2)
    return (-a2/b2)*(1 - exp(b2*age) + exp(b2*age)*b2*age)
end

"""
    DlogS_c1(age, c1)

Derivative of log survival wrt c1 with 0-based indexing.
"""
function DlogS_c1(age, c1)
    return -c1*age
end

"""
    iFFBScalcLogProbRest(i, ttt, logProbRest, X, SocGroup, LogProbDyingMat, 
                         LogProbSurvMat, logProbStoSgivenSorE, logProbStoEgivenSorE, 
                         logProbStoSgivenI, logProbStoEgivenI, logProbStoSgivenD, 
                         logProbStoEgivenD, logProbEtoE, logProbEtoI)

Calculate log probabilities for rest of states with 0-based indexing.
"""
function iFFBScalcLogProbRest(i, ttt, logProbRest,
                              X, SocGroup,
                              LogProbDyingMat, 
                              LogProbSurvMat,
                              logProbStoSgivenSorE, 
                              logProbStoEgivenSorE, 
                              logProbStoSgivenI, 
                              logProbStoEgivenI, 
                              logProbStoSgivenD, 
                              logProbStoEgivenD, 
                              logProbEtoE, 
                              logProbEtoI)
    
    # Apply 0-based indexing to input arrays
    X = @zero_based X
    SocGroup = @zero_based SocGroup
    LogProbDyingMat = @zero_based LogProbDyingMat
    LogProbSurvMat = @zero_based LogProbSurvMat
    logProbStoSgivenSorE = @zero_based logProbStoSgivenSorE
    logProbStoEgivenSorE = @zero_based logProbStoEgivenSorE
    logProbStoSgivenI = @zero_based logProbStoSgivenI
    logProbStoEgivenI = @zero_based logProbStoEgivenI
    logProbStoSgivenD = @zero_based logProbStoSgivenD
    logProbStoEgivenD = @zero_based logProbStoEgivenD
    logProbRest = @zero_based logProbRest
    
    g = SocGroup[i+1, ttt+1]
    
    state_t = X[i+1, ttt+1]
    state_t1 = X[i+1, ttt+2]
    
    if state_t == 0 && state_t1 == 0
        logProbRest[ttt+1, 1, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbStoSgivenSorE[g+1, ttt+1]
        logProbRest[ttt+1, 2, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbStoSgivenSorE[g+1, ttt+1]
        logProbRest[ttt+1, 3, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbStoSgivenI[g+1, ttt+1]
        logProbRest[ttt+1, 4, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbStoSgivenD[g+1, ttt+1]
    elseif state_t == 0 && state_t1 == 3
        logProbRest[ttt+1, 1, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbStoEgivenSorE[g+1, ttt+1]
        logProbRest[ttt+1, 2, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbStoEgivenSorE[g+1, ttt+1]
        logProbRest[ttt+1, 3, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbStoEgivenI[g+1, ttt+1]
        logProbRest[ttt+1, 4, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbStoEgivenD[g+1, ttt+1]
    elseif state_t == 3 && state_t1 == 3
        logProbRest[ttt+1, 1, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbEtoE
        logProbRest[ttt+1, 2, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbEtoE
        logProbRest[ttt+1, 3, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbEtoE
        logProbRest[ttt+1, 4, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbEtoE
    elseif state_t == 3 && state_t1 == 1
        logProbRest[ttt+1, 1, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbEtoI
        logProbRest[ttt+1, 2, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbEtoI
        logProbRest[ttt+1, 3, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbEtoI
        logProbRest[ttt+1, 4, i+1] = LogProbSurvMat[i+1, ttt+2] + logProbEtoI
    elseif state_t == 1 && state_t1 == 1
        logProbRest[ttt+1, 1, i+1] = LogProbSurvMat[i+1, ttt+2]
        logProbRest[ttt+1, 2, i+1] = LogProbSurvMat[i+1, ttt+2]
        logProbRest[ttt+1, 3, i+1] = LogProbSurvMat[i+1, ttt+2]
        logProbRest[ttt+1, 4, i+1] = LogProbSurvMat[i+1, ttt+2]
    elseif state_t == 0 && state_t1 == 9
        logProbRest[ttt+1, 1, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 2, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 3, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 4, i+1] = LogProbDyingMat[i+1, ttt+2]
    elseif state_t == 1 && state_t1 == 9
        logProbRest[ttt+1, 1, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 2, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 3, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 4, i+1] = LogProbDyingMat[i+1, ttt+2]
    elseif state_t == 3 && state_t1 == 9
        logProbRest[ttt+1, 1, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 2, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 3, i+1] = LogProbDyingMat[i+1, ttt+2]
        logProbRest[ttt+1, 4, i+1] = LogProbDyingMat[i+1, ttt+2]
    elseif state_t == 3 && state_t1 == 0
        error("Some E->S transition was found. This is not allowed in SEI model. Check individual id = $(i+1)")
    elseif state_t == 1 && state_t1 == 3
        error("Some I->E transition was found. This is not allowed in SEI model. Check individual id = $(i+1)")
    elseif state_t == 1 && state_t1 == 0
        error("Some I->S transition was found. This is not allowed in SEI model. Check individual id = $(i+1)")
    end
end

"""
    HMC_2(curLogPars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
          birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
          ageMat, epsilon, epsilonalphas, epsilonbq, epsilontau, epsilonc1, 
          nParsNotGibbs, L, hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)

Hamiltonian Monte Carlo sampler with 0-based indexing.
"""
function HMC_2(curLogPars, G, 
               X, 
               totalNumInfec,
               SocGroup,
               totalmPerGroup,
               birthTimes, 
               startSamplingPeriod,
               lastObsAliveTimes,
               capturesAfterMonit,
               ageMat,
               epsilon, 
               epsilonalphas, 
               epsilonbq, 
               epsilontau,
               epsilonc1, 
               nParsNotGibbs, 
               L, 
               hp_lambda,
               hp_beta,
               hp_q,
               hp_tau,
               hp_a2,
               hp_b2,
               hp_c1,
               k, 
               K)
    
    # Apply 0-based indexing
    X = @zero_based X
    totalNumInfec = @zero_based totalNumInfec
    SocGroup = @zero_based SocGroup
    totalmPerGroup = @zero_based totalmPerGroup
    birthTimes = @zero_based birthTimes
    startSamplingPeriod = @zero_based startSamplingPeriod
    lastObsAliveTimes = @zero_based lastObsAliveTimes
    capturesAfterMonit = @zero_based capturesAfterMonit
    ageMat = @zero_based ageMat
    
    epsilon_vec = zeros(Float64, nParsNotGibbs)
    
    for j in 0:nParsNotGibbs-1
        if j < G
            epsilon_vec[j+1] = epsilonalphas
        elseif (j == G+1) || (j == G+2)
            epsilon_vec[j+1] = epsilonbq
        elseif j == G+3
            epsilon_vec[j+1] = epsilontau
        elseif j == G+6
            epsilon_vec[j+1] = epsilonc1
        else
            epsilon_vec[j+1] = epsilon
        end
    end
    
    out = zeros(Float64, length(curLogPars))
    
    q = copy(curLogPars)
    
    p = randn(length(q))
    curp = copy(p)
    
    p = p + epsilon_vec .* grad_(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                                birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                                hp_lambda, hp_beta, hp_q, hp_tau,
                                hp_a2, hp_b2, hp_c1, k, K) / 2
    
    intL = ceil(rand() * L)
    
    for i in 0:intL-2
        q = q + epsilon_vec .* p
        
        p = p + epsilon_vec .* grad_(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                                     birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                                     hp_lambda, hp_beta, hp_q, hp_tau,
                                     hp_a2, hp_b2, hp_c1, k, K)
    end
    
    q = q + epsilon_vec .* p
    p = p + epsilon_vec .* grad_(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                                 birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                                 hp_lambda, hp_beta, hp_q, hp_tau,
                                 hp_a2, hp_b2, hp_c1, k, K) / 2
    p = -p
    
    ProposedH = logPost_(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                        birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                        hp_lambda, hp_beta, hp_q, hp_tau,
                        hp_a2, hp_b2, hp_c1, k, K) - 0.5*dot(p,p)
    CurrentH = logPost_(curLogPars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                       birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                       hp_lambda, hp_beta, hp_q, hp_tau,
                       hp_a2, hp_b2, hp_c1, k, K) - 0.5*dot(curp,curp)
    
    prob = exp(ProposedH - CurrentH)
    
    alpha = min(1.0, prob)
    u = rand()
    
    if u < alpha
        out = q
    else
        out = curLogPars
    end
    
    return out
end

"""
    RWMH_(can, curLogPars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
          birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
          ageMat, hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)

Random walk Metropolis-Hastings sampler with 0-based indexing.
"""
function RWMH_(can, 
               curLogPars, 
               G, 
               X, 
               totalNumInfec,
               SocGroup,
               totalmPerGroup,
               birthTimes,
               startSamplingPeriod,
               lastObsAliveTimes, 
               capturesAfterMonit,
               ageMat,
               hp_lambda,
               hp_beta,
               hp_q,
               hp_tau,
               hp_a2,
               hp_b2,
               hp_c1,
               k, 
               K)
    
    # Apply 0-based indexing
    X = @zero_based X
    totalNumInfec = @zero_based totalNumInfec
    SocGroup = @zero_based SocGroup
    totalmPerGroup = @zero_based totalmPerGroup
    birthTimes = @zero_based birthTimes
    startSamplingPeriod = @zero_based startSamplingPeriod
    lastObsAliveTimes = @zero_based lastObsAliveTimes
    capturesAfterMonit = @zero_based capturesAfterMonit
    ageMat = @zero_based ageMat
    
    out = similar(curLogPars)
    
    logpostDiff = logPost_(can, G, X, totalNumInfec, 
                          SocGroup, totalmPerGroup,
                          birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, 
                          ageMat, 
                          hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K) - 
                  logPost_(curLogPars, G, X, totalNumInfec, 
                          SocGroup, totalmPerGroup,
                          birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, 
                          ageMat, 
                          hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
    
    prob = exp(logpostDiff)
    
    alpha = min(1.0, prob)
    u = rand()
    
    if u < alpha
        out = can
    else
        out = curLogPars
    end
    
    return out
end

"""
    HMC_thetas_rhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                    TestField, TestTimes, hp_theta, hp_rho, epsilon, L)

HMC sampler for test parameters with 0-based indexing.
"""
function HMC_thetas_rhos(thetas, 
                          rhos, 
                          X,
                          startSamplingPeriod,
                          endSamplingPeriod,
                          TestField, 
                          TestTimes,
                          hp_theta,
                          hp_rho,
                          epsilon,
                          L)
    
    # Apply 0-based indexing
    X = @zero_based X
    startSamplingPeriod = @zero_based startSamplingPeriod
    endSamplingPeriod = @zero_based endSamplingPeriod
    
    numTests = length(thetas)
    out = zeros(Float64, 2*numTests)
    
    # stacking into a unique vector
    cur = zeros(Float64, 2*numTests)
    for iTest in 0:numTests-1
        cur[iTest+1] = thetas[iTest+1]
        cur[iTest+numTests+1] = rhos[iTest+1]
    end
    
    # multivariate normal proposal
    p = randn(2*numTests)
    curp = copy(p)
    
    q_thetas = zeros(Float64, numTests)
    q_rhos = zeros(Float64, numTests)
    for iTest in 0:numTests-1
        q_thetas[iTest+1] = cur[iTest+1]
        q_rhos[iTest+1] = cur[iTest+numTests+1]
    end
    
    q = logit(cur)
    
    p = p + epsilon * gradThetasRhos(q_thetas, 
                                     q_rhos, 
                                     X, 
                                     startSamplingPeriod,
                                     endSamplingPeriod,
                                     TestField,
                                     TestTimes,
                                     hp_theta, 
                                     hp_rho) / 2
    
    intL = ceil(rand() * L)
    
    for i in 0:intL-2
        q = q + epsilon * p
        
        for iTest in 0:numTests-1
            q_thetas[iTest+1] = logisticD(q[iTest+1])
            q_rhos[iTest+1] = logisticD(q[iTest+numTests+1])
        end
        
        p = p + epsilon * gradThetasRhos(q_thetas, 
                                         q_rhos, 
                                         X, 
                                         startSamplingPeriod,
                                         endSamplingPeriod,
                                         TestField,
                                         TestTimes,
                                         hp_theta, 
                                         hp_rho)
    end
    
    q = q + epsilon * p
    for iTest in 0:numTests-1
        q_thetas[iTest+1] = logisticD(q[iTest+1])
        q_rhos[iTest+1] = logisticD(q[iTest+numTests+1])
    end
    p = p + epsilon * gradThetasRhos(q_thetas, 
                                     q_rhos, 
                                     X, 
                                     startSamplingPeriod,
                                     endSamplingPeriod,
                                     TestField,
                                     TestTimes,
                                     hp_theta, 
                                     hp_rho) / 2
    p = -p
    
    ProposedH = logPostThetasRhos(q_thetas, q_rhos, X, startSamplingPeriod, endSamplingPeriod,
                                  TestField, TestTimes, 
                                  hp_theta, hp_rho) - 0.5*dot(p,p)
    CurrentH = logPostThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod,
                                 TestField, TestTimes, 
                                 hp_theta, hp_rho) - 0.5*dot(curp,curp)
    
    prob = exp(ProposedH - CurrentH)
    
    alpha = min(1.0, prob)
    u = rand()
    
    if u < alpha
        out = logistic(q)
    else
        out = cur
    end
    
    return out
end

"""
    RWMH_thetas_rhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                     TestField, TestTimes, hp_theta, hp_rho, Sigma2)

RWMH sampler for test parameters with 0-based indexing.
"""
function RWMH_thetas_rhos(thetas, 
                           rhos, 
                           X,
                           startSamplingPeriod,
                           endSamplingPeriod,
                           TestField, 
                           TestTimes,
                           hp_theta,
                           hp_rho,
                           Sigma2)
    
    # Apply 0-based indexing
    X = @zero_based X
    startSamplingPeriod = @zero_based startSamplingPeriod
    endSamplingPeriod = @zero_based endSamplingPeriod
    
    numTests = length(thetas)
    out = zeros(Float64, 2*numTests)
    
    cur = zeros(Float64, 2*numTests)
    for iTest in 0:numTests-1
        cur[iTest+1] = thetas[iTest+1]
        cur[iTest+numTests+1] = rhos[iTest+1]
    end
    
    curLogit = logit(cur)
    
    # multivariate normal proposal
    canLogit = multrnorm(curLogit, Sigma2)
    
    thetas_canLogit = zeros(Float64, numTests)
    rhos_canLogit = zeros(Float64, numTests)
    for iTest in 0:numTests-1
        thetas_canLogit[iTest+1] = canLogit[iTest+1]
        rhos_canLogit[iTest+1] = canLogit[iTest+numTests+1]
    end
    
    thetas_can = logistic(thetas_canLogit)
    rhos_can = logistic(rhos_canLogit)
    
    logPost_can = logPostThetasRhos(thetas_can, 
                                    rhos_can, 
                                    X, 
                                    startSamplingPeriod,
                                    endSamplingPeriod,
                                    TestField,
                                    TestTimes,
                                    hp_theta, 
                                    hp_rho)
    
    logPost_cur = logPostThetasRhos(thetas, 
                                    rhos, 
                                    X, 
                                    startSamplingPeriod,
                                    endSamplingPeriod,
                                    TestField,
                                    TestTimes,
                                    hp_theta, 
                                    hp_rho)
    
    logPostDiff = logPost_can - logPost_cur
    
    prob = exp(logPostDiff)
    
    alpha = min(1.0, prob)
    u = rand()
    
    if u < alpha
        out = logistic(canLogit)
    else
        out = cur
    end
    
    return out
end

"""
    RWMH_xi(can, cur, hp_xi, TestFieldProposal, TestField, TestTimes, 
            thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)

RWMH sampler for Brock changepoint with 0-based indexing.
"""
function RWMH_xi(can, 
                 cur, 
                 hp_xi,
                 TestFieldProposal,
                 TestField, 
                 TestTimes, 
                 thetas,
                 rhos,
                 phis,
                 X,
                 startSamplingPeriod,
                 endSamplingPeriod)
    
    # Apply 0-based indexing
    X = @zero_based X
    startSamplingPeriod = @zero_based startSamplingPeriod
    endSamplingPeriod = @zero_based endSamplingPeriod
    
    # range where changes in xi modify likelihood
    xiMin = 0
    xiMax = 0
    if can < cur
        xiMin = can
        xiMax = cur
    else
        xiMin = cur
        xiMax = can
    end
    
    logpostDiff = logPostXi(xiMin, xiMax, can, hp_xi, TestFieldProposal, TestTimes, 
                            thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod) - 
                    logPostXi(xiMin, xiMax, cur, hp_xi, TestField, TestTimes,  
                              thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
    
    prob = exp(logpostDiff)
    
    alpha = min(1.0, prob)
    u = rand()
    
    if u < alpha
        out = can
        TestField .= TestFieldProposal
    else
        out = cur
        TestFieldProposal .= TestField
    end
    
    return out
end

"""
    logPost_(logPars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
             birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
             ageMat, hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)

Calculate log posterior with 0-based indexing.
"""
function logPost_(logPars, G, 
                  X, 
                  totalNumInfec,
                  SocGroup,
                  totalmPerGroup,
                  birthTimes,
                  startSamplingPeriod,
                  lastObsAliveTimes, 
                  capturesAfterMonit,
                  ageMat, 
                  hp_lambda,
                  hp_beta,
                  hp_q,
                  hp_tau,
                  hp_a2,
                  hp_b2,
                  hp_c1, 
                  k, 
                  K)
    
    # Apply 0-based indexing
    X = @zero_based X
    totalNumInfec = @zero_based totalNumInfec
    SocGroup = @zero_based SocGroup
    totalmPerGroup = @zero_based totalmPerGroup
    birthTimes = @zero_based birthTimes
    startSamplingPeriod = @zero_based startSamplingPeriod
    lastObsAliveTimes = @zero_based lastObsAliveTimes
    capturesAfterMonit = @zero_based capturesAfterMonit
    ageMat = @zero_based ageMat
    
    m = size(X, 1)
    a = 0.0
    
    lambda = exp(logPars[G+1])
    alpha_js = zeros(Float64, G)
    for g in 0:G-1
        alpha_js[g+1] = exp(logPars[g+1]) * lambda
    end
    b = exp(logPars[G+2])
    q = logisticD(logPars[G+3])
    ql = logPars[G+3]
    tau = exp(logPars[G+4])
    a2 = exp(logPars[G+5])
    b2 = exp(logPars[G+6])
    c1 = exp(logPars[G+7])
    
    logpost = 0.0
    loglik = 0.0
    
    for i in 0:m-1
        mint_i = startSamplingPeriod[i+1]
        
        if birthTimes[i+1] < startSamplingPeriod[i+1]
            loglik += logS(ageMat[i+1, startSamplingPeriod[i+1] - birthTimes[i+1] + 1], a2, b2, c1)
        end
        
        for j in mint_i:lastObsAliveTimes[i+1]-1
            g = SocGroup[i+1, j]
            
            age_ij = Float64(ageMat[i+1, j+1])
            log_pti = TrProbSurvive_(age_ij, a2, b2, c1, true)
            z_t_1 = X[i+1, j]
            z_t = X[i+1, j+1]
            
            if ((z_t_1 == 0 || z_t_1 == 1 || z_t_1 == 3) && z_t == 9)
                log_qti = TrProbDeath_(age_ij, a2, b2, c1, true)
                loglik += log_qti
            elseif z_t_1 == 0 && z_t == 0
                inf_mgt = totalNumInfec[g+1, j] / ((Float64(totalmPerGroup[g+1, j])/K)^q)
                a = alpha_js[g+1]
                loglik += log_pti - a - b*inf_mgt
            elseif z_t_1 == 0 && z_t == 3
                inf_mgt = totalNumInfec[g+1, j] / ((Float64(totalmPerGroup[g+1, j])/K)^q)
                a = alpha_js[g+1]
                loglik += log_pti + log1mexp(a+b*inf_mgt)
            elseif z_t_1 == 3 && z_t == 3
                loglik += log_pti + log(1 - ErlangCDF(1, k, tau/Float64(k)))
            elseif z_t_1 == 3 && z_t == 1
                loglik += log_pti + log(ErlangCDF(1, k, tau/Float64(k)))
            elseif z_t_1 == 1 && z_t == 1
                loglik += log_pti
            end
        end
    end
    
    # add correction term for the captures occurring after the monitoring period
    numRows = size(capturesAfterMonit, 1)
    for ir in 0:numRows-1
        i = capturesAfterMonit[ir+1, 1]
        lastCaptTime = capturesAfterMonit[ir+1, 2]
        
        for j in lastObsAliveTimes[i+1]:lastCaptTime-1
            age_ij = Float64(ageMat[i+1, j+1])
            log_pti = TrProbSurvive_(age_ij, a2, b2, c1, true)
            loglik += log_pti
        end
    end
    
    a_prior = 0.0
    for g in 0:G-1
        a_prior += -exp(logPars[g+1]) + logPars[g+1]
    end
    lambda_prior = -hp_lambda[2]*lambda + log(lambda)
    
    b_prior = logpdf(Gamma(hp_beta[1], 1/hp_beta[2]), b) + log(b)
    q_prior = hp_q[1]*ql - (hp_q[1] + hp_q[2])*log(1 + exp(ql))
    tau_prior = logpdf(Gamma(hp_tau[1], 1/hp_tau[2]), tau) + log(tau)
    a2_prior = logpdf(Gamma(hp_a2[1], 1/hp_a2[2]), a2) + log(a2)
    b2_prior = logpdf(Gamma(hp_b2[1], 1/hp_b2[2]), b2) + log(b2)
    c1_prior = logpdf(Gamma(hp_c1[1], 1/hp_c1[2]), c1) + log(c1)
    
    logprior = a_prior + lambda_prior + b_prior + q_prior + tau_prior +
               a2_prior + b2_prior + c1_prior
    
    logpost = loglik + logprior
    
    return logpost
end

"""
    grad_(logPars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
          birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
          ageMat, hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)

Calculate gradient of log posterior with 0-based indexing.
"""
function grad_(logPars, G, 
               X, 
               totalNumInfec,
               SocGroup,
               totalmPerGroup,
               birthTimes,
               startSamplingPeriod,
               lastObsAliveTimes, 
               capturesAfterMonit,
               ageMat, 
               hp_lambda,
               hp_beta,
               hp_q,
               hp_tau,
               hp_a2,
               hp_b2,
               hp_c1,
               k, 
               K)
    
    # Apply 0-based indexing
    X = @zero_based X
    totalNumInfec = @zero_based totalNumInfec
    SocGroup = @zero_based SocGroup
    totalmPerGroup = @zero_based totalmPerGroup
    birthTimes = @zero_based birthTimes
    startSamplingPeriod = @zero_based startSamplingPeriod
    lastObsAliveTimes = @zero_based lastObsAliveTimes
    capturesAfterMonit = @zero_based capturesAfterMonit
    ageMat = @zero_based ageMat
    
    m = size(X, 1)
    
    likeas = zeros(Float64, length(logPars))
    likelam = 0.0
    likeb = 0.0
    likeq = 0.0
    liketau = 0.0
    likea2 = 0.0
    likeb2 = 0.0
    likec1 = 0.0
    
    a = 0.0
    lambda = exp(logPars[G+1])
    alpha_js = zeros(Float64, G)
    for g in 0:G-1
        alpha_js[g+1] = exp(logPars[g+1]) * lambda
    end
    b = exp(logPars[G+2])
    q = logisticD(logPars[G+3])
    ql = logPars[G+3]
    tau = exp(logPars[G+4])
    a2 = exp(logPars[G+5])
    b2 = exp(logPars[G+6])
    c1 = exp(logPars[G+7])
    
    gradient = zeros(Float64, length(logPars))
    
    for i in 0:m-1
        mint_i = startSamplingPeriod[i+1]
        
        if birthTimes[i+1] < startSamplingPeriod[i+1]
            likea2 += DlogS_a2(ageMat[i+1, startSamplingPeriod[i+1] - birthTimes[i+1] + 1], a2, b2)
            likeb2 += DlogS_b2(ageMat[i+1, startSamplingPeriod[i+1] - birthTimes[i+1] + 1], a2, b2)
            likec1 += DlogS_c1(ageMat[i+1, startSamplingPeriod[i+1] - birthTimes[i+1] + 1], c1)
        end
        
        for j in mint_i:lastObsAliveTimes[i+1]-1
            g = SocGroup[i+1, j]
            
            z_t_1 = X[i+1, j]
            z_t = X[i+1, j+1]
            
            if ((z_t_1 == 0 || z_t_1 == 1 || z_t_1 == 3) && z_t == 9)
                age_ij = Float64(ageMat[i+1, j+1])
                ptiOverqti = TrProbSurvive_(age_ij, a2, b2, c1, false) / 
                             TrProbDeath_(age_ij, a2, b2, c1, false)
                likea2 -= ptiOverqti * Dlogpt_a2(age_ij, a2, b2)
                likeb2 -= ptiOverqti * Dlogpt_b2(age_ij, a2, b2)
                likec1 -= ptiOverqti * Dlogpt_c1(c1)
            elseif z_t_1 == 0 && z_t == 0
                mdenom = Float64(totalmPerGroup[g+1, j]) / K
                inf_mgt = totalNumInfec[g+1, j] / (mdenom^q)
                a = alpha_js[g+1]
                likeas[g+1] -= a
                likelam -= a
                likeb -= b * inf_mgt
                likeq += b * inf_mgt * log(mdenom) * exp(ql) / ((1.0 + exp(ql))^2)
                age_ij = Float64(ageMat[i+1, j+1])
                likea2 += Dlogpt_a2(age_ij, a2, b2)
                likeb2 += Dlogpt_b2(age_ij, a2, b2)
                likec1 += Dlogpt_c1(c1)
            elseif z_t_1 == 1 && z_t == 1
                age_ij = Float64(ageMat[i+1, j+1])
                likea2 += Dlogpt_a2(age_ij, a2, b2)
                likeb2 += Dlogpt_b2(age_ij, a2, b2)
                likec1 += Dlogpt_c1(c1)
            elseif z_t_1 == 0 && z_t == 3
                mdenom = Float64(totalmPerGroup[g+1, j]) / K
                inf_mgt = totalNumInfec[g+1, j] / (mdenom^q)
                a = alpha_js[g+1]
                toBeExp = a + b * inf_mgt
                if toBeExp < 1e-15
                    likeas[g+1] += 1.0
                    likelam += 1.0
                    if totalNumInfec[g+1, j] == 0
                        likeb += 0.0
                    else
                        likeb += 1.0
                    end
                else
                    ratio = exp(-a - b*inf_mgt - log1mexp(a + b*inf_mgt))
                    likeas[g+1] += a * ratio
                    likelam += a * ratio
                    likeb += b * ratio * inf_mgt
                    likeq -= b * ratio * inf_mgt * log(mdenom) * exp(ql) / ((1.0 + exp(ql))^2)
                end
                
                age_ij = Float64(ageMat[i+1, j+1])
                likea2 += Dlogpt_a2(age_ij, a2, b2)
                likeb2 += Dlogpt_b2(age_ij, a2, b2)
                likec1 += Dlogpt_c1(c1)
            elseif z_t_1 == 3 && z_t == 3
                liketau -= DerivErlangCDF(1, k, tau/Float64(k)) / (1 - ErlangCDF(1, k, tau/Float64(k)))
                age_ij = Float64(ageMat[i+1, j+1])
                likea2 += Dlogpt_a2(age_ij, a2, b2)
                likeb2 += Dlogpt_b2(age_ij, a2, b2)
                likec1 += Dlogpt_c1(c1)
            elseif z_t_1 == 3 && z_t == 1
                liketau += DerivErlangCDF(1, k, tau/Float64(k)) / ErlangCDF(1, k, tau/Float64(k))
                age_ij = Float64(ageMat[i+1, j+1])
                likea2 += Dlogpt_a2(age_ij, a2, b2)
                likeb2 += Dlogpt_b2(age_ij, a2, b2)
                likec1 += Dlogpt_c1(c1)
            end
        end
    end
    
    # add correction term for the captures occurring after the monitoring period
    numRows = size(capturesAfterMonit, 1)
    for ir in 0:numRows-1
        i = capturesAfterMonit[ir+1, 1]
        lastCaptTime = capturesAfterMonit[ir+1, 2]
        
        for j in lastObsAliveTimes[i+1]:lastCaptTime-1
            age_ij = Float64(ageMat[i+1, j+1])
            likea2 += Dlogpt_a2(age_ij, a2, b2)
            likeb2 += Dlogpt_b2(age_ij, a2, b2)
            likec1 += Dlogpt_c1(c1)
        end
    end
    
    for g in 0:G-1
        likeas[g+1] += 1.0 - exp(logPars[g+1])
    end
    
    likelam += 1.0 - hp_lambda[2] * lambda
    likeb += hp_beta[1] - hp_beta[2] * b
    likeq += hp_q[1] - (hp_q[1] + hp_q[2]) * q
    liketau += hp_tau[1] - hp_tau[2] * tau
    likea2 += hp_a2[1] - hp_a2[2] * a2
    likeb2 += hp_b2[1] - hp_b2[2] * b2
    likec1 += hp_c1[1] - hp_c1[2] * c1
    
    for g in 0:G-1
        gradient[g+1] = likeas[g+1]
    end
    gradient[G+1] = likelam
    gradient[G+2] = likeb
    gradient[G+3] = likeq
    gradient[G+4] = liketau
    gradient[G+5] = likea2
    gradient[G+6] = likeb2
    gradient[G+7] = likec1
    
    return gradient
end

"""
    logPostThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod,
                      TestField, TestTimes, hp_theta, hp_rho)

Calculate log posterior for test parameters with 0-based indexing.
"""
function logPostThetasRhos(thetas,
                           rhos,
                           X,
                           startSamplingPeriod,
                           endSamplingPeriod,
                           TestField, 
                           TestTimes, 
                           hp_theta,
                           hp_rho)
    
    # Apply 0-based indexing
    X = @zero_based X
    startSamplingPeriod = @zero_based startSamplingPeriod
    endSamplingPeriod = @zero_based endSamplingPeriod
    
    m = size(X, 1)
    
    numTests = size(TestField[1], 2)
    idxTests = 0:numTests-1
    
    logLik = 0.0
    for jj in 0:m-1
        id = jj
        TestMat_i = TestField[jj+1]
        TestTimes_i = TestTimes[jj+1]
        
        t0 = startSamplingPeriod[jj+1]
        maxt_i = endSamplingPeriod[jj+1] - t0
        
        for tt in 0:maxt_i-1
            rows = findall(x -> x - t0 == tt, TestTimes_i)
            
            if length(rows) > 0
                TestMat_i_tt = TestMat_i[rows, :]
                
                for ir in 0:length(rows)-1
                    Tests_ir = TestMat_i_tt[ir+1, :]
                    idx = findall(x -> (Tests_ir[x] == 0 || Tests_ir[x] == 1), idxTests)
                    
                    for ic in 0:length(idx)-1
                        i = idx[ic]
                        if X[id+1, tt+t0] == 3
                            logLik += log((thetas[i+1]*rhos[i+1])^TestMat_i_tt[ir+1, i+1] *
                                        (1-thetas[i+1]*rhos[i+1])^(1-TestMat_i_tt[ir+1, i+1]))
                        elseif X[id+1, tt+t0] == 1
                            logLik += log(thetas[i+1]^TestMat_i_tt[ir+1, i+1] *
                                        (1-thetas[i+1])^(1-TestMat_i_tt[ir+1, i+1]))
                        end
                    end
                end
            end
        end
    end
    
    thetasLogPriorWithJac = 0.0
    rhosLogPriorWithJac = 0.0
    thetaLogit = 0.0
    rhoLogit = 0.0
    for iTest in 0:numTests-1
        thetaLogit = logitD(thetas[iTest+1])
        rhoLogit = logitD(rhos[iTest+1])
        thetasLogPriorWithJac += hp_theta[1] * thetaLogit - 
                                 (hp_theta[1] + hp_theta[2]) * log(1 + exp(thetaLogit))
        rhosLogPriorWithJac += hp_rho[1] * rhoLogit - 
                                (hp_rho[1] + hp_rho[2]) * log(1 + exp(rhoLogit))
    end
    logPost = logLik + thetasLogPriorWithJac + rhosLogPriorWithJac
    
    return logPost
end

"""
    gradThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod,
                   TestField, TestTimes, hp_theta, hp_rho)

Calculate gradient for test parameters with 0-based indexing.
"""
function gradThetasRhos(thetas,
                        rhos,
                        X,
                        startSamplingPeriod,
                        endSamplingPeriod,
                        TestField, 
                        TestTimes, 
                        hp_theta,
                        hp_rho)
    
    # Apply 0-based indexing
    X = @zero_based X
    startSamplingPeriod = @zero_based startSamplingPeriod
    endSamplingPeriod = @zero_based endSamplingPeriod
    
    m = size(X, 1)
    
    numTests = size(TestField[1], 2)
    idxTests = 0:numTests-1
    
    derivloglik = zeros(Float64, 2*numTests)
    
    for jj in 0:m-1
        id = jj
        TestMat_i = TestField[jj+1]
        TestTimes_i = TestTimes[jj+1]
        
        t0 = startSamplingPeriod[jj+1]
        maxt_i = endSamplingPeriod[jj+1] - t0
        
        for tt in 0:maxt_i-1
            rows = findall(x -> x - t0 == tt, TestTimes_i)
            
            if length(rows) > 0
                TestMat_i_tt = TestMat_i[rows, :]
                
                for ir in 0:length(rows)-1
                    Tests_ir = TestMat_i_tt[ir+1, :]
                    idx = findall(x -> (Tests_ir[x] == 0 || Tests_ir[x] == 1), idxTests)
                    
                    for ic in 0:length(idx)-1
                        i = idx[ic]
                        if X[id+1, tt+t0] == 3
                            expThetaTilde = exp(logitD(thetas[i+1]))
                            expRhoTilde = exp(logitD(rhos[i+1]))
                            
                            # derivatives wrt theta
                            derivloglik[i+1] += TestMat_i_tt[ir+1, i+1] * (1 - thetas[i+1]) + 
                                               (1 - TestMat_i_tt[ir+1, i+1]) * (
                                                   expThetaTilde/(1+expThetaTilde+expRhoTilde) - thetas[i+1])
                            
                            # derivatives wrt rho
                            derivloglik[i+numTests+1] += TestMat_i_tt[ir+1, i+1] * (1 - rhos[i+1]) + 
                                                         (1 - TestMat_i_tt[ir+1, i+1]) * (
                                                             expRhoTilde/(1+expThetaTilde+expRhoTilde) - rhos[i+1])
                        elseif X[id+1, tt+t0] == 1
                            # derivatives wrt theta
                            derivloglik[i+1] += TestMat_i_tt[ir+1, i+1] * (1 - thetas[i+1]) -  
                                               (1 - TestMat_i_tt[ir+1, i+1]) * thetas[i+1]
                        end
                    end
                end
            end
        end
    end
    
    derivLogPriorWithJac = zeros(Float64, 2*numTests)
    for iTest in 0:numTests-1
        derivLogPriorWithJac[iTest+1] = hp_theta[1] - (hp_theta[1]+hp_theta[2])*thetas[iTest+1]
        derivLogPriorWithJac[iTest+numTests+1] = hp_rho[1] - (hp_rho[1]+hp_rho[2])*rhos[iTest+1]
    end
    
    grad = derivloglik + derivLogPriorWithJac
    
    return grad
end

"""
    logPostXi(xiMin, xiMax, xi, hp_xi, TestField_, TestTimes, thetas, rhos, phis,
              X, startSamplingPeriod, endSamplingPeriod)

Calculate log posterior for Brock changepoint with 0-based indexing.
"""
function logPostXi(xiMin,
                   xiMax,
                   xi,
                   hp_xi,
                   TestField_, 
                   TestTimes,
                   thetas,
                   rhos,
                   phis,
                   X,
                   startSamplingPeriod,
                   endSamplingPeriod)
    
    # Apply 0-based indexing
    X = @zero_based X
    startSamplingPeriod = @zero_based startSamplingPeriod
    endSamplingPeriod = @zero_based endSamplingPeriod
    
    m = size(X, 1)
    
    numBrockTests = 2
    idxTests = 0:numBrockTests-1
    
    logLik = 0.0
    for jj in 0:m-1
        id = jj
        TestMat_i = TestField_[jj+1]
        TestTimes_i = TestTimes[jj+1]
        
        t0 = startSamplingPeriod[jj+1]
        maxt_i = endSamplingPeriod[jj+1] - t0
        
        for tt in 0:maxt_i-1
            if (tt+t0 >= xiMin) && (tt+t0 < xiMax)
                rows = findall(x -> x - t0 == tt, TestTimes_i)
                
                if length(rows) > 0
                    TestMat_i_tt_allTests = TestMat_i[rows, :]
                    TestMat_i_tt = TestMat_i_tt_allTests[:, idxTests .+ 1] # Brock test columns
                    
                    for ir in 0:length(rows)-1
                        Tests_ir = TestMat_i_tt[ir+1, :]
                        idx = findall(x -> (Tests_ir[x] == 0 || Tests_ir[x] == 1), idxTests)
                        
                        for ic in 0:length(idx)-1
                            i = idx[ic]
                            
                            if X[id+1, tt+t0] == 0
                                logLik += log((1-phis[i+1])^Tests_ir[i+1] *
                                            phis[i+1]^(1-Tests_ir[i+1]))
                            elseif X[id+1, tt+t0] == 3
                                logLik += log((thetas[i+1]*rhos[i+1])^Tests_ir[i+1] *
                                            (1-thetas[i+1]*rhos[i+1])^(1-Tests_ir[i+1]))
                            elseif X[id+1, tt+t0] == 1
                                logLik += log(thetas[i+1]^Tests_ir[i+1] *
                                            (1-thetas[i+1])^(1-Tests_ir[i+1]))
                            end
                        end
                    end
                end
            end # end if
        end
    end
    
    xiLogPrior = logpdf(Normal(hp_xi[1], hp_xi[2]), xi)
    
    logPost = logLik + xiLogPrior
    
    return logPost
end

# Additional helper functions that might be needed
"""
    Dlogpt_a2(age, a2, b2)

Derivative of log survival probability wrt a2.
"""
function Dlogpt_a2(age, a2, b2)
    return (1/b2)*(1-exp(b2*age))
end

"""
    Dlogpt_b2(age, a2, b2)

Derivative of log survival probability wrt b2.
"""
function Dlogpt_b2(age, a2, b2)
    return (-a2/(b2^2))*(1 - exp(b2*age) + exp(b2*age)*b2*age)
end

"""
    Dlogpt_c1(c1)

Derivative of log survival probability wrt c1.
"""
function Dlogpt_c1(c1)
    return -1.0
end

"""
    pow(x, y)

Power function for compatibility.
"""
pow(x, y) = x^y
pow(x, y) = x^y
