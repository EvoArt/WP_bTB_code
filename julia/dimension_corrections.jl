"""
    dimension_corrections.jl

Critical corrections for dimension issues in the C++ to Julia conversions.
These address potential problems with row/column vectors and matrix operations.
"""
# Mathematical helper functions
# Note: logit and logistic are provided by StatsFuns package

"""
Create a vector indicating season for each time point with 1-based indexing.
"""
function MakeSeasonVec_(numSeasons, seasonStart, maxt)
    seasonsVec = 1:numSeasons
    rows = findall(x -> x == seasonStart, seasonsVec)
    if length(rows) == 0
        error("seasonStart must be an integer from {1, ..., numSeasons}.")
    end
    
    seasonVec = ones(Int, maxt)
    seasonVec[1] = seasonStart  # 1-based indexing
    for tt in 2:maxt
        if seasonVec[tt-1] < numSeasons
            seasonVec[tt] = seasonVec[tt-1] + 1
        else
            seasonVec[tt] = 1  # Wrap around to season 1
        end
    end
    
    return seasonVec
end

"""
Locate individuals in social groups over time with 1-based indexing.
"""
function LocateIndiv(TestMat, birthTimes)
    maxt = Int64(maximum(TestMat[:, 1][.!isnan.(TestMat[:, 1])]))
    m = Int64(maximum(TestMat[:, 2][.!isnan.(TestMat[:, 2])]))
    
    SocGroup = zeros(Int, m, maxt)
    
    id_i = TestMat[:, 2]
    for i in 1:m
        which_i = findall(x -> x == i, id_i)
        Tests_i = TestMat[which_i, :]
        times_i = Tests_i[:, 1]
        groups_i = Tests_i[:, 3]
        
        tt0 = max(1, birthTimes[i]) 
        
        firstcapttime = minimum(times_i)
        which_row = findall(x -> x == firstcapttime, times_i)
        g = groups_i[which_row[1]]  # first group it belongs to
        
        for tt in tt0:maxt
            # check if moved to another group
            tt_capt = findall(x -> x == tt, times_i)
            if length(tt_capt) > 0
                newGroup = groups_i[tt_capt[1]]
                if newGroup != g
                    g = newGroup
                end
            end
            SocGroup[i, tt] = g  # Use 1-based indexing
        end
    end
    
    return SocGroup
end

"""
Convert test matrix to field format with 1-based indexing.
"""
function TestMatAsField(TestMat, m)
    numTests = size(TestMat, 2) - 3
    cols = 4:(3+numTests)  # Test result columns (1-based)
    
    F = Vector{Matrix{Int}}(undef, m)
    
    id_i = TestMat[:, 2]
    for i in 1:m
        which_i = findall(x -> x == i, id_i)
        if length(which_i) > 0
            Tests_i = TestMat[which_i, :]
            F[i] = Tests_i[:, cols]  # Use 1-based indexing
        else
            # Handle case where individual has no test records
            F[i] = zeros(Int, 0, numTests)
        end
    end
    
    return F
end

"""
Extract test times field with 1-based indexing.
"""
function TestTimesField(TestMat, m)
    F = Vector{Vector{Int}}(undef, m)
    
    id_i = TestMat[:, 2]
    for i in 1:m
        which_i = findall(x -> x == i, id_i)
        if length(which_i) > 0
            Tests_i = TestMat[which_i, :]
            F[i] = Int.(round.(Tests_i[:, 1]))  # Convert Float64 to Int
        else
            F[i] = Int[]
        end
    end
    
    return F
end

## CRITICAL ISSUE 1: Field Operations in TestMatAsField

"""
CORRECTED VERSION - Original had potential dimension issues
"""
function TestMatAsField_CORRECTED(TestMat, m)
    numTests = size(TestMat, 2) - 3
    cols = 4:(3+numTests)  # Test result columns (1-based)
    
    F = Vector{Matrix{Float64}}(undef, m)
    
    id_i = TestMat[:, 2]
    for i in 1:m
        which_i = findall(x -> x == i, id_i)
        #if length(which_i) > 0
            Tests_i = TestMat[which_i, :]
            test_data = Tests_i[:, cols]
            
            # Convert Float64 to Int
            F[i] = test_data#Int.(round.(test_data))
        #else
        #    # Handle case where individual has no test records
        #    F[i] = zeros(Int, 0, numTests)
        #end
    end
    
    return F
end

## CRITICAL ISSUE 2: Vector vs Matrix in gradient functions

"""
CORRECTED VERSION - Ensure proper vector dimensions
"""
function grad__CORRECTED(logPars, G, 
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
    
    # Use 1-based indexing throughout
    
    m = size(X, 1)
    
    # CRITICAL: Initialize as Vector{Float64} not Matrix
    likeas = zeros(Float64, length(logPars))  # This is correct - 1D vector
    likelam = 0.0
    likeb = 0.0
    likeq = 0.0
    liketau = 0.0
    likea2 = 0.0
    likeb2 = 0.0
    likec1 = 0.0
    
    a = 0.0
    lambda = exp(logPars[G+1])
    alpha_js = zeros(Float64, G)  # This is correct - 1D vector
    
    for g in 1:G
        alpha_js[g] = exp(logPars[g]) * lambda
    end
    b = exp(logPars[G+2])
    q = logistic(logPars[G+3])
    ql = logPars[G+3]
    tau = exp(logPars[G+4])
    a2 = exp(logPars[G+5])
    b2 = exp(logPars[G+6])
    c1 = exp(logPars[G+7])
    
    # CRITICAL: Initialize as Vector{Float64} not Matrix
    gradient = zeros(Float64, length(logPars))  # This is correct - 1D vector
    
    # ... rest of function remains the same
    
    # Final gradient assembly - CRITICAL: Check indexing
    for g in 1:G
        gradient[g] = likeas[g]  # Vector indexing is correct
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

## CRITICAL ISSUE 3: Matrix indexing in CheckSensSpec

"""
CORRECTED VERSION - Check matrix indexing
"""

function CheckSensSpec__CORRECTED(numTests, TestField, TestTimes, X)

    ### Is the logic of this function correct??
    ### seems weird that suseptpos is 0,0. (pos should be 1?)
    m = size(X, 1)
    out = zeros(Int, 4, numTests)
     #Threads.@threads for i in 1:m
    for i in 1:m
        Tests_i = TestField[i]               # matrix (individual × tests)
            testTimes_i = TestTimes[i]     # convert to 0-based like C++
            X_i = X[i, :]                        # row vector
            status = X_i[testTimes_i]       # back to 1-based
        for iTest in 1:numTests
            numInfecTested = 0
            numInfecPositives = 0
            numSuscepTested = 0
            numSuscepNegatives = 0
            tests_i = @views(Tests_i[:, iTest])          # column of tests for this test type

            newInfPos = 0
            newInfTes = 0
            newSuscepTes = 0
            newSuscepPos = 0
            #### assuming x is always 1 or 0, this can be further
            #### streamlined
            for (idx,s) in enumerate(status)
                tests_i_idx = tests_i[idx]
                if s == 0.0
                    if (tests_i_idx == 0.0) 
                        newSuscepTes+=1
                        newSuscepPos += 1
                    elseif tests_i_idx == 1.0
                        newSuscepTes += 1
                    end
                elseif (s == 3.0) || (s == 1.0) 
                    if (tests_i_idx == 1.0) 
                        newInfPos +=1
                        newInfTes += 1
                    elseif tests_i_idx == 0.0
                        newInfTes += 1
                    end
                end

            end
                
            # Exposed (3) or Infectious (1)
            #which_ExpInfec = findall(x -> x == 3 || x == 1, status)
            #tests_i_inf = tests_i[which_ExpInfec]

            #newInfTes = Base.count(x -> x == 0 || x == 1, tests_i_inf)
            #newInfPos = Base.count(j -> (status[j] == 3 || status[j] == 1) && tests_i[j] == 1,
            #                  eachindex(status))

            numInfecTested += newInfTes
            numInfecPositives += newInfPos

            # Susceptible (0)
            #which_suscep = findall(x -> x == 0, status)
            #tests_i_suscep = tests_i[which_suscep]

            #newSuscepTes = Base.count(x -> x == 0 || x == 1, tests_i_suscep)
            #newSuscepPos = Base.count(j -> status[j] == 0 && tests_i[j] == 0,
            #                     eachindex(status))

            numSuscepTested += newSuscepTes
            numSuscepNegatives += newSuscepPos

            out[1, iTest] += numInfecPositives
            out[2, iTest] += numInfecTested - numInfecPositives
            out[3, iTest] += numSuscepNegatives
            out[4, iTest] += numSuscepTested - numSuscepNegatives
        end

    end
    return out
end

## CRITICAL ISSUE 4: Vector operations in HMC/RWMH

"""
CORRECTED VERSION - Ensure vector operations are consistent
"""
function HMC_2_CORRECTED(curLogPars, G, 
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
    
    # Apply 0-based indexing to input arrays
    # Use 1-based indexing throughout
    
    # CRITICAL: Vector initialization
    epsilon_vec = zeros(Float64, nParsNotGibbs)
    
    for j in 1:nParsNotGibbs
        if j <= G
            epsilon_vec[j] = epsilonalphas
        elseif (j == G+1) || (j == G+2)
            epsilon_vec[j] = epsilonbq
        elseif j == G+3
            epsilon_vec[j] = epsilontau
        elseif j == G+6
            epsilon_vec[j] = epsilonc1
        else
            epsilon_vec[j] = epsilon
        end
    end
    
    # CRITICAL: Vector operations
    out = zeros(Float64, length(curLogPars))
    q = copy(curLogPars)
    p = randn(length(q))
    curp = copy(p)
    
    # Element-wise operations - CRITICAL: use broadcasting
    p = p .+ epsilon_vec .* grad__CORRECTED(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                                birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                                hp_lambda, hp_beta, hp_q, hp_tau,
                                hp_a2, hp_b2, hp_c1, k, K) ./ 2
    
    intL = ceil(rand() * L)
    
    for i in 1:intL-1
        # CRITICAL: Broadcasting for vector operations
        q = q .+ epsilon_vec .* p
        p = p .+ epsilon_vec .* grad__CORRECTED(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                                     birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                                     hp_lambda, hp_beta, hp_q, hp_tau,
                                     hp_a2, hp_b2, hp_c1, k, K)
    end
    
    q = q .+ epsilon_vec .* p
    p = p .+ epsilon_vec .* grad__CORRECTED(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                                 birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                                 hp_lambda, hp_beta, hp_q, hp_tau,
                                 hp_a2, hp_b2, hp_c1, k, K) ./ 2
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

function ObsProcess!(corrector,
                     t0,
                     endTime,
                     id,
                     CaptHist,
                     TestMat_i,
                     TestTimes_i,
                     etas,
                     thetas,
                     rhos,
                     phis,
                     seasonVec)

    numTests = size(TestMat_i, 2)
    idxTests = collect(1:numTests)
    rows = Vector{Int}(undef, length(TestTimes_i))
    maxt_i = endTime - t0

    for tt = 0:maxt_i
        eta = etas[seasonVec[tt + t0]]

        if CaptHist[id, tt + t0] == 0
            corrector[tt + t0, :] .= (1 - eta, 1 - eta, 1 - eta, 1.0)

        else
            corrector[tt + t0, :] .= (eta, eta, eta, 0.0)

            #rows = findall(x -> x == tt + 1, TestTimes_i .- t0)
            n_row = 0
            for it in 1:size(TestTimes_i, 1)
                if TestTimes_i[it] - t0 == tt + 1
                    n_row += 1
                    rows[n_row] = it
                end
            end

            if n_row > 0
      
                #TestMat_i_tt = TestMat_i[rows, :]
                TestMat_i_tt = @views(TestMat_i[rows[1:n_row], :])
                productIfSuscep = 1.0
                productIfExposed = 1.0
                productIfInfectious = 1.0

                # === EXACT MATCH TO ORIGINAL NESTED LOOPS ===
                for ir in 1:n_row
                    Tests_ir = TestMat_i_tt[ir, :]

                    idx = findall(i -> Tests_ir[i] == 0 || Tests_ir[i] == 1, idxTests)

                    for i in idx
                        x = TestMat_i_tt[ir, i]  # 0 or 1

                        productIfSuscep *= ( (1 - phis[i])^x * phis[i]^(1 - x) )
                        productIfExposed *= ( (thetas[i] * rhos[i])^x *
                                              (1 - thetas[i] * rhos[i])^(1 - x) )
                        productIfInfectious *= ( thetas[i]^x * (1 - thetas[i])^(1 - x) )
                    end
                end

                corrector[tt + t0, 1] *= productIfSuscep
                corrector[tt + t0, 2] *= productIfExposed
                corrector[tt + t0, 3] *= productIfInfectious
            end
        end
    end
#println("corrector = $(corrector)")
    return nothing
end

function normTransProbRest(logProbs)  
  
  n = length(logProbs)
  B = maximum(logProbs)
  lse = B + logsumexp(logProbs .- B)
  out = zeros(n)
  for j in 1:n
    out[j] = exp(logProbs[j] - lse)
  end

  return out
  end

  function normTransProbRest!(logProbs,log_probs_minus_B)  
  
  n = length(logProbs)
  B = maximum(logProbs)
  log_probs_minus_B .= logProbs .- B
  lse = B + logsumexp(log_probs_minus_B)
  for j in 1:n
    logProbs[j] = exp(logProbs[j] - lse)
  end

  end

  function iFFBScalcLogProbRest(
    i,
    ttt,
    logProbRest,             # 3D array: (time, stateIndex, individual)
    X,                       # infection state matrix
    SocGroup,                # social group membership matrix
    LogProbDyingMat,
    LogProbSurvMat,
    logProbStoSgivenSorE,
    logProbStoEgivenSorE,
    logProbStoSgivenI,
    logProbStoEgivenI,
    logProbStoSgivenD,
    logProbStoEgivenD,
    logProbEtoE,
    logProbEtoI
)

    g = SocGroup[i, ttt]               # social group index
    state_t  = X[i, ttt]
    state_t1 = X[i, ttt + 1]

    if state_t == 0 && state_t1 == 0
        logProbRest[ttt, 1, i] = LogProbSurvMat[i, ttt + 1] + logProbStoSgivenSorE[g, ttt]
        logProbRest[ttt, 2, i] = LogProbSurvMat[i, ttt + 1] + logProbStoSgivenSorE[g, ttt]
        logProbRest[ttt, 3, i] = LogProbSurvMat[i, ttt + 1] + logProbStoSgivenI[g, ttt]
        logProbRest[ttt, 4, i] = LogProbSurvMat[i, ttt + 1] + logProbStoSgivenD[g, ttt]

    elseif state_t == 0 && state_t1 == 3
        logProbRest[ttt, 1, i] = LogProbSurvMat[i, ttt + 1] + logProbStoEgivenSorE[g, ttt]
        logProbRest[ttt, 2, i] = LogProbSurvMat[i, ttt + 1] + logProbStoEgivenSorE[g, ttt]
        logProbRest[ttt, 3, i] = LogProbSurvMat[i, ttt + 1] + logProbStoEgivenI[g, ttt]
        logProbRest[ttt, 4, i] = LogProbSurvMat[i, ttt + 1] + logProbStoEgivenD[g, ttt]

    elseif state_t == 3 && state_t1 == 3
        for st in 1:4
            logProbRest[ttt, st, i] = LogProbSurvMat[i, ttt + 1] + logProbEtoE
        end

    elseif state_t == 3 && state_t1 == 1
        for st in 1:4
            logProbRest[ttt, st, i] = LogProbSurvMat[i, ttt + 1] + logProbEtoI
        end

    elseif state_t == 1 && state_t1 == 1
        for st in 1:4
            logProbRest[ttt, st, i] = LogProbSurvMat[i, ttt + 1]
        end

    elseif state_t in (0,1,3) && state_t1 == 9
        for st in 1:4
            logProbRest[ttt, st, i] = LogProbDyingMat[i, ttt + 1]
        end

    elseif state_t == 3 && state_t1 == 0
        # E→S transition - IMPOSSIBLE in SEI model, assign -Inf probability
        println("DEBUG: E→S transition found for individual i=$i at time ttt=$ttt (state_t=$state_t, state_t1=$state_t1)")
        println("DEBUG: X[i, ttt] = $(X[i, ttt]), X[i, ttt+1] = $(X[i, ttt+1])")
        for st in 1:4
            logProbRest[ttt, st, i] = -Inf
        end

    elseif state_t == 1 && state_t1 == 3
        # I→E transition - IMPOSSIBLE in SEI model, assign -Inf probability  
        for st in 1:4
            logProbRest[ttt, st, i] = -Inf
        end

    elseif state_t == 1 && state_t1 == 0
        # I→S transition - IMPOSSIBLE in SEI model, assign -Inf probability
        for st in 1:4
            logProbRest[ttt, st, i] = -Inf
        end
    end
end

"""
    TrProbDeath_(age, a2, b2, c1, logar=false)

Calculate transition probability of death given age and Gompertz parameters.
Ported from C++ TrProbDeath_.cpp
"""
function TrProbDeath_(age, a2, b2, c1, logar=false)
    # calculating diffExpsLateLife = exp(b2*(age-1)) - exp(b2*age)
    y1 = b2 * (age - 1)
    y2 = b2 * age
    diffExpsLateLife = -exp(y1 + log(exp(y2 - y1) - 1))
    log_pt = -c1 + (a2 / b2) * (diffExpsLateLife)
    # Alternative formulation: double log_pt = - c1 + (a2/b2)*( exp(b2*(age-1)) - exp(b2*age) );
    out = 1 - exp(log_pt)
    
    if logar
        out = log(out)
    end
    
    return out
end

"""
    TrProbSurvive_(age, a2, b2, c1, logar=false)

Calculate transition probability of survival given age and Gompertz parameters.
Ported from C++ TrProbSurvive_.cpp
"""
function TrProbSurvive_(age, a2, b2, c1, logar=true)
    # calculating diffExpsLateLife = exp(b2*(age-1)) - exp(b2*age)
    y1 = b2 * (age - 1)
    y2 = b2 * age
    diffExpsLateLife = -exp(y1 + log(exp(y2 - y1) - 1))
    out = -c1 + (a2 / b2) * (diffExpsLateLife)
    # Alternative formulation: double out = - c1 + (a2/b2)*( exp(b2*(age-1)) - exp(b2*age) );
    
    if !logar
        out = exp(out)
    end
    
    return out
end

"""
    logPost_(logPars, G, X, totalNumInfec, SocGroup, totalmPerGroup, 
            birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
            ageMat, hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)

Calculate log posterior probability.
Ported from C++ logPost_.cpp
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
    
    m = size(X, 1)
    
    # Extract parameters - match C++ exactly
    lambda = exp(logPars[G+1])  # C++: logPars[G]
    alpha_js = zeros(Float64, G)
    for g in 1:G
        alpha_js[g] = exp(logPars[g]) * lambda  # C++: g from 0 to G-1
    end
    b = exp(logPars[G+2])  # C++: logPars[G+1]
    q = logistic(logPars[G+3])  # C++: logisticD(logPars[G+2])
    ql = logPars[G+3]  # C++: logPars[G+2]
    tau = exp(logPars[G+4])  # C++: logPars[G+3]
    a2 = exp(logPars[G+5])  # C++: logPars[G+4]
    b2 = exp(logPars[G+6])  # C++: logPars[G+5]
    c1 = exp(logPars[G+7])  # C++: logPars[G+6]
    
    loglik = 0.0
    
    # Main likelihood calculation
    for i in 1:m  # C++: i from 0 to m-1
        mint_i = startSamplingPeriod[i]  # C++: startSamplingPeriod[i]
        
        # Birth before sampling period correction
        if birthTimes[i] < startSamplingPeriod[i]
            # C++: ageMat(i, startSamplingPeriod[i] - birthTimes[i])
            age_offset = ageMat[i, startSamplingPeriod[i] - birthTimes[i] + 1]
            loglik += logS(age_offset, a2, b2, c1)
        end
        
        # Main loop over time periods
        for j in mint_i:(lastObsAliveTimes[i]-1)  # C++: j from mint_i to lastObsAliveTimes[i]-1
            g = SocGroup[i, j]  # C++: SocGroup(i, j-1)
            
            age_ij = Float64(ageMat[i, j+1])  # C++: ageMat(i,j)
            log_pti = TrProbSurvive_(age_ij, a2, b2, c1, true)
            z_t_1 = X[i, j]  # C++: X(i,j-1)
            z_t = X[i, j+1]  # C++: X(i,j)
            
            if ((z_t_1 == 0) || (z_t_1 == 1) || (z_t_1 == 3)) && (z_t == 9)
                log_qti = TrProbDeath_(age_ij, a2, b2, c1, true)
                loglik += log_qti
            elseif (z_t_1 == 0) && (z_t == 0)
                # C++: totalNumInfec(g-1, j-1), totalmPerGroup(g-1, j-1)
                inf_mgt = totalNumInfec[g, j] / ((Float64(totalmPerGroup[g, j])/K)^q)
                a = alpha_js[g]  # C++: alpha_js[g-1]
                loglik += log_pti - a - b * inf_mgt
            elseif (z_t_1 == 0) && (z_t == 3)
                inf_mgt = totalNumInfec[g, j] / ((Float64(totalmPerGroup[g, j])/K)^q)
                a = alpha_js[g]
                loglik += log_pti + safe_log1mexp(a + b * inf_mgt)
            elseif (z_t_1 == 3) && (z_t == 3)
                loglik += log_pti + log(1 - cdf(Erlang(k, k/tau), 1))
            elseif (z_t_1 == 3) && (z_t == 1)
                loglik += log_pti + log(cdf(Erlang(k, k/tau), 1))
            elseif (z_t_1 == 1) && (z_t == 1)
                loglik += log_pti
            end
        end
    end
    
    # Correction for captures after monitoring period
    numRows = size(capturesAfterMonit, 1)
    for ir in 1:numRows  # C++: ir from 0 to numRows-1
        i = capturesAfterMonit[ir, 1]  # C++: capturesAfterMonit(ir, 0)-1L
        lastCaptTime = capturesAfterMonit[ir, 2]
        
        for j in lastObsAliveTimes[i]:(lastCaptTime-1)  # C++: j from lastObsAliveTimes[i] to lastCaptTime-1
            age_ij = Float64(ageMat[i, j+1])
            log_pti = TrProbSurvive_(age_ij, a2, b2, c1, true)
            loglik += log_pti
        end
    end
    
    # Prior calculations
    a_prior = 0.0
    for g in 1:G  # C++: g from 0 to G-1
        a_prior += -exp(logPars[g]) + logPars[g]
    end
    lambda_prior = -hp_lambda[2] * lambda + log(lambda)  # C++: hp_lambda[1]
    
    b_prior = logpdf(Gamma(hp_beta[1], hp_beta[2]), b) + log(b)  # C++: hp_beta[0], 1/hp_beta[1]
    q_prior = hp_q[1] * ql - (hp_q[1] + hp_q[2]) * log(1 + exp(ql))  # C++: hp_q[0], hp_q[1]
    tau_prior = logpdf(Gamma(hp_tau[1], hp_tau[2]), tau) + log(tau)
    a2_prior = logpdf(Gamma(hp_a2[1], hp_a2[2]), a2) + log(a2)
    b2_prior = logpdf(Gamma(hp_b2[1], hp_b2[2]), b2) + log(b2)
    c1_prior = logpdf(Gamma(hp_c1[1], hp_c1[2]), c1) + log(c1)
    
    logprior = a_prior + lambda_prior + b_prior + q_prior + tau_prior +
               a2_prior + b2_prior + c1_prior
    
    logpost = loglik + logprior
    
    return logpost
end

"""
    logS(age, a2, b2, c1)

Calculate log survival probability.
Ported from C++ logS function.
"""
function logS(age, a2, b2, c1)
    return TrProbSurvive_(age, a2, b2, c1, true)
end

"""
    logisticD(x)

Logistic function (derivative version).
Ported from C++ logisticD function.
"""
function logisticD(x)
    return 1.0 / (1.0 + exp(-x))
end

"""
    multrnorm(mu, Sigma)

Generate multivariate normal random vector.
Ported from C++ multrnorm.cpp
"""
function multrnorm(mu::Vector{Float64}, Sigma::Matrix{Float64})
    # Use Julia's built-in multivariate normal distribution
    d = MvNormal(mu, Sigma)
    return rand(d)
end

"""
    RWMH_(can, curLogPars, G, X, totalNumInfec, SocGroup, totalmPerGroup,
          birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit,
          ageMat, hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)

Random Walk Metropolis-Hastings algorithm.
Ported from C++ RWMH_.cpp
"""
function RWMH_(can, curLogPars, G, 
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
    
    out = copy(curLogPars)
    
    # Calculate log posterior difference
    logpostCan = logPost_(can, G, X, totalNumInfec, 
                          SocGroup, totalmPerGroup,
                          birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, 
                          ageMat, 
                          hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
    
    logpostCur = logPost_(curLogPars, G, X, totalNumInfec, 
                          SocGroup, totalmPerGroup,
                          birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, 
                          ageMat, 
                          hp_lambda, hp_beta, hp_q, hp_tau, hp_a2, hp_b2, hp_c1, k, K)
    
    logpostDiff = logpostCan - logpostCur
    
    # Accept or reject
    if log(rand()) < logpostDiff
        out = copy(can)
    end
    
    return out
end

"""
    safe_log1mexp(x)

Safe log1mexp function for infection rates.
Computes log(1 - exp(-x)) for positive x values.
Usage: safe_log1mexp(infection_rate) where infection_rate > 0
"""
function safe_log1mexp(x)
        return StatsFuns.log1mexp(-x)  # Pass negative to StatsFuns
end

"""
    logPostThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho)

Compute log posterior for thetas and rhos parameters.
Ported from C++ logPostThetasRhos.cpp
"""
function logPostThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho)
    
    m = size(X, 1)
    numTests = size(TestField[1], 2)
    idxTests = 1:numTests  # Julia is 1-based
    rows = Vector{Int64}(undef, size(X, 2))
    valid_tests = Vector{Int64}(undef, numTests)
    
    logLik = 0.0
    
    for jj in 1:m  # Julia is 1-based, C++ was 0-based
        id = jj  # C++: id = jj+1L
        TestMat_i = TestField[jj]
        TestTimes_i = TestTimes[jj]
        
        t0 = startSamplingPeriod[jj] - 1  # C++: startSamplingPeriod[jj] - 1L
        maxt_i = endSamplingPeriod[jj] - t0
        
        for tt in 0:maxt_i-1  # C++: tt=0; tt<maxt_i; tt++
            # Find rows where TestTimes_i - t0 == tt+1
            n_row = 0
            for it in 1:size(TestTimes_i, 1)
                if TestTimes_i[it] - t0 == tt + 1
                    n_row += 1
                    rows[n_row] = it
                end
            end
            
            if n_row > 0
                #### Whay allocate whole new array?
                TestMat_i_tt = @views(TestMat_i[rows[1:n_row], :])
                
                for ir_idx in 1:n_row
                    Tests_ir = TestMat_i_tt[ir_idx, :]
                    #valid_tests = findall(x -> x == 0 || x == 1, Tests_ir)
                    n_test = 0
                    for it in 1:numTests
                        if (Tests_ir[it] == 0.0) || (Tests_ir[it] == 1.0)
                            n_test += 1
                            valid_tests[n_test] = it
                        end
                    end
                    
                    #for ic in valid_tests[1:n_test]
                    for ic in 1:n_test#valid_tests[1:n_test]
                        i = valid_tests[ic]  # Julia is 1-based
                        
                        if X[id, tt + t0 + 1] == 3  # Exposed state (C++: X(id-1,tt+t0)==3L)
                            test_result = Tests_ir[i]
                            logLik += log((thetas[i] * rhos[i])^test_result * 
                                        (1 - thetas[i] * rhos[i])^(1 - test_result))
                        elseif X[id, tt + t0 + 1] == 1  # Infectious state (C++: X(id-1,tt+t0)==1L)
                            test_result = Tests_ir[i]
                            logLik += log(thetas[i]^test_result * 
                                        (1 - thetas[i])^(1 - test_result))
                        end
                    end
                end
            end
        end
    end
    
    # Log prior with Jacobian
    thetasLogPriorWithJac = 0.0
    rhosLogPriorWithJac = 0.0
    
    for iTest in 1:numTests
        thetaLogit = logit(thetas[iTest])
        rhoLogit = logit(rhos[iTest])
        
        thetasLogPriorWithJac += hp_theta[1] * thetaLogit - 
            (hp_theta[1] + hp_theta[2]) * log(1 + exp(thetaLogit))
        rhosLogPriorWithJac += hp_rho[1] * rhoLogit - 
            (hp_rho[1] + hp_rho[2]) * log(1 + exp(rhoLogit))
    end
    
    logPost = logLik + thetasLogPriorWithJac + rhosLogPriorWithJac
    return logPost
end

"""
    gradThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho)

Compute gradient of log posterior for thetas and rhos parameters.
Ported from C++ gradThetasRhos.cpp
"""
function gradThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho)
    
    m = size(X, 1)
    numTests = size(TestField[1], 2)
    idxTests = 1:numTests  # Julia is 1-based
    
    derivloglik = zeros(Float64, 2 * numTests)
    
    for jj in 1:m  # Julia is 1-based
        id = jj  # C++: id = jj+1L
        TestMat_i = TestField[jj]
        TestTimes_i = TestTimes[jj]
        
        t0 = startSamplingPeriod[jj] - 1  # C++: startSamplingPeriod[jj] - 1L
        maxt_i = endSamplingPeriod[jj] - t0
        
        for tt in 0:maxt_i-1  # C++: tt=0; tt<maxt_i; tt++
            # Find rows where TestTimes_i - t0 == tt+1
            rows = findall(x -> x - t0 == tt + 1, TestTimes_i)
            
            if length(rows) > 0
                TestMat_i_tt = TestMat_i[rows, :]
                
                for (ir_idx, ir) in enumerate(rows)
                    Tests_ir = TestMat_i_tt[ir_idx, :]
                    valid_tests = findall(x -> x == 0 || x == 1, Tests_ir)
                    
                    for ic in valid_tests
                        i = ic  # Julia is 1-based
                        
                        if X[id, tt + t0 + 1] == 3  # Exposed state (C++: X(id-1,tt+t0)==3L)
                            expThetaTilde = exp(logit(thetas[i]))
                            expRhoTilde = exp(logit(rhos[i]))
                            test_result = Tests_ir[i]
                            
                            # Derivatives wrt theta
                            derivloglik[i] += test_result * (1 - thetas[i]) + 
                                (1 - test_result) * (expThetaTilde / (1 + expThetaTilde + expRhoTilde) - thetas[i])
                            
                            # Derivatives wrt rho
                            derivloglik[i + numTests] += test_result * (1 - rhos[i]) + 
                                (1 - test_result) * (expRhoTilde / (1 + expThetaTilde + expRhoTilde) - rhos[i])
                            
                        elseif X[id, tt + t0 + 1] == 1  # Infectious state (C++: X(id-1,tt+t0)==1L)
                            test_result = Tests_ir[i]
                            
                            # Derivatives wrt theta
                            derivloglik[i] += test_result * (1 - thetas[i]) - 
                                (1 - test_result) * thetas[i]
                        end
                    end
                end
            end
        end
    end
    
    # Derivative of log prior with Jacobian
    derivLogPriorWithJac = zeros(Float64, 2 * numTests)
    for iTest in 1:numTests
        derivLogPriorWithJac[iTest] = hp_theta[1] - (hp_theta[1] + hp_theta[2]) * thetas[iTest]
        derivLogPriorWithJac[iTest + numTests] = hp_rho[1] - (hp_rho[1] + hp_rho[2]) * rhos[iTest]
    end
    
    grad = derivloglik + derivLogPriorWithJac
    return grad
end


function gradThetasRhos2(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho)
    
    m = size(X, 1)
    numTests = size(TestField[1], 2)
    idxTests = 1:numTests  # Julia is 1-based
    
    derivloglik = zeros(Float64, 2 * numTests)
    rows = Vector{Int64}(undef, size(X, 2))
    valid_tests = Vector{Int64}(undef, size(X, 2))
    Tests_ir = Vector{Float64}(undef, numTests)
            ### Looping wrong way Stilm column major!!
    for jj in 1:m  # Julia is 1-based
        id = jj  # C++: id = jj+1L
        TestMat_i = TestField[jj]
        TestTimes_i = TestTimes[jj]
        
        t0 = startSamplingPeriod[jj] - 1  # C++: startSamplingPeriod[jj] - 1L
        maxt_i = endSamplingPeriod[jj] - t0
        
        for tt in 0:maxt_i-1  # C++: tt=0; tt<maxt_i; tt++
            # Find rows where TestTimes_i - t0 == tt+1
            n_row = 0
            for it in 1:size(TestTimes_i, 1)
                if TestTimes_i[it] - t0 == tt + 1
                    n_row += 1
                    rows[n_row] = it
                end
            end
            #### This whole section could likely be a lot faster.
            #### Why make a vec of 
            if n_row > 0
                TestMat_i_tt = @views(TestMat_i[rows[1:n_row], :])
                
                #=
                for ir_idx in 1:n_row#(ir_idx, ir) in enumerate(rows[1:n_row])
                    Tests_ir .= TestMat_i_tt[ir_idx, :]
                    n_test = 0
                    for it in 1:numTests
                        if (Tests_ir[it] == 0.0) || (Tests_ir[it] == 1.0)
                            n_test += 1
                            valid_tests[n_test] = it
                        end
                    end
                =#
                for col in 1:numTests
                    n_test = 0
                    for it in 1:n_row
                        if (TestMat_i_tt[it,col] == 0.0) || (TestMat_i_tt[it,col] == 1.0)
                            n_test += 1
                            valid_tests[n_test] = it
                        end
                    end

                    #for ic in valid_tests[1:n_test]
                    for ic in 1:n_test#valid_tests[1:n_test]
                        i = valid_tests[ic]  # Julia is 1-based
                        
                        if X[id, tt + t0 + 1] == 3.0  # Exposed state (C++: X(id-1,tt+t0)==3L)
                            expThetaTilde = exp(logit(thetas[col]))
                            expRhoTilde = exp(logit(rhos[col]))
                            #test_result = Tests_ir[i]
                            test_result = TestMat_i_tt[i,col]
                            
                            # Derivatives wrt theta
                            derivloglik[i] += test_result * (1 - thetas[col]) + 
                                (1 - test_result) * (expThetaTilde / (1.0 + expThetaTilde + expRhoTilde) - thetas[col])
                            
                            # Derivatives wrt rho
                            derivloglik[col + numTests] += test_result * (1.0 - rhos[col]) + 
                                (1 - test_result) * (expRhoTilde / (1.0 + expThetaTilde + expRhoTilde) - rhos[col])
                            
                        elseif X[id, tt + t0 + 1] == 1.0  # Infectious state (C++: X(id-1,tt+t0)==1L)
                            test_result = TestMat_i_tt[i,col]
                            
                            # Derivatives wrt theta
                            derivloglik[i] += test_result * (1.0 - thetas[col]) - 
                                (1 - test_result) * thetas[col]
                        end
                    end
                end
            end
        end
    end
    
    # Derivative of log prior with Jacobian
    derivLogPriorWithJac = zeros(Float64, 2 * numTests)
    for iTest in 1:numTests
        derivLogPriorWithJac[iTest] = hp_theta[1] - (hp_theta[1] + hp_theta[2]) * thetas[iTest]
        derivLogPriorWithJac[iTest + numTests] = hp_rho[1] - (hp_rho[1] + hp_rho[2]) * rhos[iTest]
    end
    
    grad = derivloglik + derivLogPriorWithJac
    return grad
end

"""
    HMC_thetas_rhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho, epsilonsens, L)

Hamiltonian Monte Carlo update for thetas and rhos parameters.
Ported from C++ HMC_thetas_rhos.cpp
"""
function HMC_thetas_rhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, TestField, TestTimes, hp_theta, hp_rho, epsilonsens, L)
    
    numTests = length(thetas)
    out = zeros(Float64, 2 * numTests)
    
    # Stack into unique vector
    cur = zeros(Float64, 2 * numTests)
    for iTest in 1:numTests
        cur[iTest] = thetas[iTest]
        cur[iTest + numTests] = rhos[iTest]
    end
    
    # Multivariate normal proposal
    p = randn(2 * numTests)
    curp = copy(p)
    
    q_thetas = zeros(Float64, numTests)
    q_rhos = zeros(Float64, numTests)
    for iTest in 1:numTests
        q_thetas[iTest] = cur[iTest]
        q_rhos[iTest] = cur[iTest + numTests]
    end
    
    q = logit.(cur)
    
    # Half step for momentum
    p = p + epsilonsens * gradThetasRhos2(q_thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                                        TestField, TestTimes, hp_theta, hp_rho) / 2
    
    # Random number of leapfrog steps
    intL = ceil(rand() * L)
    
    for i in 1:intL-1
        q = q + epsilonsens * p
        
        for iTest in 1:numTests
            q_thetas[iTest] = logistic(q[iTest])
            q_rhos[iTest] = logistic(q[iTest + numTests])
        end
        
        p = p + epsilonsens * gradThetasRhos2(q_thetas, q_rhos, X, startSamplingPeriod, endSamplingPeriod, 
                                           TestField, TestTimes, hp_theta, hp_rho)
    end
    
    # Final leapfrog step
    q = q + epsilonsens * p
    for iTest in 1:numTests
        q_thetas[iTest] = logistic(q[iTest])
        q_rhos[iTest] = logistic(q[iTest + numTests])
    end
    
    # Half step for momentum
    p = p + epsilonsens * gradThetasRhos2(q_thetas, q_rhos, X, startSamplingPeriod, endSamplingPeriod, 
                                        TestField, TestTimes, hp_theta, hp_rho) / 2
    p = -p
    
    # Calculate Hamiltonian
    ProposedH = logPostThetasRhos(q_thetas, q_rhos, X, startSamplingPeriod, endSamplingPeriod,
                                 TestField, TestTimes, hp_theta, hp_rho) - 0.5 * dot(p, p)
    CurrentH = logPostThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod,
                                TestField, TestTimes, hp_theta, hp_rho) - 0.5 * dot(curp, curp)
    
    prob = exp(ProposedH - CurrentH)
    alpha = min(1.0, prob)
    u = rand()
    
    if u < alpha
        out = logistic.(q)
    else
        out = cur
    end
    
    return out
end

"""
    TestMatAsFieldProposal(TestFieldProposal, TestField, TestTimes, xi, xiCan, m)

Correct TestFieldProposal given the current TestField for Brock changepoint proposal.
Ported from C++ TestMatAsField.cpp
"""
function TestMatAsFieldProposal(TestFieldProposal, TestField, TestTimes, xi, xiCan, m)
    
    for i in 1:m  # Julia is 1-based, C++ was 0-based
        TestTimes_i = TestTimes[i]
        Tests_i = copy(TestFieldProposal[i])  # Make a copy to modify
        numCapt = size(Tests_i, 1)
        
        if xiCan < xi  # Proposing an earlier changepoint
            for irow in 1:numCapt
                t = TestTimes_i[irow]
                if (t >= xiCan) && (t < xi)
                    # Swap Brock1 and Brock2 (columns 1 and 2 in test matrix, 0-based in C++)
                    brock1 = Tests_i[irow, 1]  # Julia is 1-based
                    Tests_i[irow, 1] = Tests_i[irow, 2]
                    Tests_i[irow, 2] = brock1
                end
            end
        else  # Proposing a later changepoint
            for irow in 1:numCapt
                t = TestTimes_i[irow]
                if (t >= xi) && (t < xiCan)
                    # Swap Brock1 and Brock2 (columns 1 and 2 in test matrix, 0-based in C++)
                    brock1 = Tests_i[irow, 1]  # Julia is 1-based
                    Tests_i[irow, 1] = Tests_i[irow, 2]
                    Tests_i[irow, 2] = brock1
                end
            end
        end
        
        TestFieldProposal[i] = Tests_i
    end
end

"""
    logPostXi(xiMin, xiMax, xi, hp_xi, TestField_, TestTimes, thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)

Compute log posterior for xi (Brock changepoint) parameter.

The Brock test has two versions (Brock1 and Brock2) that were used before and after a changepoint.
This function calculates the likelihood of test results in the range [xiMin, xiMax) given:
- The proposed changepoint xi
- The test sensitivities (thetas), specificities (phis), and relative sensitivity for exposed (rhos)
- The hidden infection states X

The key insight: when xi changes, only test results in the range between the old and new xi 
need to be re-evaluated, because outside this range the test assignments don't change.

Ported from C++ logPostXi.cpp
"""
function logPostXi(xiMin, xiMax, xi, hp_xi, TestField_, TestTimes, thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
    
    # m = number of individuals
    m = size(X, 1)
    
    # We only care about the first 2 test columns (Brock1 and Brock2)
    numBrockTests = 2
    idxTests = 1:numBrockTests  # Julia 1-based indexing
    
    # Initialize log-likelihood
    logLik = 0.0

    # Loop over all individuals
    for jj in 1:m
        id = jj  # Individual ID
        TestMat_i = TestField_[jj]  # Test results for this individual (rows = test events, cols = test types)
        TestTimes_i = TestTimes[jj]  # Times when tests were performed for this individual
        
        # Sampling period for this individual
        t0 = startSamplingPeriod[jj]-1  # Start of monitoring
        maxt_i = endSamplingPeriod[jj] - t0  # Length of monitoring period
        
        # Loop over time points in this individual's monitoring period
        for tt in 1:maxt_i
            # CRITICAL: Only evaluate likelihood for times in the range [xiMin, xiMax)
            # This is the range where the changepoint proposal differs from the current value
            # Outside this range, test assignments are identical for both xi values
            if (tt + t0 >= xiMin) && (tt + t0 < xiMax)
                
                # Find all test events that occurred at this time point
                # TestTimes_i stores absolute times, so we need to match: TestTimes_i[row] - t0 == tt
                rows = findall(x -> x - t0 == tt, TestTimes_i)
            
                # If there were test events at this time
                if !isempty(rows)

                    # Extract test results for these events
                    TestMat_i_tt_allTests = TestMat_i[rows, :]  # All test types for these events
                    TestMat_i_tt = TestMat_i_tt_allTests[:, idxTests]  # Only Brock1 and Brock2 columns
                  # println(TestMat_i_tt)
                    
                    # Loop over each test event (row) at this time
                    for ir in 1:length(rows)
                        Tests_ir = TestMat_i_tt[ir, :]  # Test results for this event (Brock1, Brock2)
                        
                        # Find which Brock tests were actually performed (not missing)
                        # Test results are coded as: 0 = negative, 1 = positive, NaN = not performed
                        idx = [i for i in idxTests if ((Tests_ir[i] == 0) || (Tests_ir[i] == 1))]
                       # println("size(idx): $(size(idx))")
                        
                        # For each performed test, add its likelihood contribution
                        for i in idx
                            
                            # Get the hidden infection state at this time
                            state = X[id, tt + t0]  # 0=Susceptible, 1=Infectious, 3=Exposed
                            test_result = TestMat_i_tt[ir, i]  # 0=negative, 1=positive
                          #  println("state: $state, test_result: $test_result")
                            
                            # Calculate likelihood based on state and test result
                            # Using test sensitivity (theta) and specificity (1-phi)
                            
                            if state == 0  # Susceptible (truly negative)
                                # P(test=1|S) = 1-specificity = phi (false positive rate)
                                # P(test=0|S) = specificity = 1-phi
                                logLik += log(
                                    (1 - phis[i])^test_result *  # If positive: false positive
                                    phis[i]^(1 - test_result)     # If negative: true negative
                                )
                                
                            elseif state == 3  # Exposed (early infection, reduced sensitivity)
                                # P(test=1|E) = theta * rho (sensitivity reduced by factor rho)
                                # P(test=0|E) = 1 - theta * rho
                                logLik += log(
                                    (thetas[i] * rhos[i])^test_result * 
                                    (1 - thetas[i] * rhos[i])^(1 - test_result)
                                )
                                
                            elseif state == 1  # Infectious (truly positive)
                                # P(test=1|I) = theta (sensitivity)
                                # P(test=0|I) = 1 - theta (false negative)
                                logLik += log(
                                    thetas[i]^test_result * 
                                    (1 - thetas[i])^(1 - test_result)
                                )
                            end
                        end
                    end
                end
            end  # end if time in range [xiMin, xiMax)
        end
    end
    
    # Add log prior for xi (normal distribution)
    # hp_xi = [mean, sd]
    xiLogPrior = logpdf(Normal(hp_xi[1], hp_xi[2]), Float64(xi))
    
    # Log posterior = log likelihood + log prior
    logPost = logLik + xiLogPrior

    return logPost
end

"""
    RWMH_xi(can, cur, hp_xi, TestFieldProposal, TestField, TestTimes, thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)

Random walk Metropolis-Hastings update for xi (Brock changepoint) parameter.

This function implements the Metropolis-Hastings accept/reject step for updating the changepoint.
The key efficiency: we only need to evaluate the likelihood in the range [xiMin, xiMax) where
the two changepoint values differ, because outside this range the test assignments are identical.

Arguments:
- can: candidate (proposed) changepoint value
- cur: current changepoint value
- TestFieldProposal: test data with Brock1/Brock2 assigned according to candidate changepoint
- TestField: test data with Brock1/Brock2 assigned according to current changepoint

Ported from C++ RWMH_xi.cpp
"""
function RWMH_xi(can, cur, hp_xi, TestFieldProposal, TestField, TestTimes, thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
    
    # Determine the range [xiMin, xiMax) where the two changepoint values differ
    # This is the only range where we need to re-evaluate the likelihood
    if can < cur
        xiMin = can  # Candidate is earlier
        xiMax = cur
    else
        xiMin = cur  # Current is earlier
        xiMax = can
    end
    
    # Diagnostic: check how many tests fall in this window

    # Calculate log posterior difference
    # For candidate: use TestFieldProposal (Brock tests assigned according to 'can')
    # For current: use TestField (Brock tests assigned according to 'cur')
    logpostDiff = logPostXi(xiMin, xiMax, can, hp_xi, TestFieldProposal, TestTimes, 
                           thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod) - 
                  logPostXi(xiMin, xiMax, cur, hp_xi, TestField, TestTimes,  
                           thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
    
    # Metropolis-Hastings acceptance probability
    prob = exp(logpostDiff)  # Posterior ratio
    alpha = min(1.0, prob)   # Accept with probability min(1, ratio)
    u = rand()               # Uniform(0,1) random number
    
    # Accept or reject the proposal
    if u < alpha
        # ACCEPT: use candidate changepoint
        out = can
        # Update TestField to match the proposal (for next iteration)
        for i in 1:length(TestField)
            TestField[i] = copy(TestFieldProposal[i])
        end
    else
        # REJECT: keep current changepoint
        out = cur
        # Revert TestFieldProposal to match current (for next iteration)
        for i in 1:length(TestFieldProposal)
            TestFieldProposal[i] = copy(TestField[i])
        end
    end
    
    return out
end

