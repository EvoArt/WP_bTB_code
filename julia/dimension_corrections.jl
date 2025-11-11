"""
    dimension_corrections.jl

Critical corrections for dimension issues in the C++ to Julia conversions.
These address potential problems with row/column vectors and matrix operations.
"""

## Helper Functions with 1-based indexing

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
    q = logisticD(logPars[G+3])
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
function CheckSensSpec__CORRECTED(numTests, 
                        TestField, 
                        TestTimes,
                        X)
    
    m = size(X, 1)
    
    # Use 1-based indexing throughout
    # CRITICAL: Matrix dimensions - (4 x numTests) is correct
    out = zeros(Int, 4, numTests)
    
    for iTest in 1:numTests
        numInfecTested = 0
        numInfecPositives = 0
        numSuscepTested = 0
        numSuscepNegatives = 0
        
        for i in 1:m
            Tests_i = TestField[i]
            testTimes_i = TestTimes[i]  # Keep 1-based
            X_i = X[i, :]
            
            # CRITICAL: Check indexing logic
            if length(testTimes_i) > 0 && maximum(testTimes_i) <= length(X_i)
                status = X_i[testTimes_i]  # Use 1-based indexing
                tests_i = Tests_i[:, iTest]
                
                # exposed and infectious individuals
                which_ExpInfec = findall(x -> x == 3 || x == 1, status)
                tests_i_inf = tests_i[which_ExpInfec]
                status_inf = status[which_ExpInfec]
                
                newInfTes = count(x -> x == 0 || x == 1, tests_i_inf)
                newInfPos = count(x -> (status_inf[x] == 3 || status_inf[x] == 1) && (tests_i_inf[x] == 1), eachindex(status_inf))
                
                numInfecTested += newInfTes
                numInfecPositives += newInfPos
                
                # susceptible individuals
                which_suscep = findall(x -> x == 0, status)
                tests_i_suscep = tests_i[which_suscep]
                status_suscep = status[which_suscep]
                
                newSuscepTes = count(x -> x == 0 || x == 1, tests_i_suscep)
                newSuscepPos = count(x -> (status_suscep[x] == 0) && (tests_i_suscep[x] == 0), eachindex(status_suscep))
                
                numSuscepTested += newSuscepTes
                numSuscepNegatives += newSuscepPos
            end
        end
        
        # CRITICAL: Matrix indexing - (row, col) format
        out[1, iTest] = numInfecPositives
        out[2, iTest] = numInfecTested - numInfecPositives
        out[3, iTest] = numSuscepNegatives
        out[4, iTest] = numSuscepTested - numSuscepNegatives
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
    p = p .+ epsilon_vec .* grad_(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                                birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                                hp_lambda, hp_beta, hp_q, hp_tau,
                                hp_a2, hp_b2, hp_c1, k, K) ./ 2
    
    intL = ceil(rand() * L)
    
    for i in 1:intL-1
        # CRITICAL: Broadcasting for vector operations
        q = q .+ epsilon_vec .* p
        p = p .+ epsilon_vec .* grad_(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
                                     birthTimes, startSamplingPeriod, lastObsAliveTimes, capturesAfterMonit, ageMat, 
                                     hp_lambda, hp_beta, hp_q, hp_tau,
                                     hp_a2, hp_b2, hp_c1, k, K)
    end
    
    q = q .+ epsilon_vec .* p
    p = p .+ epsilon_vec .* grad_(q, G, X, totalNumInfec, SocGroup, totalmPerGroup,
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

    maxt_i = endTime - (t0-1)

    for tt = 0:maxt_i-1
        eta = etas[seasonVec[tt + t0]]

        if CaptHist[id, tt + t0] == 0
            corrector[tt + t0, :] .= (1 - eta, 1 - eta, 1 - eta, 1.0)

        else
            corrector[tt + t0, :] .= (eta, eta, eta, 0.0)

            rows = findall(x -> x == tt + 1, TestTimes_i .- t0)

            if !isempty(rows)

                TestMat_i_tt = TestMat_i[rows, :]

                productIfSuscep = 1.0
                productIfExposed = 1.0
                productIfInfectious = 1.0

                # === EXACT MATCH TO ORIGINAL NESTED LOOPS ===
                for ir in 1:length(rows)
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
