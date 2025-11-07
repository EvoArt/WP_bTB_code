"""
    dimension_corrections.jl

Critical corrections for dimension issues in the C++ to Julia conversions.
These address potential problems with row/column vectors and matrix operations.
"""

## CRITICAL ISSUE 1: Field Operations in TestMatAsField

"""
CORRECTED VERSION - Original had potential dimension issues
"""
function TestMatAsField_CORRECTED(TestMat, m)
    numTests = size(TestMat, 2) - 3
    cols = 4:(3+numTests)  # Julia is 1-based, so columns 4:end are test results
    
    F = Vector{Matrix{Int}}(undef, m)
    
    # Apply 0-based indexing to TestMat for operations
    TestMat_zb = @zero_based TestMat
    
    id_i = TestMat_zb[:, 2]  # id column (0-based indexing)
    
    for i in 0:m-1
        which_i = findall(x -> x == i, id_i)
        if length(which_i) > 0
            Tests_i = TestMat_zb[which_i, :]
            # Extract test result columns (0-based indexing: 3,4,5,...)
            F[i+1] = @zero_based Tests_i[:, 3:end]
        else
            # Handle case where individual has no test records
            F[i+1] = @zero_based zeros(Int, 0, numTests)
        end
    end
    
    return F
end

## CRITICAL ISSUE 2: Vector vs Matrix in gradient functions

"""
CORRECTED VERSION - Ensure proper vector dimensions
"""
function grad__CORRECTED(logPars::Vector{Float64}, G::Int, 
               X::Matrix{Int}, 
               totalNumInfec::Matrix{Int},
               SocGroup::Matrix{Int},
               totalmPerGroup::Matrix{Int},
               birthTimes::Vector{Int},
               startSamplingPeriod::Vector{Int},
               lastObsAliveTimes::Vector{Int}, 
               capturesAfterMonit::Matrix{Int},
               ageMat::Matrix{Int}, 
               hp_lambda::Vector{Float64},
               hp_beta::Vector{Float64},
               hp_q::Vector{Float64},
               hp_tau::Vector{Float64},
               hp_a2::Vector{Float64},
               hp_b2::Vector{Float64},
               hp_c1::Vector{Float64},
               k::Int, 
               K::Float64)
    
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
    
    # CRITICAL: Initialize as Vector{Float64} not Matrix
    gradient = zeros(Float64, length(logPars))  # This is correct - 1D vector
    
    # ... rest of function remains the same
    
    # Final gradient assembly - CRITICAL: Check indexing
    for g in 0:G-1
        gradient[g+1] = likeas[g+1]  # Vector indexing is correct
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
function CheckSensSpec__CORRECTED(numTests::Int, 
                        TestField::Vector{Matrix{Int}}, 
                        TestTimes::Vector{Vector{Int}},
                        X::Matrix{Int})
    
    m = size(X, 1)
    
    # Apply 0-based indexing
    X = @zero_based X
    
    # CRITICAL: Matrix dimensions - (4 x numTests) is correct
    out = zeros(Int, 4, numTests)
    
    for iTest in 0:numTests-1
        numInfecTested = 0
        numInfecPositives = 0
        numSuscepTested = 0
        numSuscepNegatives = 0
        
        for i in 0:m-1
            Tests_i = TestField[i+1]
            testTimes_i = TestTimes[i+1] .- 1  # Convert to 0-based
            X_i = X[i+1, :]
            
            # CRITICAL: Check indexing logic
            if length(testTimes_i) > 0 && maximum(testTimes_i) < length(X_i)
                status = X_i[testTimes_i .+ 1]  # Back to 1-based for Julia indexing
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
        end
        
        # CRITICAL: Matrix indexing - (row, col) format
        out[1, iTest+1] = numInfecPositives
        out[2, iTest+1] = numInfecTested - numInfecPositives
        out[3, iTest+1] = numSuscepNegatives
        out[4, iTest+1] = numSuscepTested - numSuscepNegatives
    end
    
    return out
end

## CRITICAL ISSUE 4: Vector operations in HMC/RWMH

"""
CORRECTED VERSION - Ensure vector operations are consistent
"""
function HMC_2_CORRECTED(curLogPars::Vector{Float64}, G::Int, 
               X::Matrix{Int}, 
               totalNumInfec::Matrix{Int},
               SocGroup::Matrix{Int},
               totalmPerGroup::Matrix{Int},
               birthTimes::Vector{Int}, 
               startSamplingPeriod::Vector{Int},
               lastObsAliveTimes::Vector{Int},
               capturesAfterMonit::Matrix{Int},
               ageMat::Matrix{Int},
               epsilon::Float64, 
               epsilonalphas::Float64, 
               epsilonbq::Float64, 
               epsilontau::Float64,
               epsilonc1::Float64, 
               nParsNotGibbs::Int, 
               L::Int, 
               hp_lambda::Vector{Float64},
               hp_beta::Vector{Float64},
               hp_q::Vector{Float64},
               hp_tau::Vector{Float64},
               hp_a2::Vector{Float64},
               hp_b2::Vector{Float64},
               hp_c1::Vector{Float64},
               k::Int, 
               K::Float64)
    
    # Apply 0-based indexing to input arrays
    X = @zero_based X
    totalNumInfec = @zero_based totalNumInfec
    SocGroup = @zero_based SocGroup
    totalmPerGroup = @zero_based totalmPerGroup
    birthTimes = @zero_based birthTimes
    startSamplingPeriod = @zero_based startSamplingPeriod
    lastObsAliveTimes = @zero_based lastObsAliveTimes
    capturesAfterMonit = @zero_based capturesAfterMonit
    ageMat = @zero_based ageMat
    
    # CRITICAL: Vector initialization
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
    
    for i in 0:intL-2
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

println("Dimension corrections prepared. Review these functions for critical fixes.")
println("Key issues addressed:")
println("1. Field operations with proper indexing")
println("2. Vector vs Matrix initialization")
println("3. Broadcasting in vector operations")
println("4. Matrix indexing consistency")
