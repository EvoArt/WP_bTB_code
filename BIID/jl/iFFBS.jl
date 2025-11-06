using Random
using LinearAlgebra
# optionally: using StatsBase  # uncomment if you prefer StatsBase.sample

# -----------------------
# Helper utilities
# -----------------------

# safe discrete sampler (returns index chosen according to probabilities)
function discrete_sample(states::Vector{Int}, probs::Vector{Float64})
    # assumes probs sum roughly to 1 and are non-negative
    # fallback implementation without StatsBase
    c = cumsum(probs)
    r = rand() * c[end]
    return states[searchsortedfirst(c, r)]
end

# Normalize a row of log-probabilities to probabilities (like normTransProbRest)
# Accepts a vector of log-probs (may be -Inf). Returns normalized probabilities.
function normTransProbRest(logrow::AbstractVector{<:Real})
    # subtract max for numerical stability
    mx = maximum(logrow)
    ex = exp.(logrow .- mx)
    s = sum(ex)
    return ex ./ s
end

# Equivalent of Rf_log1mexp for positive x:
# returns log(1 - exp(-x)) in a numerically stable way (x > 0).
function log1mexp_pos(x::Float64)
    # log(1 - exp(-x)) = log1p(-exp(-x))
    return log1p(-exp(-x))
end

# Placeholder Erlang CDF. You must replace with the project's function.
# In the original C++ code this was called as ErlangCDF(1, k, tau/(k))
# I provide a standard Gamma/Erlang CDF at value t = 1 with shape=k and scale = tau/k:
#    CDF(1) = cdf(Gamma(shape=k, scale=tau/k), 1.0)
# If your original ErlangCDF has different semantics, replace accordingly.
function ErlangCDF_at1(k::Int, scale::Float64)
    # using incomplete gamma: lower incomplete gamma / gamma(k)
    # CDF(x; k, θ) = γ(k, x/θ) / Γ(k)
    x = 1.0
    # For integer k we can compute sum_{j=0}^{k-1} exp(-x/θ) (x/θ)^j / j!
    λ = x / scale
    s = 0.0
    for j in 0:(k-1)
        s += exp(-λ) * λ^j / factorial(big(j))
    end
    return Float64(1.0 - s)  # note: we return 1 - tail sum
end

# -----------------------
# Stubs - replace these with your project functions
# -----------------------

# ObsProcess_ in C++ updates `corrector` based on observation process
# Signature follows the original - you'll need to provide an implementation
function ObsProcess!(corrector::AbstractMatrix{Float64},
                     t0::Int, endTime::Int, id::Int,
                     CaptHist::AbstractMatrix{Int},
                     TestMat_i::AbstractMatrix{Int},
                     TestTimes_i::AbstractVector{Int},
                     etas::AbstractVector{Float64},
                     thetas::AbstractVector{Float64},
                     rhos::AbstractVector{Float64},
                     phis::AbstractVector{Float64},
                     seasonVec::AbstractVector{Int})
    # TODO: implement application-specific observation-corrector update
    # This is a placeholder that leaves `corrector` unchanged.
    return
end

# iFFBScalcLogProbRest - compute logProbRest slice for individual `jj` at time tt
function iFFBScalcLogProbRest!(jj::Int, tt::Int,
                               logProbRest::Array{Float64,3},
                               X::AbstractMatrix{Int},
                               SocGroup::AbstractMatrix{Int},
                               LogProbDyingMat::AbstractMatrix{Float64},
                               LogProbSurvMat::AbstractMatrix{Float64},
                               logProbStoSgivenSorE::AbstractMatrix{Float64},
                               logProbStoEgivenSorE::AbstractMatrix{Float64},
                               logProbStoSgivenI::AbstractMatrix{Float64},
                               logProbStoEgivenI::AbstractMatrix{Float64},
                               logProbStoSgivenD::AbstractMatrix{Float64},
                               logProbStoEgivenD::AbstractMatrix{Float64},
                               logProbEtoE::Float64,
                               logProbEtoI::Float64)
    # TODO: implement the same calculations that your C++ version does for logProbRest(tt,*,jj)
    # For now we set to zeros to allow Julia function to run; replace with correct logic.
    logProbRest[tt, :, jj] .= 0.0
    return
end

# -----------------------
# The Julia translation of iFFBS_
# -----------------------

"""
    iFFBS!(
        alpha_js::Vector{Float64},
        b::Float64, q::Float64, tau::Float64, k::Int, K::Float64,
        probDyingMat::Matrix{Float64},
        LogProbDyingMat::Matrix{Float64},
        LogProbSurvMat::Matrix{Float64},
        logProbRest::Array{Float64,3},
        nuTimes::Vector{Int},
        nuEs::Vector{Float64},
        nuIs::Vector{Float64},
        thetas::Vector{Float64},
        rhos::Vector{Float64},
        phis::Vector{Float64},
        etas::Vector{Float64},
        id::Int,
        birthTime::Int,
        startTime::Int,
        endTime::Int,
        X::Matrix{Int},
        seasonVec::Vector{Int},
        TestMat_i::Matrix{Int},
        TestTimes_i::Vector{Int},
        CaptHist::Matrix{Int},
        corrector::Matrix{Float64},
        predProb::Matrix{Float64},
        filtProb::Matrix{Float64},
        logTransProbRest::Matrix{Float64},
        numInfecMat::Matrix{Int},
        SocGroup::Matrix{Int},
        mPerGroup::Matrix{Int},
        idVecAll::Vector{Int},
        logProbStoSgivenSorE::Matrix{Float64},
        logProbStoEgivenSorE::Matrix{Float64},
        logProbStoSgivenI::Matrix{Float64},
        logProbStoEgivenI::Matrix{Float64},
        logProbStoSgivenD::Matrix{Float64},
        logProbStoEgivenD::Matrix{Float64},
        logProbEtoE::Float64,
        logProbEtoI::Float64,
        whichRequireUpdate::Vector{Vector{Int}},
        sumLogCorrector::Ref{Float64}
    )
In-place update of X, predProb, filtProb, logProb* and other passed structures.
"""
function iFFBS!(
    alpha_js::Vector{Float64},
    b::Float64, q::Float64, tau::Float64, k::Int, K::Float64,
    probDyingMat::Matrix{Float64},
    LogProbDyingMat::Matrix{Float64},
    LogProbSurvMat::Matrix{Float64},
    logProbRest::Array{Float64,3},
    nuTimes::Vector{Int},
    nuEs::Vector{Float64},
    nuIs::Vector{Float64},
    thetas::Vector{Float64},
    rhos::Vector{Float64},
    phis::Vector{Float64},
    etas::Vector{Float64},
    id::Int,
    birthTime::Int,
    startTime::Int,
    endTime::Int,
    X::Matrix{Int},
    seasonVec::Vector{Int},
    TestMat_i::Matrix{Int},
    TestTimes_i::Vector{Int},
    CaptHist::Matrix{Int},
    corrector::Matrix{Float64},
    predProb::Matrix{Float64},
    filtProb::Matrix{Float64},
    logTransProbRest::Matrix{Float64},
    numInfecMat::Matrix{Int},
    SocGroup::Matrix{Int},
    mPerGroup::Matrix{Int},
    idVecAll::Vector{Int},
    logProbStoSgivenSorE::Matrix{Float64},
    logProbStoEgivenSorE::Matrix{Float64},
    logProbStoSgivenI::Matrix{Float64},
    logProbStoEgivenI::Matrix{Float64},
    logProbStoSgivenD::Matrix{Float64},
    logProbStoEgivenD::Matrix{Float64},
    logProbEtoE::Float64,
    logProbEtoI::Float64,
    whichRequireUpdate::Vector{Vector{Int}},    # linearized per (id-1)*(maxt-1) + tt
    sumLogCorrector::Ref{Float64}
)
    m, maxt = size(X)
    numStates = size(filtProb, 2)

    t0 = startTime - 1
    maxt_i = endTime - t0

    # Update corrector (must be provided by you)
    ObsProcess!(corrector, t0, endTime, id, CaptHist, TestMat_i, TestTimes_i,
                etas, thetas, rhos, phis, seasonVec)

    # idNext logic (1-based indexing)
    idNext = (id < m) ? id+1 : 1

    # Forward filtering preparations
    unnormFiltProb = zeros(Float64, numStates)
    transProbRest = zeros(Float64, numStates)

    # initialize nuE_i, nuI_i
    nuE_i = 0.0
    nuI_i = 0.0
    if birthTime < startTime
        # find index in nuTimes equal to startTime (first match)
        idx = findfirst(==(startTime), nuTimes)
        if idx !== nothing
            nuE_i = nuEs[idx]
            nuI_i = nuIs[idx]
        end
    end

    # initialize predProb at t0 (note Julia index t0+1 for matrices)
    # predProb and filtProb are expected to be indexed 1..maxt, states in columns
    predProb[t0+1, 1] = 1.0 - nuE_i - nuI_i
    predProb[t0+1, 2] = nuE_i
    predProb[t0+1, 3] = nuI_i
    predProb[t0+1, 4] = 0.0

    if t0 < maxt - 1
        logTransProbRest_row = view(logTransProbRest, t0+1, :)
        transProbRest .= normTransProbRest(collect(logTransProbRest_row))
        for s in 1:numStates
            unnormFiltProb[s] = corrector[t0+1, s] * predProb[t0+1, s] * transProbRest[s]
        end
    else
        for s in 1:numStates
            unnormFiltProb[s] = corrector[t0+1, s] * predProb[t0+1, s]
        end
    end
    filtProb[t0+1, :] .= unnormFiltProb ./ sum(unnormFiltProb)

    # t = 2...T-1 forward loop (tt counts from 1..maxt_i-2 in 0-based C++ -> here adjust)
    if maxt_i > 2
        for tt in 1:(maxt_i-2)
            g = SocGroup[id, tt + t0]   # SocGroup is 1-based
            prDeath = probDyingMat[id, tt + t0 + 1]

            p00 = (1-prDeath)*exp(logProbStoSgivenSorE[g, tt + t0])
            p01 = (1-prDeath)*exp(logProbStoEgivenSorE[g, tt + t0])
            p11 = (1-prDeath)*exp(logProbEtoE)
            p12 = (1-prDeath)*exp(logProbEtoI)
            p22 = (1-prDeath)

            predProb[tt + t0 + 1, 1] = p00 * filtProb[tt + t0, 1]
            predProb[tt + t0 + 1, 2] = p01 * filtProb[tt + t0, 1] + p11 * filtProb[tt + t0, 2]
            predProb[tt + t0 + 1, 3] = p12 * filtProb[tt + t0, 2] + p22 * filtProb[tt + t0, 3]
            predProb[tt + t0 + 1, 4] = prDeath*(filtProb[tt + t0, 1] + filtProb[tt + t0, 2] + filtProb[tt + t0, 3]) + 1.0 * filtProb[tt + t0, 4]

            logTransProbRest_row = view(logTransProbRest, tt + t0 + 1, :)
            transProbRest .= normTransProbRest(collect(logTransProbRest_row))

            for s in 1:numStates
                unnormFiltProb[s] = corrector[tt + t0 + 1, s] * predProb[tt + t0 + 1, s] * transProbRest[s]
            end
            filtProb[tt + t0 + 1, :] .= unnormFiltProb ./ sum(unnormFiltProb)
        end
    end

    # t = T (last time point)
    if maxt_i >= 1
        tt = maxt_i - 1
        g = SocGroup[id, tt + t0]
        prDeath = probDyingMat[id, tt + t0 + 1]

        p00 = (1-prDeath)*exp(logProbStoSgivenSorE[g, tt + t0])
        p01 = (1-prDeath)*exp(logProbStoEgivenSorE[g, tt + t0])
        p11 = (1-prDeath)*exp(logProbEtoE)
        p12 = (1-prDeath)*exp(logProbEtoI)
        p22 = (1-prDeath)

        predProb[tt + t0 + 1, 1] = p00 * filtProb[tt + t0, 1]
        predProb[tt + t0 + 1, 2] = p01 * filtProb[tt + t0, 1] + p11 * filtProb[tt + t0, 2]
        predProb[tt + t0 + 1, 3] = p12 * filtProb[tt + t0, 2] + p22 * filtProb[tt + t0, 3]
        predProb[tt + t0 + 1, 4] = prDeath*(filtProb[tt + t0, 1] + filtProb[tt + t0, 2] + filtProb[tt + t0, 3]) + 1.0 * filtProb[tt + t0, 4]

        if tt + t0 < maxt - 1
            logTransProbRest_row = view(logTransProbRest, tt + t0 + 1, :)
            transProbRest .= normTransProbRest(collect(logTransProbRest_row))
            for s in 1:numStates
                unnormFiltProb[s] = corrector[tt + t0 + 1, s] * predProb[tt + t0 + 1, s] * transProbRest[s]
            end
        else
            for s in 1:numStates
                unnormFiltProb[s] = corrector[tt + t0 + 1, s] * predProb[tt + t0 + 1, s]
            end
        end
        filtProb[tt + t0 + 1, :] .= unnormFiltProb ./ sum(unnormFiltProb)
    end

    # Backward sampling
    states = [1, 4, 2, 10]   # mapping: C++ {0,3,1,9} -> using +1 for Julia 1-based indexing states (but original coded those numbers as labels)
    # IMPORTANT: the values used to assign to X are original labels (0,3,1,9). In Julia we keep those labels as they were in C++.
    # To avoid confusion, create states_labels matching C++ numeric labels:
    states_labels = [0, 3, 1, 9]

    # take probs as last row of filtProb at endTime
    probs = copy(vec(filtProb[endTime, :]))

    # sample newStatus (this returns the label)
    # We supply discrete_sample with labels
    newStatus = discrete_sample(states_labels, probs)

    X[id, endTime] = newStatus

    # backward loop: tt from maxt_i-1 down to 0 (C++ used tt --> 0)
    if maxt_i > 1
        for tt in (maxt_i-1):-1:0
            # note: indices in Julia: use tt + t0 + 1 to index into matrices
            g = SocGroup[id, tt + t0 + 1]
            mgt = mPerGroup[g, tt + t0 + 1]
            a = alpha_js[g]

            inf_mgt = numInfecMat[g, tt + t0 + 1] / ( ((mgt + 1.0)/K)^q )

            prDeath = probDyingMat[id, tt + t0 + 2]   # probDyingMat(id-1, tt+1+t0) in C++ -> here +2

            p00 = (1-prDeath)*exp(-a - b*inf_mgt)
            p01 = (1-prDeath)*(1 - exp(-a - b*inf_mgt))
            p11 = (1-prDeath)*(1.0 - ErlangCDF_at1(k, tau/(k)))   # replace with your ErlangCDF if different
            p12 = (1-prDeath)*ErlangCDF_at1(k, tau/(k))
            p22 = (1-prDeath)

            p09 = prDeath
            p19 = prDeath
            p29 = prDeath

            probSuscep_t = 0.0
            probE_t = 0.0
            probI_t = 0.0
            probDead_t = 0.0

            next_state_label = X[id, tt + t0 + 2]   # X(id-1, tt+1+t0) in C++ -> here

            if next_state_label == 0
                probSuscep_t = (p00 * filtProb[tt + t0 + 1, 1]) / predProb[tt + t0 + 2, 1]
            elseif next_state_label == 3
                probSuscep_t = (p01 * filtProb[tt + t0 + 1, 1]) / predProb[tt + t0 + 2, 2]
                probE_t      = (p11 * filtProb[tt + t0 + 1, 2]) / predProb[tt + t0 + 2, 2]
            elseif next_state_label == 1
                probE_t      = (p12 * filtProb[tt + t0 + 1, 2]) / predProb[tt + t0 + 2, 3]
                probI_t      = (p22 * filtProb[tt + t0 + 1, 3]) / predProb[tt + t0 + 2, 3]
            elseif next_state_label == 9
                probSuscep_t = (p09 * filtProb[tt + t0 + 1, 1]) / predProb[tt + t0 + 2, 4]
                probE_t      = (p19 * filtProb[tt + t0 + 1, 2]) / predProb[tt + t0 + 2, 4]
                probI_t      = (p29 * filtProb[tt + t0 + 1, 3]) / predProb[tt + t0 + 2, 4]
                probDead_t   = filtProb[tt + t0 + 1, 4] / predProb[tt + t0 + 2, 4]
            end

            probs_vec = [probSuscep_t, probE_t, probI_t, probDead_t]

            # boundary: if tt==0 and birthTime >= startTime => deterministic susceptible
            if (tt == 0) && (birthTime >= startTime)
                probs_vec = [1.0, 0.0, 0.0, 0.0]
            end

            newStatus = discrete_sample(states_labels, probs_vec)
            X[id, tt + t0 + 1] = newStatus
        end
    end

    # compute sumLogCorrector contribution for this individual over its interval
    for tt in 0:(maxt_i-1)
        lab = X[id, tt + t0 + 1]
        if lab == 0
            sumLogCorrector[] += log(corrector[tt + t0 + 1, 1])
        elseif lab == 3
            sumLogCorrector[] += log(corrector[tt + t0 + 1, 2])
        elseif lab == 1
            sumLogCorrector[] += log(corrector[tt + t0 + 1, 3])
        elseif lab == 9
            sumLogCorrector[] += log(corrector[tt + t0 + 1, 4])
        end
    end

    # Update numInfecMat, mPerGroup, and logProb* for next individual idNext
    for tt in 1:(maxt-1)
        g = SocGroup[id, tt]
        g_idNext = SocGroup[idNext, tt]

        if (g == g_idNext) && (g != 0)
            infecToAdd = (X[id, tt] == 1 ? 1 : 0) - (X[idNext, tt] == 1 ? 1 : 0)
            if (X[id, tt] == 1) || (X[idNext, tt] == 1)
                numInfecMat[g, tt] += infecToAdd
            end

            mToAdd = ((X[id, tt] in (0,1,3)) ? 1 : 0) - ((X[idNext, tt] in (0,1,3)) ? 1 : 0)
            if ((X[id, tt] in (0,1,3)) || (X[idNext, tt] in (0,1,3)))
                mPerGroup[g, tt] += mToAdd
            end

            if id < m
                a = alpha_js[g]
                mgt = mPerGroup[g, tt]
                inf_mgt = numInfecMat[g, tt] / ((mgt / K)^q)
                logProbStoSgivenSorE[g, tt] = -a - b*inf_mgt
                logProbStoEgivenSorE[g, tt] = log1mexp_pos(a + b*inf_mgt)

                inf_mgt = (numInfecMat[g, tt] + 1) / (( (mgt + 1.0) / K )^q)
                logProbStoSgivenI[g, tt] = -a - b*inf_mgt
                logProbStoEgivenI[g, tt] = log1mexp_pos(a + b*inf_mgt)

                inf_mgt = numInfecMat[g, tt] / ((mgt / K)^q)
                logProbStoSgivenD[g, tt] = -a - b*inf_mgt
                logProbStoEgivenD[g, tt] = log1mexp_pos(a + b*inf_mgt)
            end
        else
            if g != 0
                if X[id, tt] == 1
                    numInfecMat[g, tt] += 1
                end
                if X[id, tt] in (0,1,3)
                    mPerGroup[g, tt] += 1
                end
            end

            if (id < m) && (g != 0)
                a = alpha_js[g]
                mgt = mPerGroup[g, tt]
                inf_mgt = numInfecMat[g, tt] / ((mgt / K)^q)
                logProbStoSgivenSorE[g, tt] = -a - b*inf_mgt
                logProbStoEgivenSorE[g, tt] = log1mexp_pos(a + b*inf_mgt)
                logProbStoSgivenI[g, tt] = -a - b*inf_mgt
                logProbStoEgivenI[g, tt] = log1mexp_pos(a + b*inf_mgt)
                logProbStoSgivenD[g, tt] = -a - b*inf_mgt
                logProbStoEgivenD[g, tt] = log1mexp_pos(a + b*inf_mgt)
            end

            if g_idNext != 0
                if X[idNext, tt] == 1
                    numInfecMat[g_idNext, tt] -= 1
                end
                if X[idNext, tt] in (0,1,3)
                    mPerGroup[g_idNext, tt] -= 1
                end
            end

            if (id < m) && (g_idNext != 0)
                a = alpha_js[g_idNext]
                mgt = mPerGroup[g_idNext, tt]
                inf_mgt = numInfecMat[g_idNext, tt] / ( ((mgt + 1.0)/K)^q )
                logProbStoSgivenSorE[g_idNext, tt] = -a - b*inf_mgt
                logProbStoEgivenSorE[g_idNext, tt] = log1mexp_pos(a + b*inf_mgt)
                inf_mgt = (numInfecMat[g_idNext, tt] + 1) / ( ((mgt + 1.0)/K)^q )
                logProbStoSgivenI[g_idNext, tt] = -a - b*inf_mgt
                logProbStoEgivenI[g_idNext, tt] = log1mexp_pos(a + b*inf_mgt)
                inf_mgt = numInfecMat[g_idNext, tt] / ((mgt / K)^q)
                logProbStoSgivenD[g_idNext, tt] = -a - b*inf_mgt
                logProbStoEgivenD[g_idNext, tt] = log1mexp_pos(a + b*inf_mgt)
            end
        end
    end

    # Update logTransProbRest and logProbRest for next id (if id < m)
    if id < m
        c = (id-1)*(maxt-1)
        for tt in 1:(maxt-1)
            # update logProbRest entry for id-1 (id in Julia is 1-based)
            iFFBScalcLogProbRest!(id, tt, logProbRest, X, SocGroup,
                                  LogProbDyingMat, LogProbSurvMat,
                                  logProbStoSgivenSorE, logProbStoEgivenSorE,
                                  logProbStoSgivenI, logProbStoEgivenI,
                                  logProbStoSgivenD, logProbStoEgivenD,
                                  logProbEtoE, logProbEtoI)

            # add logProbRest(tt, :, id-1) and remove logProbRest(tt,:,idNext)
            for s in 1:numStates
                logTransProbRest[tt, s] += (logProbRest[tt, s, id] - logProbRest[tt, s, idNext])
                logProbRest[tt, s, idNext] = 0.0
            end

            # update flagged individuals
            for jj in whichRequireUpdate[c + tt]
                if X[jj, tt] == 0
                    # subtract previous contribution
                    for s in 1:numStates
                        logTransProbRest[tt, s] -= logProbRest[tt, s, jj]
                    end

                    # recompute logProbRest for jj at tt with direct logic (mirroring C++)
                    g1 = SocGroup[jj, tt] - 1  # careful if you used 0-based groups
                    # in original code they used g_1 = SocGroup(jj,tt)-1
                    # Here we assume SocGroup is already suitable (adjust if necessary)
                    # For now we mimic the C++ logic:
                    if X[jj, tt+1] == 0
                        logProbRest[tt, 1, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenSorE[g1+1, tt]
                        logProbRest[tt, 2, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenSorE[g1+1, tt]
                        logProbRest[tt, 3, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenI[g1+1, tt]
                        logProbRest[tt, 4, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenD[g1+1, tt]
                    elseif X[jj, tt+1] == 3
                        logProbRest[tt, 1, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenSorE[g1+1, tt]
                        logProbRest[tt, 2, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenSorE[g1+1, tt]
                        logProbRest[tt, 3, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenI[g1+1, tt]
                        logProbRest[tt, 4, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenD[g1+1, tt]
                    end

                    for s in 1:numStates
                        logTransProbRest[tt, s] += logProbRest[tt, s, jj]
                    end
                end
            end
        end
    end

    return nothing
end
