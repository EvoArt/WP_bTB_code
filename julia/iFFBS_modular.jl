"""
Modular version of iFFBS_ algorithm broken into separate functions.
Each function handles a specific part of the forward-filtering backward-sampling algorithm.
"""

function iFFBS_initializeForwardFiltering(
    birthTime,
    startTime,
    nuTimes,
    nuEs,
    nuIs,
    predProb,
    t0,
    numStates)
    
    nuE_i = 0.0
    nuI_i = 0.0
    
    if birthTime < startTime
        nuIdx = findfirst(nuTimes .== startTime)
        if nuIdx !== nothing
            nuE_i = nuEs[nuIdx]
            nuI_i = nuIs[nuIdx]
        end
    end
    
    predProb[t0, 1] = 1.0 - nuE_i - nuI_i  # Susceptible
    predProb[t0, 2] = nuE_i  # Exposed
    predProb[t0, 3] = nuI_i  # Infectious
    predProb[t0, 4] = 0.0    # Dead
    
    return (nuE_i=nuE_i, nuI_i=nuI_i, predProb=predProb)
end

function iFFBS_forwardFilteringFirstStep(
    corrector,
    predProb,
    filtProb,
    logTransProbRest,
    t0,
    maxt,
    numStates)
    
    unnormFiltProb = zeros(numStates)
    transProbRest = zeros(numStates)
    log_probs_minus_B = zeros(numStates)
    
    if t0 < maxt - 1
        transProbRest .= logTransProbRest[:, t0]
        normTransProbRest!(transProbRest, log_probs_minus_B)
        for s in 1:numStates
            unnormFiltProb[s] = corrector[t0, s] * predProb[t0, s] * transProbRest[s]
        end
    else
        for s in 1:numStates
            unnormFiltProb[s] = corrector[t0, s] * predProb[t0, s]
        end
    end
    
    # Handle numerical issues
    if sum(unnormFiltProb) == 0.0
        unnormFiltProb .= fill(1.0 / numStates, numStates)
    end
    
    filtProb[t0, :] .= unnormFiltProb ./ sum(unnormFiltProb)
    
    # Handle NaN probabilities
    if any(isnan, filtProb[t0, :])
        filtProb[t0, :] .= fill(1.0 / numStates, numStates)
    end
    
    return (filtProb=filtProb, unnormFiltProb=unnormFiltProb)
end

function iFFBS_forwardFilteringLoop(
    predProb,
    filtProb,
    corrector,
    logTransProbRest,
    probDyingMat,
    logProbStoSgivenSorE,
    logProbStoEgivenSorE,
    logProbEtoE,
    logProbEtoI,
    SocGroup,
    id,
    t0,
    maxt_i,
    numStates)
    
    unnormFiltProb = zeros(numStates)
    transProbRest = zeros(numStates)
    log_probs_minus_B = zeros(numStates)
    
    # Pre-compute constant exp() values
    expLogProbEtoE = exp(logProbEtoE)
    expLogProbEtoI = exp(logProbEtoI)
    
    if maxt_i > 1
        for tt in 1:maxt_i-1
            g = SocGroup[id, tt-1+t0]
            prDeath = probDyingMat[id, tt+t0]
            prSurv = 1.0 - prDeath
            
            p00 = prSurv * exp(logProbStoSgivenSorE[g, tt-1+t0])
            p01 = prSurv * exp(logProbStoEgivenSorE[g, tt-1+t0])
            p11 = prSurv * expLogProbEtoE
            p12 = prSurv * expLogProbEtoI
            p22 = prSurv
            
            predProb[tt+t0, 1] = p00 * filtProb[tt-1+t0, 1]
            predProb[tt+t0, 2] = p01 * filtProb[tt-1+t0, 1] + p11 * filtProb[tt-1+t0, 2]
            predProb[tt+t0, 3] = p12 * filtProb[tt-1+t0, 2] + p22 * filtProb[tt-1+t0, 3]
            predProb[tt+t0, 4] = prDeath * filtProb[tt-1+t0, 1] + 
                                 prDeath * filtProb[tt-1+t0, 2] + 
                                 prDeath * filtProb[tt-1+t0, 3] +
                                 filtProb[tt-1+t0, 4]
            
            transProbRest .= logTransProbRest[:, tt+t0]
            normTransProbRest!(transProbRest, log_probs_minus_B)
            
            for s in 1:numStates
                unnormFiltProb[s] = corrector[tt+t0, s] * predProb[tt+t0, s] * transProbRest[s]
            end
            
            filtProb[tt+t0, :] .= unnormFiltProb ./ sum(unnormFiltProb)
        end
    end
    
    return (predProb=predProb, filtProb=filtProb)
end

function iFFBS_forwardFilteringFinalStep(
    predProb,
    filtProb,
    corrector,
    logTransProbRest,
    probDyingMat,
    logProbStoSgivenSorE,
    logProbStoEgivenSorE,
    logProbEtoE,
    logProbEtoI,
    SocGroup,
    id,
    t0,
    maxt_i,
    maxt,
    numStates)
    
    unnormFiltProb = zeros(numStates)
    transProbRest = zeros(numStates)
    log_probs_minus_B = zeros(numStates)
    
    # Pre-compute constant exp() values
    expLogProbEtoE = exp(logProbEtoE)
    expLogProbEtoI = exp(logProbEtoI)
    
    if maxt_i >= 1
        tt = maxt_i
        g = SocGroup[id, tt-1+t0]
        prDeath = probDyingMat[id, tt+t0]
        prSurv = 1.0 - prDeath
        
        p00 = prSurv * exp(logProbStoSgivenSorE[g, tt-1+t0])
        p01 = prSurv * exp(logProbStoEgivenSorE[g, tt-1+t0])
        p11 = prSurv * expLogProbEtoE
        p12 = prSurv * expLogProbEtoI
        p22 = prSurv
        
        predProb[tt+t0, 1] = p00 * filtProb[tt-1+t0, 1]
        predProb[tt+t0, 2] = p01 * filtProb[tt-1+t0, 1] + p11 * filtProb[tt-1+t0, 2]
        predProb[tt+t0, 3] = p12 * filtProb[tt-1+t0, 2] + p22 * filtProb[tt-1+t0, 3]
        predProb[tt+t0, 4] = prDeath * filtProb[tt-1+t0, 1] + 
                             prDeath * filtProb[tt-1+t0, 2] + 
                             prDeath * filtProb[tt-1+t0, 3] +
                             filtProb[tt-1+t0, 4]
        
        if tt + t0 < maxt - 1
            transProbRest .= logTransProbRest[:, tt+t0]
            normTransProbRest!(transProbRest, log_probs_minus_B)
            for s in 1:numStates
                unnormFiltProb[s] = corrector[tt+t0, s] * predProb[tt+t0, s] * transProbRest[s]
            end
        else
            for s in 1:numStates
                unnormFiltProb[s] = corrector[tt+t0, s] * predProb[tt+t0, s]
            end
        end
        
        # Handle numerical issues
        sum_unnorm = sum(unnormFiltProb)
        if sum_unnorm == 0.0 || !isfinite(sum_unnorm)
            filtProb[tt+t0, :] .= 1.0 / numStates
        else
            filtProb[tt+t0, :] .= unnormFiltProb ./ sum_unnorm
        end
    end
    
    return (predProb=predProb, filtProb=filtProb)
end

function iFFBS_backwardSampling(
    X,
    filtProb,
    predProb,
    probDyingMat,
    alpha_js,
    b,
    q,
    tau,
    k,
    K,
    SocGroup,
    mPerGroup,
    numInfecMat,
    id,
    birthTime,
    startTime,
    endTime,
    t0,
    maxt_i)
    
    
    states = [0, 3, 1, 9]
    probs = filtProb[endTime, :]
    
    # Handle numerical issues
    if any(isnan, probs) || any(isinf, probs)
        probs = fill(1.0 / length(probs), length(probs))
    else
        probs = probs ./ sum(probs)
    end
    
    newStatus = sample(states, Weights(probs))
    X[id, endTime] = newStatus
    
    # Pre-compute Erlang CDF
    erlang_dist = Erlang(k, tau/k)
    erlang_cdf_1 = cdf(erlang_dist, 1)
    
    if maxt_i > 1
        for tt in maxt_i-1:-1:0
            g = SocGroup[id, tt+t0]
            mgt = mPerGroup[g, tt+t0]
            
            a = alpha_js[g]
            inf_mgt = numInfecMat[g, tt+t0] / ((Float64(mgt + 1.0) / K)^q)
            prDeath = probDyingMat[id, tt+1+t0]
            prSurv = 1.0 - prDeath
            
            exp_neg_force = exp(-a - b * inf_mgt)
            p00 = prSurv * exp_neg_force
            p01 = prSurv * (1.0 - exp_neg_force)
            p11 = prSurv * (1.0 - erlang_cdf_1)
            p12 = prSurv * erlang_cdf_1
            p22 = prSurv
            
            p09 = prDeath
            p19 = prDeath
            p29 = prDeath
            
            probSuscep_t = 0.0
            probE_t = 0.0
            probI_t = 0.0
            probDead_t = 0.0
            
            if X[id, tt+1+t0] == 0.0
                probSuscep_t = (p00 * filtProb[tt+t0, 1]) / (predProb[tt+1+t0, 1])
                probE_t = 0.0
                probI_t = 0.0
                probDead_t = 0.0
            elseif X[id, tt+1+t0] == 3
                probSuscep_t = (p01 * filtProb[tt+t0, 1]) / (predProb[tt+1+t0, 2])
                probE_t = (p11 * filtProb[tt+t0, 2]) / (predProb[tt+1+t0, 2])
                probI_t = 0.0
                probDead_t = 0.0
            elseif X[id, tt+1+t0] == 1
                probSuscep_t = 0.0
                probE_t = (p12 * filtProb[tt+t0, 2]) / (predProb[tt+1+t0, 3])
                probI_t = (p22 * filtProb[tt+t0, 3]) / (predProb[tt+1+t0, 3])
                probDead_t = 0.0
            elseif X[id, tt+1+t0] == 9
                probSuscep_t = (p09 * filtProb[tt+t0, 1]) / (predProb[tt+1+t0, 4])
                probE_t = (p19 * filtProb[tt+t0, 2]) / (predProb[tt+1+t0, 4])
                probI_t = (p29 * filtProb[tt+t0, 3]) / (predProb[tt+1+t0, 3])
                probDead_t = filtProb[tt+t0, 4] / predProb[tt+1+t0, 4]
            end
            
            probs = [probSuscep_t, probE_t, probI_t, probDead_t]
            
            # Handle numerical issues
            if any(isnan, probs) || any(isinf, probs)
                probs = [0.25, 0.25, 0.25, 0.25]
            end
            
            if (tt == 1) && (birthTime >= startTime)
                probs = [1.0, 0.0, 0.0, 0.0]
            end
            
            newStatus = sample(states, Weights(probs))
            X[id, tt+t0] = newStatus
        end
    end
    
    return (X=X,)
end

function iFFBS_calculateLogCorrector(
    X,
    corrector,
    id,
    t0,
    maxt_i)
    
    sumLogCorrector = 0.0
    
    for tt in 1:maxt_i
        if X[id, tt+t0] == 0.0
            sumLogCorrector += log(corrector[tt+t0, 1])
        elseif X[id, tt+t0] == 3
            sumLogCorrector += log(corrector[tt+t0, 2])
        elseif X[id, tt+t0] == 1
            sumLogCorrector += log(corrector[tt+t0, 3])
        elseif X[id, tt+t0] == 9
            sumLogCorrector += log(corrector[tt+t0, 4])
        end
    end
    
    return sumLogCorrector
end

function iFFBS_updateGroupStatistics(
    X,
    numInfecMat,
    mPerGroup,
    logProbStoSgivenSorE,
    logProbStoEgivenSorE,
    logProbStoSgivenI,
    logProbStoEgivenI,
    logProbStoSgivenD,
    logProbStoEgivenD,
    alpha_js,
    b,
    q,
    K,
    SocGroup,
    id,
    idNext,
    m,
    maxt)
    
    for tt in 1:maxt-1
        g = SocGroup[id, tt]
        g_idNext = SocGroup[idNext, tt]
        
        if (g == g_idNext) && (g != 0)
            infecToAdd = 0
            if X[id, tt] == 1
                infecToAdd += 1
            end
            if X[idNext, tt] == 1
                infecToAdd -= 1
            end
            
            if (X[id, tt] == 1) || (X[idNext, tt] == 1)
                numInfecMat[g_idNext, tt] += infecToAdd
            end
            
            mToAdd = 0
            if (X[id, tt] == 0.0) || (X[id, tt] == 1) || (X[id, tt] == 3)
                mToAdd += 1
            end
            if (X[idNext, tt] == 0.0) || (X[idNext, tt] == 1) || (X[idNext, tt] == 3)
                mToAdd -= 1
            end
            
            if ((X[id, tt] == 0.0) || (X[id, tt] == 1) || (X[id, tt] == 3)) || 
               ((X[idNext, tt] == 0.0) || (X[idNext, tt] == 1) || (X[idNext, tt] == 3))
                mPerGroup[g_idNext, tt] += mToAdd
            end
            
            if id < m
                a = alpha_js[g_idNext]
                mgt = mPerGroup[g_idNext, tt]
                
                inf_mgt = numInfecMat[g_idNext, tt] / ((Float64(mgt + 1.0) / K)^q)
                logProbStoSgivenSorE[g_idNext, tt] = -a - b * inf_mgt
                logProbStoEgivenSorE[g_idNext, tt] = safe_log1mexp(a + b * inf_mgt)
                
                inf_mgt = (numInfecMat[g_idNext, tt] + 1) / ((Float64(mgt + 1.0) / K)^q)
                logProbStoSgivenI[g_idNext, tt] = -a - b * inf_mgt
                logProbStoEgivenI[g_idNext, tt] = safe_log1mexp(a + b * inf_mgt)
                
                inf_mgt = numInfecMat[g_idNext, tt] / ((Float64(mgt) / K)^q)
                logProbStoSgivenD[g_idNext, tt] = -a - b * inf_mgt
                logProbStoEgivenD[g_idNext, tt] = safe_log1mexp(a + b * inf_mgt)
            end
            
        else
            if g != 0
                if X[id, tt] == 1
                    numInfecMat[g, tt] += 1
                end
                if (X[id, tt] == 0.0) || (X[id, tt] == 1) || (X[id, tt] == 3)
                    mPerGroup[g, tt] += 1
                end
            end
            
            if (id < m) && (g != 0)
                a = alpha_js[g]
                mgt = mPerGroup[g, tt]
                
                inf_mgt = numInfecMat[g, tt] / ((Float64(mgt) / K)^q)
                logProbStoSgivenSorE[g, tt] = -a - b * inf_mgt
                logProbStoEgivenSorE[g, tt] = safe_log1mexp(a + b * inf_mgt)
                logProbStoSgivenI[g, tt] = -a - b * inf_mgt
                logProbStoEgivenI[g, tt] = safe_log1mexp(a + b * inf_mgt)
                logProbStoSgivenD[g, tt] = -a - b * inf_mgt
                logProbStoEgivenD[g, tt] = safe_log1mexp(a + b * inf_mgt)
            end
            
            if g_idNext != 0
                if X[idNext, tt] == 1
                    numInfecMat[g_idNext, tt] -= 1
                end
                if (X[idNext, tt] == 0.0) || (X[idNext, tt] == 1) || (X[idNext, tt] == 3)
                    mPerGroup[g_idNext, tt] -= 1
                end
            end
            
            if (id < m) && (g_idNext != 0)
                a = alpha_js[g_idNext]
                mgt = mPerGroup[g_idNext, tt]
                
                inf_mgt = numInfecMat[g_idNext, tt] / ((Float64(mgt + 1.0) / K)^q)
                logProbStoSgivenSorE[g_idNext, tt] = -a - b * inf_mgt
                logProbStoEgivenSorE[g_idNext, tt] = safe_log1mexp(a + b * inf_mgt)
                
                inf_mgt = (numInfecMat[g_idNext, tt] + 1) / ((Float64(mgt + 1.0) / K)^q)
                logProbStoSgivenI[g_idNext, tt] = -a - b * inf_mgt
                logProbStoEgivenI[g_idNext, tt] = safe_log1mexp(a + b * inf_mgt)
                
                inf_mgt = numInfecMat[g_idNext, tt] / ((Float64(mgt) / K)^q)
                logProbStoSgivenD[g_idNext, tt] = -a - b * inf_mgt
                logProbStoEgivenD[g_idNext, tt] = safe_log1mexp(a + b * inf_mgt)
            end
        end
    end
    
    return (numInfecMat=numInfecMat, mPerGroup=mPerGroup,
            logProbStoSgivenSorE=logProbStoSgivenSorE, logProbStoEgivenSorE=logProbStoEgivenSorE,
            logProbStoSgivenI=logProbStoSgivenI, logProbStoEgivenI=logProbStoEgivenI,
            logProbStoSgivenD=logProbStoSgivenD, logProbStoEgivenD=logProbStoEgivenD)
end

function iFFBS_updateTransitionProbabilities(
    logProbRest,
    logTransProbRest,
    X,
    SocGroup,
    LogProbDyingMat,
    LogProbSurvMat,
    logProbStoSgivenSorE,
    logProbStoEgivenSorE,
    logProbStoSgivenI,
    logProbStoEgivenI,
    logProbStoSgivenD,
    logProbStoEgivenD,
    logProbEtoE,
    logProbEtoI,
    whichRequireUpdate,
    id,
    idNext,
    m,
    maxt,
    numStates)
    
    if id < m
        c = (id - 1) * (maxt - 1)
        
        for tt in 1:maxt-1
            iFFBScalcLogProbRest(id, tt, logProbRest, X, SocGroup,
                               LogProbDyingMat, LogProbSurvMat,
                               logProbStoSgivenSorE, logProbStoEgivenSorE,
                               logProbStoSgivenI, logProbStoEgivenI,
                               logProbStoSgivenD, logProbStoEgivenD,
                               logProbEtoE, logProbEtoI)
            
            for s in 1:numStates
                if id == 1
                    logTransProbRest[s, tt] += (0.0 - logProbRest[tt, s, idNext])
                else
                    logTransProbRest[s, tt] += (logProbRest[tt, s, id] - logProbRest[tt, s, idNext])
                end
                logProbRest[tt, s, idNext] = 0.0
            end
            
            for jj in whichRequireUpdate[c + tt]
                if X[jj, tt] == 0.0
                    for s in 1:numStates
                        logTransProbRest[s, tt] -= logProbRest[tt, s, jj]
                    end
                    
                    g_1 = SocGroup[jj, tt]
                    
                    if X[jj, tt+1] == 0.0
                        logProbRest[tt, 1, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenSorE[g_1, tt]
                        logProbRest[tt, 2, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenSorE[g_1, tt]
                        logProbRest[tt, 3, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenI[g_1, tt]
                        logProbRest[tt, 4, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenD[g_1, tt]
                    elseif X[jj, tt+1] == 3.0
                        logProbRest[tt, 1, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenSorE[g_1, tt]
                        logProbRest[tt, 2, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenSorE[g_1, tt]
                        logProbRest[tt, 3, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenI[g_1, tt]
                        logProbRest[tt, 4, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenD[g_1, tt]
                    end
                    
                    for s in 1:numStates
                        logTransProbRest[s, tt] += logProbRest[tt, s, jj]
                    end
                end
            end
        end
    end
    
    return (logProbRest=logProbRest, logTransProbRest=logTransProbRest)
end
