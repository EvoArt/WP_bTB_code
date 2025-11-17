function iFFBS_(alpha_js, 
            b, q, tau,k, K,
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
            id,
            birthTime,
            startTime,
            endTime,
            X,
            seasonVec,
            TestMat_i, 
            TestTimes_i,
            CaptHist,
            corrector,
            predProb,
            filtProb,
            logTransProbRest,
            numInfecMat, 
            SocGroup,
            mPerGroup,
            idVecAll, 
            logProbStoSgivenSorE, 
            logProbStoEgivenSorE, 
            logProbStoSgivenI, 
            logProbStoEgivenI, 
            logProbStoSgivenD, 
            logProbStoEgivenD, 
            logProbEtoE, 
            logProbEtoI,
            whichRequireUpdate,
            sumLogCorrector)
  
  m,maxt = size(X)
  numStates = size(filtProb,2)
  t0 = startTime #- 1
  maxt_i = endTime - t0
#println("startTime = $startTime")
#println("birthTime = $birthTime")
#println("Soc group:")
#println(SocGroup[id, :])
##println(hcat(SocGroup[id, :],logProbStoSgivenSorE[SocGroup[id, :]]))

  # update corrector
  ObsProcess!(corrector, t0, endTime, id, CaptHist, TestMat_i, TestTimes_i, 
              etas, thetas, rhos, phis, seasonVec)
  

  idNext = id<m ? id+1 : 1 # 
  # Forward Filtering --------------------------------------
  
  prDeath = 0.5 # place holder
            # These lines allocate. Does it matter?
  unnormFiltProb = zeros(numStates)
  transProbRest =  zeros(numStates)
  log_probs_minus_B = zeros(numStates)
  

  nuE_i = 0.0
  nuI_i = 0.0
  
  if birthTime < startTime   # born before monitoring started
                # Is min the best function to use here?
    nuIdx = findfirst(nuTimes .== startTime)
    if nuIdx === nothing
      #println("nuIdx = nothing")
      #println("startTime = $startTime")
      #println("nuTimes = $nuTimes")
    end
    nuE_i = nuEs[nuIdx]
    nuI_i = nuIs[nuIdx]
  end# otherwise, nuE_i = nuI_i = 0.0

  # The grid of forward sweep starts at t0.
  # t0: either the beginning of the study or the date of birth
  # The individual must be alive at t0.
  # If it was born at or after the beginning of the study, it's assumed to be 
  # susceptible at t0. Otherwise, nuE and nuI are used.
  predProb[t0,1] = 1.0-nuE_i-nuI_i  # State 1: Susceptible
  predProb[t0,2] = nuE_i  # State 2: Exposed
  predProb[t0,3] = nuI_i  # State 3: Infectious
  predProb[t0,4] = 0.0  # State 4: Dead
 # println("nuI_i:birth:start = $([nuI_i birthTime startTime])")
  #println("t0 = $(t0)")
  if t0 < maxt-1
              # why is transProbRest not pre calculated for each row of logTransProbRest?
              # or tick off each t0 and only calc once for each?
              # (Maybe answered below - updating transprobs in loop).
              # Breaking code into smaller functions may make thi clearer.
              # also logTransProbRest.row(t0) allocates.
    #logTransProbRest_row = logTransProbRest[t0,:]
    #transProbRest = normTransProbRest(logTransProbRest_row)
    transProbRest .= logTransProbRest[:,t0]
    normTransProbRest!(transProbRest,log_probs_minus_B)
    for s in 1:numStates
      unnormFiltProb[s] = corrector[t0,s] * predProb[t0,s] * transProbRest[s]
    end
   
  else
    for s in 1:numStates
      unnormFiltProb[s] = corrector[t0,s] * predProb[t0,s]
    end
    
  end
  # Track when all filtered probabilities are zero
  if sum(unnormFiltProb) == 0.0
    println("DEBUG: All filtered probabilities are zero for individual i=$id at time t0=$t0")
    println("DEBUG: oldStatus=$(X[id, t0]), predProb=$(predProb[t0,:]), transProbRest=$(transProbRest)")
    println("DEBUG: corrector=$(corrector[:,t0]), logTransProbRest=$(logTransProbRest[:,t0])")
    println("DEBUG: unnormFiltProb calculation:")
    for s in 1:numStates
      println("  State $s: corrector[$t0,$s]=$(corrector[t0,s]) * predProb[$t0,$s]=$(predProb[t0,s]) * transProbRest[$s]=$(transProbRest[s]) = $(unnormFiltProb[s])")
    end
    # Set to uniform before normalization to prevent NaN
    unnormFiltProb .= [0.25, 0.25, 0.25, 0.25]
  end
  
  filtProb[t0,:] .= unnormFiltProb ./ sum(unnormFiltProb)
  
  # Handle NaN probabilities - set to uniform as fallback
  if any(isnan, filtProb[t0,:])
    println("WARNING: NaN filtering probabilities detected, setting to uniform")
    filtProb[t0,:] .= [0.25, 0.25, 0.25, 0.25]
  end
  # #println("unnormFiltProb = $(unnormFiltProb)")
  #  #println("predProb = $(predProb[t0,:])")
  #  #println("transProbRest = $(transProbRest)")
  
            # I guess because of movement between groups and births/deaths it is
            # not straightforward to do pre-calc for each group
  if maxt_i>2
    hitnan = false
    # t=2,...,T-1
    for tt in 1:maxt_i-1

      g = SocGroup[id, tt-1+t0]

      prDeath = probDyingMat[id, tt+t0]
      
      p00 = (1-prDeath)*exp(logProbStoSgivenSorE[g, tt-1+t0])
      p01 = (1-prDeath)*exp(logProbStoEgivenSorE[g, tt-1+t0])
      p11 = (1-prDeath)*exp(logProbEtoE)
      p12 = (1-prDeath)*exp(logProbEtoI)
      p22 = (1-prDeath)
      
      predProb[tt+t0,1] = p00*(filtProb[tt-1+t0,1])
      predProb[tt+t0,2] = p01*(filtProb[tt-1+t0,1]) + p11*filtProb[tt-1+t0,2]
      predProb[tt+t0,3] = p12*(filtProb[tt-1+t0,2]) + p22*filtProb[tt-1+t0,3]
      predProb[tt+t0,4] = prDeath*filtProb[tt-1+t0,1] + 
                          prDeath*filtProb[tt-1+t0,2] + 
                          prDeath*filtProb[tt-1+t0,3] +
                          1*filtProb[tt-1+t0,4]
      
      #logTransProbRest_row = logTransProbRest[tt+t0,:]
      #transProbRest = normTransProbRest(logTransProbRest_row)
       transProbRest .= logTransProbRest[:,tt+t0]
      normTransProbRest!(transProbRest,log_probs_minus_B)
      # row major indexing pain
      for s in 1:numStates 
        unnormFiltProb[s] = corrector[tt+t0,s] * predProb[tt+t0,s] * transProbRest[s]
      end
      
      filtProb[tt+t0,:] .= unnormFiltProb ./ sum(unnormFiltProb)
      if any(isnan,filtProb[tt+t0,:]) & (hitnan == false)
        hitnan = true
        println("p00 = $p00")
        println("p01 = $p01")
        println("p11 = $p11")
        println("p12 = $p12")
        println("p22 = $p22")
        println("prDeath = $prDeath")
        println("logprobStoSgivenSorE = $(logProbStoSgivenSorE[g, tt-1+t0])")
        println("logprobStoEgivenSorE = $(logProbStoEgivenSorE[g, tt-1+t0])")
        println("filtProb = $(filtProb[tt+t0,:])")
        println("filtProb WAS $(filtProb[tt+t0-1,:])")
        println("unnormFiltProb = $(unnormFiltProb)")
        println("predProb = $(predProb[tt+t0,:])")
        println("predProb WAS $(predProb[tt+t0-1,:])")
        println("transProbRest = $(transProbRest)")
        println("corrector = $(corrector[tt+t0,:])")
        println("logTransProbRest = $(logTransProbRest[:,tt+t0])")
        println("logprobEtoE = $(logProbEtoE)")
        println("logprobEtoI = $(logProbEtoI)")
        println("prDeath = $prDeath")
      end
      #=
      #println("p00 = $p00")
      #println("p01 = $p01")
      #println("p11 = $p11")
      #println("p12 = $p12")
      #println("p22 = $p22")
      #println("prDeath = $prDeath")
      #println("logprobStoSgivenSorE = $(logProbStoSgivenSorE[g, tt-1+t0])")
      #println("logprobStoEgivenSorE = $(logProbStoEgivenSorE[g, tt-1+t0])")
      #println("filtProb = $(filtProb[tt+t0,:])")
      #println("unnormFiltProb = $(unnormFiltProb)")
    #println("predProb = $(predProb[tt+t0,:])")
    #println("transProbRest = $(transProbRest)")
    =#
    end
  end
      ##println(filtProb[end-4:end,:])

  # t=T
  if maxt_i>=1
    
    tt = maxt_i
    
    g = SocGroup[id, tt-1+t0]
    
    prDeath = probDyingMat[id, tt+t0]
    
    p00 = (1-prDeath)*exp(logProbStoSgivenSorE[g, tt-1+t0]) 
    p01 = (1-prDeath)*exp(logProbStoEgivenSorE[g, tt-1+t0])
    p11 = (1-prDeath)*exp(logProbEtoE)
    p12 = (1-prDeath)*exp(logProbEtoI)
    p22 = (1-prDeath)
    
    predProb[tt+t0,1] = p00*(filtProb[tt-1+t0,1])
    predProb[tt+t0,2] = p01*(filtProb[tt-1+t0,1]) + p11*filtProb[tt-1+t0,2]
    predProb[tt+t0,3] = p12*(filtProb[tt-1+t0,2]) + p22*filtProb[tt-1+t0,3]
    predProb[tt+t0,4] = prDeath*filtProb[tt-1+t0,1] + 
      prDeath*filtProb[tt-1+t0,2] + 
      prDeath*filtProb[tt-1+t0,3] +
      1*filtProb[tt-1+t0,4]

    if tt+t0 < maxt-1
      #logTransProbRest_row = logTransProbRest[tt+t0,:]
      #transProbRest = normTransProbRest(logTransProbRest_row)
      transProbRest .= logTransProbRest[:,tt+t0]
      normTransProbRest!(transProbRest,log_probs_minus_B)
      for s in 1:numStates
        unnormFiltProb[s] = corrector[tt+t0,s] * predProb[tt+t0,s] * transProbRest[s]
      end
    else 
      for s=1:numStates
        unnormFiltProb[s] = corrector[tt+t0,s] * predProb[tt+t0,s]
      end
    end

    # Handle numerical issues in normalization
    sum_unnorm = sum(unnormFiltProb)
    if sum_unnorm == 0.0 || !isfinite(sum_unnorm)
      # If sum is zero or infinite/NaN, use uniform distribution
      println("⚠️  NUMERICAL ISSUE in forward filter at tt=$tt, id=$id: sum_unnorm=$sum_unnorm")
      println("   unnormFiltProb: $unnormFiltProb")
      println("   Falling back to uniform distribution")
      filtProb[tt+t0,:] .= 1.0 / numStates
    else
      filtProb[tt+t0,:] .= unnormFiltProb ./ sum_unnorm
    end

  end
  
  
  
  # Rcout << "forward sweep: " << std::endl
  
  # Backward Sampling --------------------------------------

 # #println(logProbStoSgivenSorE[g,t0:maxt_i])
  ##println(logProbStoSgivenSorE[g,:])

   # #println(filtProb[endTime-4:endTime,:])
  states =  [0, 3, 1, 9]  # 1-based states
  probs =  filtProb[endTime,:]
  
  # Additional safety check for probs before sampling
  if any(isnan,probs) || any(isinf,probs)
    # Replace any NaN/Inf with uniform distribution
    println("⚠️  NUMERICAL ISSUE in backward sampling for id=$id at endTime=$endTime")
    println("   probs: $probs")
    println("   Falling back to uniform distribution")
    probs = fill(1.0 / length(probs), length(probs))
  else
    # Normalize to ensure sum = 1
    probs = probs ./ sum(probs)
  end
  #=
  println("endTime = $endTime")
  println("probs = $probs")
  println("maxt = $maxt_i")
  println("t0 = $t0")
  println("predProb = $(predProb[1:4,:])")
  =#
  ##println(sum(isnan.(logProbStoEgivenSorE)))
  if sum(isnan,probs)>50
    println("Soc group:")
    println(SocGroup[id, :])
    println("filtProb = $(filtProb)")
    println("predProb = $(predProb)")
    println("corrector = $(corrector)")
    println("logTransProbRest = $(logTransProbRest)") 
  end
  #newStatus = sample(states,Weights(probs))
  newStatus = states[rand(Categorical(probs))] 
  ##println("newStatus = $newStatus")

  X[id, endTime] = newStatus
  oldStatus = newStatus
    
  # tt will start from maxt_i-1 and finish at 1
  ##println(maxt_i)
  if maxt_i>1
    for tt=maxt_i-1:-1:1
      
      g = SocGroup[id, tt+t0]
      mgt = mPerGroup[g, tt+t0]
      
      a = alpha_js[g]
        
      inf_mgt = numInfecMat[g, tt+t0]/((Float64(mgt + 1.0)/K)^q)  

      prDeath = probDyingMat[id, tt+1+t0]
      
      p00 = (1-prDeath)*exp(-a - b*inf_mgt)
      p01 = (1-prDeath)*(1 - exp(-a - b*inf_mgt))
                # Are these type conversions costly?
                # Does the compiled code calculate tau/k every iter?
      p11 = (1-prDeath)*(1.0 - cdf(Erlang(k, tau/k), 1))
      p12 = (1-prDeath)*cdf(Erlang(k, tau/k), 1)
      p22 = (1-prDeath)
      ##println("pars")
      ##println("-----")
      ##println(p00)
      ##println(p01)
      ##println(p11)
      ##println(p12)
      ##println(p22)
      ##println(prDeath)
      ##println(a)
      ##println(inf_mgt)
      ##println(mgt)
      ##println("-----")
      
      p09 = prDeath
      p19 = prDeath
      p29 = prDeath
      
      probSuscep_t=0.0
      probE_t=0.0
      probI_t=0.0
      probDead_t=0.0
      
      if X[id, tt+1+t0] == 0.0
        probSuscep_t = (p00*filtProb[tt+t0, 1])/(predProb[tt+1+t0, 1])
       ###println("1: probSuscep_t = $(probSuscep_t)")
        probE_t = 0.0
        probI_t = 0.0
        probDead_t = 0.0
      elseif X[id, tt+1+t0] == 3
        probSuscep_t = (p01*filtProb[tt+t0, 1])/(predProb[tt+1+t0, 2])
       # ##println("2: probSuscep_t = $(probSuscep_t)")
        probE_t = (p11*filtProb[tt+t0, 2])/(predProb[tt+1+t0, 2])
        probI_t = 0.0
        probDead_t = 0.0
      elseif X[id, tt+1+t0] == 1
        probSuscep_t = 0.0
        probE_t = (p12*filtProb[tt+t0, 2])/(predProb[tt+1+t0, 3])
        probI_t = (p22*filtProb[tt+t0, 3])/(predProb[tt+1+t0, 3])
        ###println("3: probSuscep_t = $(probSuscep_t)")
        probDead_t = 0.0
      elseif X[id, tt+1+t0] == 9
        probSuscep_t = (p09*filtProb[tt+t0, 1])/(predProb[tt+1+t0, 4])
       # ##println("4: probSuscep_t = $(probSuscep_t)")
        probE_t = (p19*filtProb[tt+t0, 2])/(predProb[tt+1+t0, 4])
        probI_t = (p29*filtProb[tt+t0, 3])/(predProb[tt+1+t0, 4])
        probDead_t = filtProb[tt+t0, 4]/predProb[tt+1+t0, 4]
      end
      
      probs = [probSuscep_t, probE_t, probI_t, probDead_t]
      
      # Handle NaN/Inf probabilities - set to uniform as fallback
      # Using any(f, collection) avoids allocating intermediate boolean array
      if any(isnan, probs) || any(isinf, probs)
          println("WARNING: NaN/Inf sampling probabilities detected, setting to uniform")
          probs = [0.25, 0.25, 0.25, 0.25]
      end
      
      ###println(probs)
      if (tt==1) && (birthTime>=startTime) 
        probs = [1.0, 0.0, 0.0, 0.0]
      end

       ##println("probs = $probs")
       
  # Debug: Track when E→S transitions might occur
          #### Weights normalises. C++ also does this.
  newStatus = sample(states,Weights(probs))
  #  newStatus = states[rand(Categorical(probs))] 
  ##println("newStatus = $newStatus")


      X[id, tt+t0] = newStatus
      
    end
  end
  
  # Rcout << "backward probs calculated: " << std::endl
  
  #   #    #   #    #   #    #   #  
  # calculating log of density of observation process
  
  for tt in 1:maxt_i
    
    # Rcout << "tt+t0: " << tt+t0 << std::endl
    # Rcout << "corrector.row(tt+t0): " << corrector.row(tt+t0) << std::endl
    
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
  #   #    #   #    #   #    #   #    
  
  
  
  
  
  # Rcout << "now will start updating numInfecMat, etc for next ID " << std::endl
  
  
  # Updating mPerGroup, numInfecMat, logProbStoSgivenSorE, etc for the next individual (id or 0) -----
  
  for tt in 1:maxt-1  # idNext may be included after endTime
  
    g = SocGroup[id, tt]
    g_idNext = SocGroup[idNext, tt]

    # Rcout << "id: " << id << std::endl
    # Rcout << "idNext: " << idNext << std::endl
    # Rcout << "g: " << g << std::endl
    # Rcout << "g_idNext: " << g_idNext << std::endl
    # Rcout << "a: " << a << std::endl
              # is long type (L) needed here?
    if (g == g_idNext) && (g != 0)
      
      # # updating/correcting numInfecMat for the next individual
                # This seems convoluted but avoids accessing numInfecMat twice.
                # Rather than 3 `if` statements, could just add infecToAdd regardless.
                # If it's 0, doesn't matter.
                # Could drop all `if` statements in fact. unless these are much cheaper
                # than addition and array access:
                # numInfecMat(g_idNext-1, tt) += X(id-1, tt) - X(idNext, tt)
                # possible alternative, still avoiding array access:
                # infecToAdd = X(id-1,tt) - X(idNext,tt)
                # if (infecToAdd != 0) {
                # numInfecMat(g_idNext-1, tt) += infecToAdd
                #end

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

      # # updating/correcting mPerGroup for the next individual
                # This eems okay. most Xs will be 0, so should be fast.
                # Don't know if it's worth storing dead badgers in a speific
                # array for this.
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
        
        # mgt
        # inf_mgt
        mgt = mPerGroup[g_idNext, tt]
        
        inf_mgt = numInfecMat[g_idNext, tt]/((Float64(mgt + 1.0)/K)^q)
        logProbStoSgivenSorE[g_idNext, tt] = -a - b*inf_mgt
        logProbStoEgivenSorE[g_idNext, tt] = safe_log1mexp(a + b*inf_mgt)
        inf_mgt = (numInfecMat[g_idNext, tt]+1)/((Float64(mgt + 1.0)/K)^q)
        logProbStoSgivenI[g_idNext, tt] = -a - b*inf_mgt
        logProbStoEgivenI[g_idNext, tt] = safe_log1mexp(a + b*inf_mgt)
        
        inf_mgt = numInfecMat[g_idNext, tt]/((Float64(mgt)/K)^q)
        logProbStoSgivenD[g_idNext, tt] = -a - b*inf_mgt
        logProbStoEgivenD[g_idNext, tt] = safe_log1mexp(a + b*inf_mgt)
        
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
        
        # mgt
        # inf_mgt
        
        mgt = mPerGroup[g, tt] 
        
        inf_mgt = numInfecMat[g, tt]/((Float64(mgt)/K)^q)
        logProbStoSgivenSorE[g, tt] = -a - b*inf_mgt
        logProbStoEgivenSorE[g, tt] = safe_log1mexp(a + b*inf_mgt)
        # inf_mgt = numInfecMat(g-1, tt)/(^(float mgt, q))
        logProbStoSgivenI[g, tt] = -a - b*inf_mgt
        logProbStoEgivenI[g, tt] = safe_log1mexp(a + b*inf_mgt)
        
        # inf_mgt = numInfecMat(g-1, tt)/(^(float mgt, q))
        logProbStoSgivenD[g, tt] = -a - b*inf_mgt
        logProbStoEgivenD[g, tt] = safe_log1mexp(a + b*inf_mgt)    
              
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
        
        # mgt
        # inf_mgt
        mgt = mPerGroup[g_idNext, tt]
        
        inf_mgt = numInfecMat[g_idNext, tt]/((Float64(mgt + 1.0)/K)^q)
        logProbStoSgivenSorE[g_idNext, tt] = -a - b*inf_mgt
        logProbStoEgivenSorE[g_idNext, tt] = safe_log1mexp(a + b*inf_mgt)

        inf_mgt = (numInfecMat[g_idNext, tt]+1)/((Float64(mgt + 1.0)/K)^q)
        logProbStoSgivenI[g_idNext, tt] = -a - b*inf_mgt
        logProbStoEgivenI[g_idNext, tt] = safe_log1mexp(a + b*inf_mgt)
        
        inf_mgt = numInfecMat[g_idNext, tt]/((Float64(mgt)/K)^q)
        logProbStoSgivenD[g_idNext, tt] = -a - b*inf_mgt
        logProbStoEgivenD[g_idNext, tt] = safe_log1mexp(a + b*inf_mgt)
        
      end
      
    end
    
  end
  
  # Rcout << "numInfecMat for next ID calculated " << std::endl
  
  # at id==m, we don't need to update logTransProbRest, because
  # the next individual (id==1) will be updated separately to 
  # consider new rate values
  if id < m
    
    c = (id-1)*(maxt-1)
    # g_1  # This variable will be used later

    # current individual (id) will be in logProbRest when updating idNext
    for tt in 1:maxt-1
    # for(unsigned tt=0 tt<endTime-1 tt++){
      
      # update logProbRest(tt,_,id-1)
      # if(X(id-1, tt)==0){
                # So this is why we cant pre-calc transpobs?
        iFFBScalcLogProbRest(id, tt, logProbRest, X, SocGroup, 
                             LogProbDyingMat, LogProbSurvMat, 
                             logProbStoSgivenSorE, logProbStoEgivenSorE, 
                             logProbStoSgivenI, logProbStoEgivenI, 
                             logProbStoSgivenD, logProbStoEgivenD, 
                             logProbEtoE, logProbEtoI)      
      
      # adding logProbRest(tt,_,id-1) to logTransProbRest and 
      # removing logProbRest(tt,s,idNext)
      # (idNext must not be included in logProbRest when updating idNext)
      for s in 1:numStates
        if id == 1
          # Special case: individual 1 was excluded from logTransProbRest initialization
          # so we add zeros (equivalent to C++ logProbRest(tt,s,0L) which is uninitialized)
          logTransProbRest[s, tt] += (0.0 - logProbRest[tt, s, idNext])
        else
          # Normal case: use the individual's contribution
          logTransProbRest[s, tt] += (logProbRest[tt, s, id] - logProbRest[tt, s, idNext])
        end
        logProbRest[tt, s, idNext] = 0.0
      end
      
      # update logProbRest(tt,_,idNext) at flagged time points for the 
      # remaining m-2 elements
      # This should be required only for individuals that belong to the 
      # same group as id or idNext

      # pos = (id-1)*(maxt-1) + tt
      # whichArma = whichRequireUpdate(pos)
      # for(auto & jj : whichArma){
      
      for jj in whichRequireUpdate[c + tt]
      
        if X[jj, tt] == 0.0

          for s in 1:numStates
            logTransProbRest[s, tt] -= logProbRest[tt, s, jj]
          end
              
          g_1 = SocGroup[jj, tt]# do i need the -1 here? Dont think so
          
          if X[jj, tt+1] == 0.0
            logProbRest[tt, 1, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenSorE[g_1, tt]
            logProbRest[tt, 2, jj] = LogProbSurvMat[jj, tt+1] + logProbStoEgivenSorE[g_1, tt]
            logProbRest[tt, 3, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenI[g_1, tt]
            logProbRest[tt, 4, jj] = LogProbSurvMat[jj, tt+1] + logProbStoSgivenD[g_1, tt]
        elseif X[jj, tt+1] == 3
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
  
  # Rcout << "logTransProbRest for next ID calculated: " << std::endl

end

# Mathematical helper functions
# safe_log1mexp is defined in dimension_corrections.jl

# Using this:
# for(auto & jj : whichRequireUpdate(c + tt)){
#   if(X(jj, tt)==0){
# is equivalent to using the loop and if condition below
# 
# for(jj=0 jj<m jj++){
# if(((jj!=id-1)&&(jj!=idNext))&&
#    ((SocGroup(jj, tt)==SocGroup(id-1, tt))||
#     (SocGroup(jj, tt)==SocGroup(idNext, tt)))&&
# (X(jj, tt)==0)){


############ 
#
# List outList(5)
# outList[0] = Xid
# outList[1] = predProb
# outList[2] = filtProb
# return(outList)