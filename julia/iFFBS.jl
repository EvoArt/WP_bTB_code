
using OffsetArrays: Origin # to use 0-based indexing

macro zero_based(x)
  return :( Origin(0)($x) )
end

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
  t0 = startTime - 1
  maxt_i = endTime - t0

  # update corrector
  ObsProcess!(corrector, t0, endTime, id, CaptHist, TestMat_i, TestTimes_i, 
              etas, thetas, rhos, phis, seasonVec)
  

  idNext = id<m ? id : 0
  # Forward Filtering --------------------------------------
  
  prDeath = 0.5 # place holder
            # These lines allocate. Does it matter?
  unnormFiltProb = @zero_based zeros(numStates)
  transProbRest =  @zero_based zeros(numStates)
  

  nuE_i = 0.0
  nuI_i = 0.0
  
  if birthTime < startTime   # born before monitoring started
                # Is min the best function to use here?
    nuIdx = findfirst(nuTimes .== startTime)
    nuE_i = nuEs[nuIdx]
    nuI_i = nuIs[nuIdx]
  end# otherwise, nuE_i = nuI_i = 0.0

  # The grid of forward sweep starts at t0.
  # t0: either the beginning of the study or the date of birth
  # The individual must be alive at t0.
  # If it was born at or after the beginning of the study, it's assumed to be 
  # susceptible at t0. Otherwise, nuE and nuI are used.
  predProb[t0,0] = 1.0-nuE_i-nuI_i
  predProb[t0,1] = nuE_i
  predProb[t0,2] = nuI_i
  predProb[t0,3] = 0.0
  
  
  if t0 < maxt-1
              # why is transProbRest not pre calculated for each row of logTransProbRest?
              # or tick off each t0 and only calc once for each?
              # (Maybe answered below - updating transprobs in loop).
              # Breaking code into smaller functions may make thi clearer.
              # also logTransProbRest.row(t0) allocates.
    logTransProbRest_row = @zero_based logTransProbRest[t0,:]
    transProbRest = @zero_based normTransProbRest(logTransProbRest_row)
    for s in 0:numStates-1
      unnormFiltProb[s] = corrector[t0,s] * predProb[t0,s] * transProbRest[s]
    end
  else
    for s in 0:numStates-1
      unnormFiltProb[s] = corrector[t0,s] * predProb[t0,s]
    end
  end
  filtProb[t0,:] .= unnormFiltProb ./ sum(unnormFiltProb)
  
            # I guess because of movement between groups and births/deaths it is
            # not straightforward to do pre-calc for each group
  if maxt_i>2
    # t=2,...,T-1
    for tt in 1:maxt_i-2
      
      g = SocGroup[id-1, tt-1+t0]

      prDeath = probDyingMat[id-1, tt+t0]
      
      p00 = (1-prDeath)*exp(logProbStoSgivenSorE[g-1, tt-1+t0])
      p01 = (1-prDeath)*exp(logProbStoEgivenSorE[g-1, tt-1+t0])
      p11 = (1-prDeath)*exp(logProbEtoE)
      p12 = (1-prDeath)*exp(logProbEtoI)
      p22 = (1-prDeath)
      
      predProb[tt+t0,0] = p00*(filtProb[tt-1+t0,0])
      predProb[tt+t0,1] = p01*(filtProb[tt-1+t0,0]) + p11*filtProb[tt-1+t0,1]
      predProb[tt+t0,2] = p12*(filtProb[tt-1+t0,1]) + p22*filtProb[tt-1+t0,2]
      predProb[tt+t0,3] = prDeath*filtProb[tt-1+t0,0] + 
                          prDeath*filtProb[tt-1+t0,1] + 
                          prDeath*filtProb[tt-1+t0,2] +
                          1*filtProb[tt-1+t0,3]
      
      logTransProbRest_row = @zero_based logTransProbRest[tt+t0,:]
      transProbRest = @zero_based normTransProbRest(logTransProbRest_row)
      for s in 0:numStates-1 
        unnormFiltProb[s] = corrector[tt+t0,s] * predProb[tt+t0,s] * transProbRest[s]
      end
      
      filtProb[tt+t0,:] .= unnormFiltProb ./ sum(unnormFiltProb)

    end
  end
  
  # t=T
  if maxt_i>=1
    
    tt = maxt_i-1
    
    g = SocGroup[id-1, tt-1+t0]
    
    prDeath = probDyingMat[id-1, tt+t0]
    
    p00 = (1-prDeath)*exp(logProbStoSgivenSorE[g-1, tt-1+t0]) 
    p01 = (1-prDeath)*exp(logProbStoEgivenSorE[g-1, tt-1+t0])
    p11 = (1-prDeath)*exp(logProbEtoE)
    p12 = (1-prDeath)*exp(logProbEtoI)
    p22 = (1-prDeath)
    
    predProb[tt+t0,0] = p00*(filtProb[tt-1+t0,0])
    predProb[tt+t0,1] = p01*(filtProb[tt-1+t0,0]) + p11*filtProb[tt-1+t0,1]
    predProb[tt+t0,2] = p12*(filtProb[tt-1+t0,1]) + p22*filtProb[tt-1+t0,2]
    predProb[tt+t0,3] = prDeath*filtProb[tt-1+t0,0] + 
      prDeath*filtProb[tt-1+t0,1] + 
      prDeath*filtProb[tt-1+t0,2] +
      1*filtProb[tt-1+t0,3]

    if tt+t0 < maxt-1
      logTransProbRest_row = @zero_based logTransProbRest[tt+t0,:]
      transProbRest = @zero_based normTransProbRest(logTransProbRest_row)
      for s in 0:numStates-1
        unnormFiltProb[s] = corrector[tt+t0,s] * predProb[tt+t0,s] * transProbRest[s]
      end
    else 
      for s=0:numStates-1
        unnormFiltProb[s] = corrector[tt+t0,s] * predProb[tt+t0,s]
      end
    end

    filtProb[tt+t0,:] .= unnormFiltProb ./ sum(unnormFiltProb)

  end
  
  
  
  # Rcout << "forward sweep: " << std::endl
  
  # Backward Sampling --------------------------------------
    
  states =  (0, 3, 1, 9)
  probs =  @zero_based filtProb[endTime-1,:]
  
  newStatus = sample(states,Weights(probs))
  
  X[id-1, endTime-1] = newStatus
    
  # tt will start from maxt_i-2 and finish at 0
  if maxt_i>1
    for tt=maxt_i-1:-1:0
      
      g = SocGroup[id+1, tt+t0+1]
      mgt = mPerGroup[g+1, tt+t0+1]
      
      a = alpha_js[g+1]
        
      inf_mgt = numInfecMat[g+1, tt+t0+1]/((Float64(mgt + 1.0)/K)^q)  

      prDeath = probDyingMat[id+1, tt+1+t0]
      
      p00 = (1-prDeath)*exp(-a - b*inf_mgt)
      p01 = (1-prDeath)*(1 - exp(-a - b*inf_mgt))
                # Are these type conversions costly?
                # Does the compiled code calculate tau/k every iter?
      p11 = (1-prDeath)*(1.0 - cdf(Erlang(k, tau/k), 1))
      p12 = (1-prDeath)*cdf(Erlang(k, tau/k), 1)
      p22 = (1-prDeath)
      
      p09 = prDeath
      p19 = prDeath
      p29 = prDeath
      
      probSuscep_t=0.0
      probE_t=0.0
      probI_t=0.0
      probDead_t=0.0
      
      if X[id+1, tt+1+t0+1] == 0
        probSuscep_t = (p00*filtProb[tt+t0+1, 1])/(predProb[tt+1+t0+1, 1])
        probE_t = 0.0
        probI_t = 0.0
        probDead_t = 0.0
      elseif X[id+1, tt+1+t0+1] == 3
        probSuscep_t = (p01*filtProb[tt+t0+1, 1])/(predProb[tt+1+t0+1, 2])
        probE_t = (p11*filtProb[tt+t0+1, 2])/(predProb[tt+1+t0+1, 2])
        probI_t = 0.0
        probDead_t = 0.0
      elseif X[id+1, tt+1+t0+1] == 1
        probSuscep_t = 0.0
        probE_t = (p12*filtProb[tt+t0+1, 2])/(predProb[tt+1+t0+1, 3])
        probI_t = (p22*filtProb[tt+t0+1, 3])/(predProb[tt+1+t0+1, 3])
        probDead_t = 0.0
      elseif X[id+1, tt+1+t0+1] == 9
        probSuscep_t = (p09*filtProb[tt+t0+1, 1])/(predProb[tt+1+t0+1, 4])
        probE_t = (p19*filtProb[tt+t0+1, 2])/(predProb[tt+1+t0+1, 4])
        probI_t = (p29*filtProb[tt+t0+1, 3])/(predProb[tt+1+t0+1, 4])
        probDead_t = filtProb[tt+t0+1, 4]/predProb[tt+1+t0+1, 4]
      end
      
      probs = @zero_based [probSuscep_t, probE_t, probI_t, probDead_t]
      
      if (tt==0) && (birthTime>=startTime) 
        probs = @zero_based [1.0, 0.0, 0.0, 0.0]
      end

      newStatus = sample(states,Weights(probs))


      X[id+1, tt+t0+1] = newStatus
      
    end
  end
  
  # Rcout << "backward probs calculated: " << std::endl
  
  #   #    #   #    #   #    #   #  
  # calculating log of density of observation process
  
  for tt in 0:maxt_i-1
    
    # Rcout << "tt+t0: " << tt+t0 << std::endl
    # Rcout << "corrector.row(tt+t0): " << corrector.row(tt+t0) << std::endl
    
    if X[id+1, tt+t0+1] == 0
      sumLogCorrector += log(corrector[tt+t0+1, 1])
    elseif X[id+1, tt+t0+1] == 3
      sumLogCorrector += log(corrector[tt+t0+1, 2])
    elseif X[id+1, tt+t0+1] == 1
      sumLogCorrector += log(corrector[tt+t0+1, 3])
    elseif X[id+1, tt+t0+1] == 9
      sumLogCorrector += log(corrector[tt+t0+1, 4])
    end
    
  end
  #   #    #   #    #   #    #   #    
  
  
  
  
  
  # Rcout << "now will start updating numInfecMat, etc for next ID " << std::endl
  
  
  # Updating mPerGroup, numInfecMat, logProbStoSgivenSorE, etc for the next individual (id or 0) -----
  
  for tt in 0:maxt-2  # idNext may be included after endTime
  
    g = SocGroup[id+1, tt+1]
    g_idNext = SocGroup[idNext+1, tt+1]

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
      if X[id+1, tt+1] == 1
        infecToAdd += 1
      end
      if X[idNext+1, tt+1] == 1
        infecToAdd -= 1
      end

      if (X[id+1, tt+1] == 1) || (X[idNext+1, tt+1] == 1)
        numInfecMat[g_idNext+1, tt+1] += infecToAdd
      end

      # # updating/correcting mPerGroup for the next individual
                # This eems okay. most Xs will be 0, so should be fast.
                # Don't know if it's worth storing dead badgers in a speific
                # array for this.
      mToAdd = 0
      if (X[id+1, tt+1] == 0) || (X[id+1, tt+1] == 1) || (X[id+1, tt+1] == 3)
        mToAdd += 1
      end
      if (X[idNext+1, tt+1] == 0) || (X[idNext+1, tt+1] == 1) || (X[idNext+1, tt+1] == 3)
        mToAdd -= 1
      end
      
      if ((X[id+1, tt+1] == 0) || (X[id+1, tt+1] == 1) || (X[id+1, tt+1] == 3)) || 
         ((X[idNext+1, tt+1] == 0) || (X[idNext+1, tt+1] == 1) || (X[idNext+1, tt+1] == 3))
        mPerGroup[g_idNext+1, tt+1] += mToAdd
      end
            
      if id < m
        
        a = alpha_js[g_idNext+1]
        
        # mgt
        # inf_mgt
        mgt = mPerGroup[g_idNext+1, tt+1]
        
        inf_mgt = numInfecMat[g_idNext+1, tt+1]/((Float64(mgt + 1.0)/K)^q)
        logProbStoSgivenSorE[g_idNext+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenSorE[g_idNext+1, tt+1] = log1mexp(a + b*inf_mgt)
        
        inf_mgt = (numInfecMat[g_idNext+1, tt+1]+1)/((Float64(mgt + 1.0)/K)^q)
        logProbStoSgivenI[g_idNext+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenI[g_idNext+1, tt+1] = log1mexp(a + b*inf_mgt)
        
        inf_mgt = numInfecMat[g_idNext+1, tt+1]/((Float64(mgt)/K)^q)
        logProbStoSgivenD[g_idNext+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenD[g_idNext+1, tt+1] = log1mexp(a + b*inf_mgt)
        
      end

      
    else
      
      if g != 0
        if X[id+1, tt+1] == 1
          numInfecMat[g+1, tt+1] += 1
        end
        if (X[id+1, tt+1] == 0) || (X[id+1, tt+1] == 1) || (X[id+1, tt+1] == 3)
          mPerGroup[g+1, tt+1] += 1
        end
      end
      
      if (id < m) && (g != 0)
        
        a = alpha_js[g+1]
        
        # mgt
        # inf_mgt
        
        mgt = mPerGroup[g+1, tt+1] 
        
        inf_mgt = numInfecMat[g+1, tt+1]/((Float64(mgt)/K)^q)
        logProbStoSgivenSorE[g+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenSorE[g+1, tt+1] = log1mexp(a + b*inf_mgt)
        
        # inf_mgt = numInfecMat(g-1, tt)/(^(float mgt, q))
        logProbStoSgivenI[g+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenI[g+1, tt+1] = log1mexp(a + b*inf_mgt)
        
        # inf_mgt = numInfecMat(g-1, tt)/(^(float mgt, q))
        logProbStoSgivenD[g+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenD[g+1, tt+1] = log1mexp(a + b*inf_mgt)    
              
      end
            
      if g_idNext != 0
        if X[idNext+1, tt+1] == 1
          numInfecMat[g_idNext+1, tt+1] -= 1
        end
        if (X[idNext+1, tt+1] == 0) || (X[idNext+1, tt+1] == 1) || (X[idNext+1, tt+1] == 3)
          mPerGroup[g_idNext+1, tt+1] -= 1
        end
      end

      if (id < m) && (g_idNext != 0)
        
        a = alpha_js[g_idNext+1]
        
        # mgt
        # inf_mgt
        mgt = mPerGroup[g_idNext+1, tt+1]
        
        inf_mgt = numInfecMat[g_idNext+1, tt+1]/((Float64(mgt + 1.0)/K)^q)
        logProbStoSgivenSorE[g_idNext+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenSorE[g_idNext+1, tt+1] = log1mexp(a + b*inf_mgt)
        
        inf_mgt = (numInfecMat[g_idNext+1, tt+1]+1)/((Float64(mgt + 1.0)/K)^q)
        logProbStoSgivenI[g_idNext+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenI[g_idNext+1, tt+1] = log1mexp(a + b*inf_mgt)
        
        inf_mgt = numInfecMat[g_idNext+1, tt+1]/((Float64(mgt)/K)^q)
        logProbStoSgivenD[g_idNext+1, tt+1] = -a - b*inf_mgt
        logProbStoEgivenD[g_idNext+1, tt+1] = log1mexp(a + b*inf_mgt)
        
      end
      
    end
    
  end
  
  # Rcout << "numInfecMat for next ID calculated " << std::endl
  
  # at id==m, we don't need to update logTransProbRest, because
  # the next individual (id==1) will be updated separately to 
  # consider new rate values
  if id < m
    
    c = (id)*(maxt-1)
    # g_1  # This variable will be used later

    # current individual (id) will be in logProbRest when updating idNext
    for tt in 0:maxt-2
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
      for s in 0:numStates-1
        logTransProbRest[tt+1, s+1] += (logProbRest[tt+1, s+1, id+1] - logProbRest[tt+1, s+1, idNext+1])
        logProbRest[tt+1, s+1, idNext+1] = 0.0
      end
      
      # update logProbRest(tt,_,idNext) at flagged time points for the 
      # remaining m-2 elements
      # This should be required only for individuals that belong to the 
      # same group as id or idNext

      # pos = (id-1)*(maxt-1) + tt
      # whichArma = whichRequireUpdate(pos)
      # for(auto & jj : whichArma){
      
      for jj in whichRequireUpdate[c + tt + 1]
      
        if X[jj+1, tt+1] == 0

          for s in 0:numStates-1
            logTransProbRest[tt+1, s+1] -= logProbRest[tt+1, s+1, jj+1]
          end
              
          g_1 = SocGroup[jj+1, tt+1] - 1
          
          if X[jj+1, tt+1] == 0
            logProbRest[tt+1, 1, jj+1] = LogProbSurvMat[jj+1, tt+1] + logProbStoSgivenSorE[g_1+1, tt+1]
            logProbRest[tt+1, 2, jj+1] = LogProbSurvMat[jj+1, tt+1] + logProbStoEgivenSorE[g_1+1, tt+1]
            logProbRest[tt+1, 3, jj+1] = LogProbSurvMat[jj+1, tt+1] + logProbStoSgivenI[g_1+1, tt+1]
            logProbRest[tt+1, 4, jj+1] = LogProbSurvMat[jj+1, tt+1] + logProbStoSgivenD[g_1+1, tt+1]
        elseif X[jj+1, tt+1] == 3
            logProbRest[tt+1, 1, jj+1] = LogProbSurvMat[jj+1, tt+1] + logProbStoEgivenSorE[g_1+1, tt+1]
            logProbRest[tt+1, 2, jj+1] = LogProbSurvMat[jj+1, tt+1] + logProbStoEgivenSorE[g_1+1, tt+1]
            logProbRest[tt+1, 3, jj+1] = LogProbSurvMat[jj+1, tt+1] + logProbStoEgivenI[g_1+1, tt+1]
            logProbRest[tt+1, 4, jj+1] = LogProbSurvMat[jj+1, tt+1] + logProbStoEgivenD[g_1+1, tt+1]
        end
              
        for s in 0:numStates-1
          logTransProbRest[tt+1, s+1] += logProbRest[tt+1, s+1, jj+1]
        end
          

        end
      end

    end
      
  end
  
  # Rcout << "logTransProbRest for next ID calculated: " << std::endl

end

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