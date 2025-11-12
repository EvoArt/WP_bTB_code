# Julia implementations of TrProbDeath_ and TrProbSurvive_ functions
# Ported from C++ versions in TrProbDeath_.cpp and TrProbSurvive_.cpp

"""
    TrProbDeath_(age, a2, b2, c1, logar=false)

Calculate transition probability of death given age and Gompertz parameters.
Ported from C++ TrProbDeath_.cpp
"""
function TrProbDeath_(age::Float64, a2::Float64, b2::Float64, c1::Float64, logar::Bool=false)
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
function TrProbSurvive_(age::Float64, a2::Float64, b2::Float64, c1::Float64, logar::Bool=true)
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
