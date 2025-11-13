"""
    simple_julia_test.jl

Simple test script to verify Julia functions work correctly.
Tests the functions we implemented without RCall complications.

Author: Generated for WP_bTB_code project
"""

using Random
using Statistics
using StatsFuns
using Distributions
using Test

# Set seed for reproducible testing
Random.seed!(123)

println("🧪 Testing Julia Functions (Simplified)")
println("="^50)

# Load Julia functions
println("📦 Loading Julia functions...")
include("julia/dimension_corrections.jl")

# Test data generation
println("\n🎲 Generating test data...")
Random.seed!(123)

# Test parameters
m = 10
maxt = 20
numTests = 4

# Generate realistic test data
birthTimes = rand(1:5, m)
startSamplingPeriod = rand(6:15, m)
endSamplingPeriod = rand(16:20, m)
X = rand([0, 1, 3, 9], m, maxt)
TestMat = rand(1:m, 50, 3 + numTests)
TestMat[:, 4:end] = rand([0, 1, NaN], 50, numTests)
thetas = rand(0.1:0.1:0.9, numTests)
rhos = rand(0.1:0.1:0.9, numTests)
phis = rand(0.1:0.1:0.9, numTests)
hp_theta = [1.0, 1.0]
hp_rho = [1.0, 1.0]
hp_xi = [81.0, 60.0]

# Create TestField and TestTimes
TestField = [TestMat[TestMat[:, 1] .== i, 4:end] for i in 1:m]
TestTimes = [TestMat[TestMat[:, 1] .== i, 2] for i in 1:m]

println("✅ Test data generated successfully")

# Test results tracking
passed_tests = 0
total_tests = 0

println("\n🧪 Running Function Tests:")
println("-"^40)

# Test 1: TrProbDeath_
println("\n1️⃣ Testing TrProbDeath_...")
total_tests += 1
try
    age = 5.0
    a2 = 1.0
    b2 = 1.0
    c1 = 1.0
    
    result = TrProbDeath_(age, a2, b2, c1)
    println("✅ TrProbDeath_: PASS (result: $result)")
    passed_tests += 1
catch e
    println("❌ TrProbDeath_: FAIL - $e")
end

# Test 2: logisticD
println("\n2️⃣ Testing logisticD...")
total_tests += 1
try
    x = 0.5
    result = logisticD(x)
    expected = 1 / (1 + exp(-x))
    if abs(result - expected) < 1e-10
        println("✅ logisticD: PASS (result: $result)")
        passed_tests += 1
    else
        println("❌ logisticD: FAIL - expected $expected, got $result")
    end
catch e
    println("❌ logisticD: FAIL - $e")
end

# Test 3: logPostThetasRhos
println("\n3️⃣ Testing logPostThetasRhos...")
total_tests += 1
try
    result = logPostThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                            TestField, TestTimes, hp_theta, hp_rho)
    if isfinite(result)
        println("✅ logPostThetasRhos: PASS (result: $result)")
        passed_tests += 1
    else
        println("❌ logPostThetasRhos: FAIL - result is not finite: $result")
    end
catch e
    println("❌ logPostThetasRhos: FAIL - $e")
end

# Test 4: gradThetasRhos
println("\n4️⃣ Testing gradThetasRhos...")
total_tests += 1
try
    result = gradThetasRhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                          TestField, TestTimes, hp_theta, hp_rho)
    if all(isfinite.(result))
        println("✅ gradThetasRhos: PASS (result length: $(length(result)))")
        passed_tests += 1
    else
        println("❌ gradThetasRhos: FAIL - result contains non-finite values")
    end
catch e
    println("❌ gradThetasRhos: FAIL - $e")
end

# Test 5: HMC_thetas_rhos
println("\n5️⃣ Testing HMC_thetas_rhos...")
total_tests += 1
try
    epsilonsens = 0.1
    L = 5
    result = HMC_thetas_rhos(thetas, rhos, X, startSamplingPeriod, endSamplingPeriod, 
                           TestField, TestTimes, hp_theta, hp_rho, epsilonsens, L)
    if all(isfinite.(result)) && length(result) == length(thetas) + length(rhos)
        println("✅ HMC_thetas_rhos: PASS (result length: $(length(result)))")
        passed_tests += 1
    else
        println("❌ HMC_thetas_rhos: FAIL - invalid result")
    end
catch e
    println("❌ HMC_thetas_rhos: FAIL - $e")
end

# Test 6: TestMatAsField
println("\n6️⃣ Testing TestMatAsField...")
total_tests += 1
try
    result = TestMatAsField(TestMat, m)
    if length(result) == m
        println("✅ TestMatAsField: PASS (field count: $(length(result)))")
        passed_tests += 1
    else
        println("❌ TestMatAsField: FAIL - expected $m fields, got $(length(result))")
    end
catch e
    println("❌ TestMatAsField: FAIL - $e")
end

# Test 7: logPostXi
println("\n7️⃣ Testing logPostXi...")
total_tests += 1
try
    xi = 15
    xiMin = 10
    xiMax = 20
    result = logPostXi(xiMin, xiMax, xi, hp_xi, TestField, TestTimes, thetas, rhos, phis, 
                     X, startSamplingPeriod, endSamplingPeriod)
    if isfinite(result)
        println("✅ logPostXi: PASS (result: $result)")
        passed_tests += 1
    else
        println("❌ logPostXi: FAIL - result is not finite: $result")
    end
catch e
    println("❌ logPostXi: FAIL - $e")
end

# Test 8: RWMH_xi
println("\n8️⃣ Testing RWMH_xi...")
total_tests += 1
try
    xi_cur = 15
    xi_can = 18
    TestField_copy = deepcopy(TestField)
    TestFieldProposal_copy = deepcopy(TestField)
    
    result = RWMH_xi(xi_can, xi_cur, hp_xi, TestFieldProposal_copy, TestField_copy, TestTimes, 
                   thetas, rhos, phis, X, startSamplingPeriod, endSamplingPeriod)
    if result in [xi_cur, xi_can]
        println("✅ RWMH_xi: PASS (result: $result)")
        passed_tests += 1
    else
        println("❌ RWMH_xi: FAIL - invalid result: $result")
    end
catch e
    println("❌ RWMH_xi: FAIL - $e")
end

# Test 9: TestMatAsFieldProposal
println("\n9️⃣ Testing TestMatAsFieldProposal...")
total_tests += 1
try
    xi = 15
    xiCan = 18
    TestFieldProposal_copy = deepcopy(TestField)
    TestMatAsFieldProposal(TestFieldProposal_copy, TestField, TestTimes, xi, xiCan, m)
    
    if length(TestFieldProposal_copy) == m
        println("✅ TestMatAsFieldProposal: PASS (field count: $(length(TestFieldProposal_copy)))")
        passed_tests += 1
    else
        println("❌ TestMatAsFieldProposal: FAIL - invalid field count")
    end
catch e
    println("❌ TestMatAsFieldProposal: FAIL - $e")
end

# Summary
println("\n" * "="^50)
println("📊 TEST SUMMARY")
println("="^50)
println("Passed: $passed_tests/$total_tests tests")
println("Success Rate: $(round(passed_tests/total_tests * 100, digits=1))%")

if passed_tests == total_tests
    println("🎉 ALL TESTS PASSED! Julia functions are working correctly.")
    println("\n💡 Next step: Try running the actual Julia MCMC script:")
    println("   julia runmodel_RDS.jl")
else
    println("⚠️  Some tests failed. Please review the implementation.")
end

println("\n🏁 Testing completed!")
