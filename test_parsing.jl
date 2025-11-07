# Simple test to check if Julia files parse correctly
println("Testing Julia file parsing...")

try
    include("julia/iFFBS.jl")
    println("✓ iFFBS.jl parsed successfully!")
catch e
    println("✗ Error in iFFBS.jl: $e")
end

try
    include("julia/dimension_corrections.jl")
    println("✓ dimension_corrections.jl parsed successfully!")
catch e
    println("✗ Error in dimension_corrections.jl: $e")
end

println("Parsing test complete.")
