#
# Correctness Tests
#

using Optim
using Base.Test
using Compat

my_tests = [
    "types.jl",
    "api.jl",
    "type_stability.jl",
    "array.jl",
    "callbacks.jl",
    "optimize.jl",

    "constrained.jl",
    "precon.jl",
    "initial_convergence.jl",

    # tests for individual methods
    "bfgs.jl",
    "gradient_descent.jl",
    "momentum_gradient_descent.jl",
    "grid_search.jl",
    "l_bfgs.jl",
    "levenberg_marquardt.jl",
    "newton.jl",
    "newton_trust_region.jl",
    "cg.jl",
    "nelder_mead.jl",
    "simulated_annealing.jl",
    "particle_swarm.jl",
    "interpolating_line_search.jl",
    "golden_section.jl",
    "brent.jl",
]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end
