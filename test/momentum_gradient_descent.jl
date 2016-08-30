debug = false
let
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable && !(name in ("Polynomial", "Large Polynomial", "Himmelblau")) # it goes in a direction of ascent -> f_converged == true
                debug && @ show "** Name: $name"
                iterations = name == "Powell" ? 2000 : 1000
                res = Optim.optimize(prob.f, prob.g!, prob.initial_x, MomentumGradientDescent(),
                                     OptimizationOptions(autodiff = use_autodiff,
                                                         iterations = iterations,
                                                         show_trace = debug))
                debug && @show res.minimum
                debug && @show prob.solutions
                @test norm(res.minimum - prob.solutions, Inf) < 1e-2
            end
        end
    end
end
