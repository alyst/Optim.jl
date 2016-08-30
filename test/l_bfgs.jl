let
    for use_autodiff in (false, true)
        for (name, prob) in Optim.UnconstrainedProblems.examples
            if prob.isdifferentiable
                res = Optim.optimize(prob.f, prob.g!, prob.initial_x, LBFGS(), OptimizationOptions(autodiff = use_autodiff))
                @test norm(res.minimum - prob.solutions) < 1e-2
            end
        end
    end
end
