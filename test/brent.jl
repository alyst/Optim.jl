let
    for (name, prob) in Optim.UnivariateProblems.examples
        results = optimize(prob.f, prob.bounds..., method = Brent())

        @test Optim.converged(results)
        @test norm(Optim.minimizer(results) - prob.minimizers) < 1e-7
    end
end
