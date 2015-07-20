immutable OptimizationState
    iteration::Int
    value::Float64
    gradnorm::Float64
    metadata::Dict
end

function OptimizationState(i::Integer, f::Real)
    OptimizationState(int(i), Float64(f), NaN, Dict())
end

function OptimizationState(i::Integer, f::Real, g::Real)
    OptimizationState(int(i), Float64(f), Float64(g), Dict())
end

immutable OptimizationTrace
    states::Vector{OptimizationState}
end

OptimizationTrace() = OptimizationTrace(Array(OptimizationState, 0))

abstract OptimizationResults

type MultivariateOptimizationResults{T,N} <: OptimizationResults
    method::ASCIIString
    initial_x::Array{T,N}
    minimum::Array{T,N}
    f_minimum::Float64
    iterations::Int
    iteration_converged::Bool
    x_converged::Bool
    xtol::Float64
    f_converged::Bool
    ftol::Float64
    gr_converged::Bool
    grtol::Float64
    trace::OptimizationTrace
    f_calls::Int
    g_calls::Int
end

type UnivariateOptimizationResults{T} <: OptimizationResults
    method::ASCIIString
    initial_lower::T
    initial_upper::T
    minimum::T
    f_minimum::Float64
    iterations::Int
    converged::Bool
    rel_tol::Float64
    abs_tol::Float64
    trace::OptimizationTrace
    f_calls::Int
end

# Interface for evaluating the value and the gradient of
# a differentiable function
abstract DifferentiableFunction

# default implementation of DifferentiableFunction interface
evalf(df::DifferentiableFunction, x) = df.f(x)
evalg!(df::DifferentiableFunction, x, grad) = df.g!(x, grad)
evalfg!(df::DifferentiableFunction, x, grad) = df.fg!(x, grad)

# Implementation of DifferentialFunction based on Function objects
immutable SimpleDifferentiableFunction <: DifferentiableFunction
    f::Function
    g!::Function
    fg!::Function
end

Base.convert(::Type{DifferentiableFunction}, f::Function, g!::Function, fg!::Function) = SimpleDifferentiableFunction(f, g!, fg!)

if VERSION >= v"0.4-"
# Callable wrapper of DifferentiableFunction
# that evaluates the function properties specified by WHAT parameter
immutable DifferentiableFunctionEval{WHAT, DF<:DifferentiableFunction}
    df::DF
end

# "function"-like object that evaluates f(x)
evalf_func{DF<:DifferentiableFunction}(df::DF) = DifferentiableFunctionEval{:F, DF}(df)
# "function"-like object that evaluates the gradient of f(x)
evalg!_func{DF<:DifferentiableFunction}(df::DF) = DifferentiableFunctionEval{:G!, DF}(df)
# "function"-like object that evaluates the value and gradient of f(x)
evalfg!_func{DF<:DifferentiableFunction}(df::DF) = DifferentiableFunctionEval{:FG!, DF}(df)

Base.call(dfevalf::DifferentiableFunctionEval{:F}, x) = evalf(dfevalf.df, x)
Base.call(dfevalf::DifferentiableFunctionEval{:G!}, x, grad) = evalfg!(dfevalf.df, x, grad)
Base.call(dfevalf::DifferentiableFunctionEval{:FG!}, x, grad) = evalfg!(dfevalf.df, x, grad)
end

# Interface for evaluating the value, gradient and Hessian of
# a twice differentiable function
abstract TwiceDifferentiableFunction <: DifferentiableFunction

# Implementation of TwoceDifferentialFunction based on Function objects
immutable SimpleTwiceDifferentiableFunction <: TwiceDifferentiableFunction
    f::Function
    g!::Function
    fg!::Function
    h!::Function
end

Base.convert(::Type{TwiceDifferentiableFunction}, f::Function, g!::Function, fg!::Function, h!::Function) = SimpleTwiceDifferentiableFunction(f, g!, fg!, h!)

evalh!(df::SimpleTwiceDifferentiableFunction, x, hessian) = df.h!(x, hessian)

if VERSION >= v"0.4-"
# Callable wrapper of TwiceDifferentiableFunction
# that evaluates the function properties specified by WHAT parameter
# (only :H as all other properties are already handled by DifferentiableFunctionEval)
immutable TwiceDifferentiableFunctionEval{WHAT, DF<:TwiceDifferentiableFunction}
    df::DF
end

evalh!_func{DF<:TwiceDifferentiableFunction}(df::DF) = TwiceDifferentiableFunctionEval{:H!, DF}(df)
Base.call(dfevalf::TwiceDifferentiableFunctionEval{:H!}, x, hessian) = evalh!(dfevalf.df, x, hessian)
end

function Base.show(io::IO, t::OptimizationState)
    @printf io "%6d   %14e   %14e\n" t.iteration t.value t.gradnorm
    if !isempty(t.metadata)
        for (key, value) in t.metadata
            @printf io " * %s: %s\n" key value
        end
    end
    return
end

Base.push!(t::OptimizationTrace, s::OptimizationState) = push!(t.states, s)

Base.getindex(t::OptimizationTrace, i::Integer) = getindex(t.states, i)

function Base.setindex!(t::OptimizationTrace,
                        s::OptimizationState,
                        i::Integer)
    setindex!(t.states, s, i)
end

function Base.show(io::IO, t::OptimizationTrace)
    @printf io "Iter     Function value   Gradient norm \n"
    @printf io "------   --------------   --------------\n"
    for state in t.states
        show(io, state)
    end
    return
end

function converged(r::MultivariateOptimizationResults)
    return r.x_converged || r.f_converged || r.gr_converged
end

function Base.show(io::IO, r::MultivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" r.method
    if length(join(r.initial_x, ",")) < 40
        @printf io " * Starting Point: [%s]\n" join(r.initial_x, ",")
    else
        @printf io " * Starting Point: [%s, ...]\n" join(r.initial_x[1:2], ",")
    end
    if length(join(r.minimum, ",")) < 40
        @printf io " * Minimum: [%s]\n" join(r.minimum, ",")
    else
        @printf io " * Minimum: [%s, ...]\n" join(r.minimum[1:2], ",")
    end
    @printf io " * Value of Function at Minimum: %f\n" r.f_minimum
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * |x - x'| < %.1e: %s\n" r.xtol r.x_converged
    @printf io "   * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" r.ftol r.f_converged
    @printf io "   * |g(x)| < %.1e: %s\n" r.grtol r.gr_converged
    @printf io "   * Exceeded Maximum Number of Iterations: %s\n" r.iteration_converged
    @printf io " * Objective Function Calls: %d\n" r.f_calls
    @printf io " * Gradient Call: %d" r.g_calls
    return
end

function converged(r::UnivariateOptimizationResults)
    return r.converged
end

function Base.show(io::IO, r::UnivariateOptimizationResults)
    @printf io "Results of Optimization Algorithm\n"
    @printf io " * Algorithm: %s\n" r.method
    @printf io " * Search Interval: [%f, %f]\n" r.initial_lower r.initial_upper
    @printf io " * Minimum: %f\n" r.minimum
    @printf io " * Value of Function at Minimum: %f\n" r.f_minimum
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: max(|x - x_upper|, |x - x_lower|) <= 2*(%.1e*|x|+%.1e): %s\n" r.rel_tol r.abs_tol r.converged
    @printf io " * Objective Function Calls: %d" r.f_calls
    return
end

function Base.append!(a::MultivariateOptimizationResults, b::MultivariateOptimizationResults)
    a.iterations += b.iterations
    a.minimum = b.minimum
    a.f_minimum = b.f_minimum
    a.iteration_converged = b.iteration_converged
    a.x_converged = b.x_converged
    a.f_converged = b.f_converged
    a.gr_converged = b.gr_converged
    append!(a.trace, b.trace)
    a.f_calls += b.f_calls
    a.g_calls += b.g_calls
end

# Using finite difference to approximate the gradient
immutable FiniteDifferenceDifferentiableFunction <: DifferentiableFunction
    f::Function
    kind::Symbol # :central :forward :backward
end

evalg!(f::FiniteDifferenceDifferentiableFunction, x::Array, grad::Array) = Calculus.finite_difference!(f.f, x, grad, f.kind)
function evalfg!(f::FiniteDifferenceDifferentiableFunction, x::Array, grad::Array)
    Calculus.finite_difference!(f.f, x, grad, f.kind)
    return f.f(x)
end

Base.convert(::Type{DifferentiableFunction}, f::Function, kind::Symbol=:central) = FiniteDifferenceDifferentiableFunction(f, kind)

immutable ComposeFG!DifferentiableFunction <: DifferentiableFunction
    f::Function
    g!::Function
end

Base.convert(::Type{DifferentiableFunction}, f::Function, g!::Function) = ComposeFG!DifferentiableFunction(f, g!)

function evalfg!(f::ComposeFG!DifferentiableFunction, x::Array, grad::Array)
    evalg!(f, x, grad)
    return evalf(f, x)
end

# Using finite difference to approximate the gradient and hessian
immutable FiniteDifferenceTwiceDifferentiableFunction <: TwiceDifferentiableFunction
    f::Function
    kind::Symbol # :central :forward :backward
end

evalg!(f::FiniteDifferenceTwiceDifferentiableFunction, x::Array, grad::Array) = Calculus.finite_difference!(f.f, x, grad, f.kind)
function evalfg!(f::FiniteDifferenceTwiceDifferentiableFunction, x::Array, grad::Array)
    Calculus.finite_difference!(f.f, x, grad, f.kind)
    return f.f(x)
end
evalh!(f::FiniteDifferenceTwiceDifferentiableFunction, x::Array, grad::Array) = Calculus.finite_difference_hessian!(f, x, grad)

immutable ComposeFG!TwiceDifferentiableFunction <: DifferentiableFunction
    f::Function
    g!::Function
    h!::Function
end

Base.convert(::Type{TwiceDifferentiableFunction}, f::Function, g!::Function, h!::Function) = ComposeFG!TwiceDifferentiableFunction(f, g!)

function evalfg!(f::ComposeFG!TwiceDifferentiableFunction, x::Array, grad::Array)
    evalg!(f, x, grad)
    return evalf(f, x)
end

# A cache for results from line search methods (to avoid recomputation)
type LineSearchResults{T}
    alpha::Vector{T}
    value::Vector{T}
    slope::Vector{T}
    nfailures::Int
end

LineSearchResults{T}(::Type{T}) = LineSearchResults(T[], T[], T[], 0)

Base.length(lsr::LineSearchResults) = length(lsr.alpha)

function Base.push!{T}(lsr::LineSearchResults{T}, a::T, v::T, d::T)
    push!(lsr.alpha, a)
    push!(lsr.value, v)
    push!(lsr.slope, d)
    return
end

function clear!(lsr::LineSearchResults)
    empty!(lsr.alpha)
    empty!(lsr.value)
    empty!(lsr.slope)
    return
    # nfailures is deliberately not set to 0
end
