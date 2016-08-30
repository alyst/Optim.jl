function optimize(f::Function,
                  initial_x::Array;
                  method = NelderMead(),
                  x_tol::Real = 1e-32,
                  f_tol::Real = 1e-32,
                  g_tol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Integer = 1,
                  autodiff::Bool = false,
                  callback = nothing)
    options = OptimizationOptions(;
        x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
        iterations = iterations, store_trace = store_trace,
        show_trace = show_trace, extended_trace = extended_trace,
        callback = callback, show_every = show_every,
        autodiff = autodiff)
    optimize(f, initial_x, method, options)
end

function optimize(f::Function,
                  g!::Function,
                  initial_x::Array;
                  method = LBFGS(),
                  x_tol::Real = 1e-32,
                  f_tol::Real = 1e-32,
                  g_tol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Integer = 1,
                  callback = nothing)
    options = OptimizationOptions(;
        x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
        iterations = iterations, store_trace = store_trace,
        show_trace = show_trace, extended_trace = extended_trace,
        callback = callback, show_every = show_every)
    optimize(f, g!, initial_x, method, options)
end

function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Array;
                  method = Newton(),
                  x_tol::Real = 1e-32,
                  f_tol::Real = 1e-32,
                  g_tol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Integer = 1,
                  callback = nothing)
    options = OptimizationOptions(;
        x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
        iterations = iterations, store_trace = store_trace,
        show_trace = show_trace, extended_trace = extended_trace,
        callback = callback, show_every = show_every)
    optimize(f, g!, h!, initial_x, method, options)
end

function optimize(d::DifferentiableFunction,
                  initial_x::Array;
                  method = LBFGS(),
                  x_tol::Real = 1e-32,
                  f_tol::Real = 1e-32,
                  g_tol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Integer = 1,
                  callback = nothing)
    options = OptimizationOptions(;
        x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
        iterations = iterations, store_trace = store_trace,
        show_trace = show_trace, extended_trace = extended_trace,
        callback = callback, show_every = show_every)
    optimize(d, initial_x, method, options)
end

function optimize(d::TwiceDifferentiableFunction,
                  initial_x::Array;
                  method = Newton(),
                  x_tol::Real = 1e-32,
                  f_tol::Real = 1e-32,
                  g_tol::Real = 1e-8,
                  iterations::Integer = 1_000,
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false,
                  show_every::Integer = 1,
                  callback = nothing)
    options = OptimizationOptions(;
        x_tol = x_tol, f_tol = f_tol, g_tol = g_tol,
        iterations = iterations, store_trace = store_trace,
        show_trace = show_trace, extended_trace = extended_trace,
        callback = callback, show_every = show_every)
    optimize(d, initial_x, method, options)
end

const error_no_hessian_provided = "No gradient or Hessian was provided. Either provide a gradient and Hessian, set autodiff = true in the OptimizationOptions if applicable, or choose a solver that doesn't require a Hessian."

function optimize(f::Function,
                  g!::Function,
                  initial_x::Array,
                  method::Optimizer,
                  options::OptimizationOptions = OptimizationOptions())
    if issubtype(DifferentiableFunction, required_ftype(method))
        d = DifferentiableFunction(f, g!)
    elseif options.autodiff
        d = AutodiffFunction(f, g!)
    else
        error(error_no_hessian_provided)
    end
    optimize(d, initial_x, method, options)
end

function optimize(f::Function,
                  g!::Function,
                  h!::Function,
                  initial_x::Array,
                  method::Optimizer,
                  options::OptimizationOptions = OptimizationOptions())
    d = TwiceDifferentiableFunction(f, g!, h!)
    optimize(d, initial_x, method, options)
end

function optimize(f::Function,
                  initial_x::Array,
                  method::Optimizer,
                  options::OptimizationOptions = OptimizationOptions())
    if isa(Any, required_ftype(method))
        # function doesn't need gradient
        return optimize(f, initial_x, method, options)
    elseif options.autodiff
        return optimize(AutodiffFunction(f), initial_x, method, options)
    else
        error(error_no_hessian_provided)
    end
end

function optimize(f::DifferentiableFunction,
                  initial_x::Array,
                  method::Optimizer,
                  options::OptimizationOptions = OptimizationOptions())
    if issubtype(Any, required_ftype(method))
        # accepts any function, so we need to extract evalf(f)
        return optimize(evalf_func(f), initial_x, method, options)
    elseif isa(f, required_ftype(method))
        # should be handled by more method-specific optimize(), so something is not right
        error("optimize() method specific to $(method) and $f not found")
    elseif options.autodiff
        return optimize(AutodiffFunction(f), initial_x, method, options)
    else
        error(error_no_hessian_provided)
    end
end

function optimize{T <: AbstractFloat}(f::Function,
                                      lower::T,
                                      upper::T;
                                      method = Brent(),
                                      rel_tol::Real = sqrt(eps(T)),
                                      abs_tol::Real = eps(T),
                                      iterations::Integer = 1_000,
                                      store_trace::Bool = false,
                                      show_trace::Bool = false,
                                      callback = nothing,
                                      show_every = 1,
                                      extended_trace::Bool = false)
    show_every = show_every > 0 ? show_every: 1
    if extended_trace && callback == nothing
        show_trace = true
    end
    if show_trace
        @printf "Iter     Function value   Gradient norm \n"
    end
    optimize(f, Float64(lower), Float64(upper), method;
             rel_tol = rel_tol,
             abs_tol = abs_tol,
             iterations = iterations,
             store_trace = store_trace,
             show_trace = show_trace,
             show_every = show_every,
             callback = callback,
             extended_trace = extended_trace)
end

function optimize(f::Function,
                  lower::Real,
                  upper::Real;
                  kwargs...)
    optimize(f,
             Float64(lower),
             Float64(upper);
             kwargs...)
end
