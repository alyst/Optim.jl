immutable NNLSObjective{T<:FloatingPoint} <: DifferentiableFunction
    A::AbstractMatrix{T}
    b::AbstractVector{T}
end

discrepancy{T<:FloatingPoint}(nnls::NNLSObjective{T}, x::AbstractVector{T}) = nnls.A*x - nnls.b

evalf{T<:FloatingPoint}(nnls::NNLSObjective{T}, x::AbstractVector{T}) = sum(discrepancy(nnls, x).^2)/2
evalg!{T<:FloatingPoint}(nnls::NNLSObjective{T}, x::AbstractVector{T}, g::AbstractVector{T}) = At_mul_B!(g, nnls.A, discrepancy(nnls, x))
function evalfg!{T<:FloatingPoint}(nnls::NNLSObjective{T}, x::AbstractVector{T}, g::AbstractVector{T})
    d = discrepancy(nnls, x)
    At_mul_B!(g, nnls.A, d)
    return sum(d.^2)/2
end

function nnls(A::AbstractMatrix, b::AbstractVector)
    # Set up the preconditioner as the inverse diagonal of the Hessian
    a = sum(A.^2, 1)
    # Create the initial guess (an interior point)
    T = promote_type(eltype(A), eltype(b))
    x = fill(one(T), size(A, 2))
    # Set up constraints
    l = zeros(eltype(x), length(x))
    u = fill(convert(eltype(x), Inf), length(x))
    # Perform the optimization
    fminbox(NNLSObjective{T}(A, b), x, l, u, precondprep=(P, x, l, u, mu)->precondprepnnls(P, x, mu, a))
end

function precondprepnnls(P, x, mu, a)
    for i = 1:length(x)
        P[i] = 1/(mu/x[i]^2 + a[i])
    end
end

export nnls
