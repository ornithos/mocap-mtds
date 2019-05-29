module model
using StatsBase, LinearAlgebra, MultivariateStats
using Flux
using AxUtil     # (Math: cayley /skew-symm stuff)
using ArgCheck
include("./util.jl")
const MyStandardScalar = mocaputil.MyStandardScalar
const invert = mocaputil.invert

#==============================================================================
    ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ model utilities ⋅⋅⋅⋅⋅⋅⋅ ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
 =============================================================================#

function zero_grad!(P)
    for x in P
        x.grad .= 0
    end
end

#==============================================================================
    ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ LDS definition ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
 =============================================================================#

abstract type MyLDS end

mutable struct MyLDS_ng{T} <: MyLDS
    a::AbstractVector{T}
    B::AbstractMatrix{T}
    b::AbstractVector{T}
    C::AbstractMatrix{T}
    D::AbstractMatrix{T}
    d::AbstractVector{T}
    h::AbstractVector{T}
end

mutable struct MyLDS_g{T} <: MyLDS
    a::TrackedVector{T}
    B::TrackedMatrix{T}
    b::TrackedVector{T}
    C::TrackedMatrix{T}
    D::TrackedMatrix{T}
    d::TrackedVector{T}
    h::TrackedVector{T}
end

Base.eltype(s::MyLDS_g{T}) where T <: Real = T
Base.eltype(s::MyLDS_ng{T}) where T <: Real = T
Base.size(s::MyLDS) = (size(s.B, 1), size(s.C, 1), size(s.D, 2))
Base.size(s::MyLDS, d)::Int = (size(s.B, 1), size(s.C, 1), size(s.D, 2))[d]

Flux.mapleaves(f::Function, s::MyLDS) = typeof(s)(f(s.a), f(s.B), f(s.b), f(s.C), f(s.D), f(s.d), f(s.h))
Base.copy(s::MyLDS) = Flux.mapleaves(deepcopy, s)

has_grad(s::MyLDS_g) = true
has_grad(s::MyLDS_ng) = false

function make_grad(s::MyLDS_ng)
    f = Flux.param
    MyLDS_g{eltype(s)}(f(s.a), f(s.B), f(s.b), f(s.C), f(s.D), f(s.d), f(s.h))
end

function make_nograd(s::MyLDS_g)
    f = Tracker.data
    MyLDS_ng{eltype(s)}(f(s.a), f(s.B), f(s.b), f(s.C), f(s.D), f(s.d), f(s.h))
end

pars(lds::MyLDS_g) = Flux.params(lds.a, lds.B, lds.b, lds.C, lds.D, lds.d)
ldsparvalues(s::MyLDS) = map(Tracker.data, [s.a, s.B, s.b, s.C, s.D, s.d])

#==============================================================================
    ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ LDS constructor / initialiser ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
 =============================================================================#

 """
     init_LDS(dest_T, A, B, b, C, D, d)

Create a MyLDS object: a model defined by the following equations:

    x_t = A x_{t-1} + B u_t + b
    y_t = C x_t + D u_t + d

for t = 1, ...., size(U, 2). Note that while an LDS would at least have a noise
term in the observation space (second eqn), this is often more conveniently
expressed outside of this object, and is not included here.

The MyLDS has a very specific form of A which removes a significant source of
rotational degeneracy from the latent space, and also permits easy enforcement
of stability. In order to achieve this, the constructor will automatically
transform (rotate/reflect) the relevant parameters, which should retain the
mapping between U and Y, but will result in different parameter values.

Note also because stability is enforced via a sigmoidal function (tanh),
singular values close to the boundaries of [0,1] will be projected slightly
inside (specifically to 1e-3 inside either way).
"""
function init_LDS(dest_T::DataType, A::AbstractMatrix{T}, B::AbstractMatrix{T}, b::AbstractVector{T},
        C::AbstractMatrix{T}, D::AbstractMatrix{T}, d::AbstractVector{T}) where T <: AbstractFloat
    Asvd = svd(A)

    d_state = size(A, 1)

    # obtaining the chosen transformation of A, and splitting into s.v.s
    # cayley factors:
    Q       = Asvd.V'*Asvd.U
    num_neg = sum(isapprox.(real.(eigvals(Q)), -1))
    J_neg   = vcat(-ones(dest_T, num_neg), ones(dest_T, d_state-num_neg))
    svals   = J_neg .* atanh.(clamp.(Asvd.S, 1e-3, 1-1e-3))
    cayley_init = AxUtil.Math.inverse_cayley_orthog(f64(J_neg .* Q))*10 # *10 ∵ rescales later
    a   = vcat(svals, cayley_init)

    # rotating the other parameters
    B   = Asvd.U' * B
    b   = Asvd.U' * b
    C   = C * Asvd.U

    return MyLDS_ng{dest_T}(a, B, b, C, D, d, zeros(dest_T, d_state))
end

function init_LDS(A::AbstractMatrix{T}, B::AbstractMatrix{T}, b::AbstractVector{T},
        C::AbstractMatrix{T}, D::AbstractMatrix{T}, d::AbstractVector{T}) where T <: AbstractFloat
    init_LDS(T, A, B, b, C, D, d)
end

function _tikhonov_mrdivide(A, B, λ)
    a1, a2 = size(A)
    b1, b2 = size(B)
    Ã = hcat(A, zeros(eltype(A), a1, b1))
    B̃ = hcat(B, Matrix(I, b1, b1)*√(λ))
    Ã / B̃
end

function _tikhonov_mldivide(A, B, λ)
    a1, a2 = size(A)
    b1, b2 = size(B)
    Ã = vcat(A, Matrix(I, a2, a2)*√(λ))
    B̃ = vcat(B, zeros(eltype(B), a2, b2))
    Ã \ B̃
end


function init_LDS_spectral(Y::AbstractMatrix{T}, U::AbstractMatrix{T}, d_state::Int;
    max_iter::Int = 4, return_hist::Bool=false, λ::T=T(1e-6), t_ahead::Int=1) where T <: AbstractFloat

    @argcheck size(Y,2) == size(U,2)
    cT(x) = convert(Array{T}, x)
    rmse(Δ) = sqrt(mean(x->x^2, Δ))

    N = size(Y, 2) + 1 - t_ahead
    d_out, d_in = size(Y, 1), size(U, 1)

    Yhankel = reduce(vcat, [Y[:,i:N+i-1] for i in 1:t_ahead])  # = Y by default.
    pc_all = fit(PPCA, Yhankel)
    Xhat = transform(pc_all, Yhankel)[1:d_state,:];

    # define hankel averaging operator for parameters
    unhankel(x::AbstractMatrix) = mean([x[(i-1)*d_out+1:i*d_out,:] for i in 1:t_ahead])
    unhankel(x::AbstractVector) = mean([x[(i-1)*d_out+1:i*d_out] for i in 1:t_ahead])

    # iterates
    rmse_history = ones(T, max_iter) * Inf
    lds_history = Vector{Any}(undef, max_iter)

    # Initialise emission
    C = unhankel(projection(pc_all)[:,1:d_state])
    D = zeros(T, N, d_in)
    d = unhankel(mean(pc_all))

    # Begin quasi-coordinate descent
    for i = 1:max_iter
        # 1-step cood descent initialised from "cheat": proj. of *current* y_ts
        Xhat = hcat(zeros(T, d_state, 1), Xhat)   # X starts from zero (i.e index 2 => x_1 etc.)
        ABb = _tikhonov_mrdivide(Xhat[:, 2:N+1], vcat(Xhat[:, 1:N], U[:,1:N], ones(T, N,1)'), λ)
        # ABb = Xhat[:, 2:N+1] / vcat(Xhat[:, 1:N], U[:,1:N], ones(T, N,1)')

        A = ABb[:, 1:d_state]
        Aeig = eigen(A)

        # poor man's stable projection
        if any(abs.(Aeig.values) .> 1)
            ub = 1.0
            for i in 1:10
                ν = [v / max(ub, abs(v) + sqrt(eps(T))) for v ∈ Aeig.values]
                A = real(Aeig.vectors * Diagonal(ν) * Aeig.vectors') |> cT

                Aeig = eigen(A)
                if all(abs.(Aeig.values) .<= 1)
                    break
                elseif i == 10
                    error("Unable to stabilise A matrix (if problematic, implement proper optim).")
                end
                ub *= 0.97
            end

        end

        # rest of dynamics
        B = ABb[:,d_state+1:end-1] |> cT
        b = ABb[:,end] |> cT

        # state rollout
        Xhat = Matrix{T}(undef, d_state, N+1);
        Xhat[:,1] = zeros(T, d_state)
        # remember here that Xhat is zero-based and U is 1-based.
        for i in 1:N
            @views Xhat[:,i+1] = A*Xhat[:,i] + B*U[:,i] + b
        end

        # regression of emission pars to current latent Xs
        postmultA(x, j) = j == 1 ? x : vcat(x, postmultA(x * A, j-1))
        # CDd = Y / [Xhat[:, 2:N+1]; U; ones(1, N)]
        CDd = _tikhonov_mrdivide(Yhankel, [Xhat[:, 2:N+1]; U[:,1:N]; ones(1, N)], λ)

        # # The "correct" thing to do (closer to SSID), but its iterates are poor in this CD scheme.
        # obsmat = reduce(hcat, [obsmat[(i-1)*64+1:i*64,:] for i in 1:t_ahead])
        # Â = _tikhonov_mldivide(CDd[65:end,1:20], CDd[1:end-64,1:20], 1f-5)
        # postmultÂT(x, j) = j == 1 ? x : vcat(x, postmultÂT(x * Â', j-1))
        # Ĉ = _tikhonov_mrdivide(obsmat, Matrix(postmultÂT(Matrix(I, d_state, d_state), t_ahead)'), 1f-4);

        C = CDd[1:d_out, 1:d_state]
        D = unhankel(CDd[:, d_state+1:end-1])
        d = unhankel(CDd[:, end])
        obs_mat = postmultA(C, t_ahead)
        rmse_history[i] = rmse(Yhankel - obs_mat * Xhat[:, 2:N+1] - repeat(D, t_ahead) * U[:,1:N] .- repeat(d, t_ahead))

        A, B, b, C, D, d = A |> cT, B |>cT, b |>cT, C |>cT, D |>cT, d |>cT

        lds_history[i] = init_LDS(T, A, B, b, C, D, d)

        if i < max_iter
            input_offset = D*U .+ d
            inputhankel = reduce(vcat, [input_offset[:,i:N+i-1] for i in 1:t_ahead])
            Xhat = _tikhonov_mldivide(obs_mat, Yhankel - inputhankel, λ)
            # Xhat = C \ (Y[:, 1:end] - D*U[:,1:end] .- d)
        end
    end

    clds = lds_history[argmin(rmse_history)]
    # ** DUE TO MAX/MIN CLAMPING OF S.V.s of A, (C, D, d)s may be (very!) suboptimal
    fit_optimal_obs_params(clds, Y, U; λ=λ, t_ahead=t_ahead)

    return_hist ? (clds, rmse_history) : clds
end


function Astable(ψ, d)
    n_skew = Int(d*(d-1)/2)
    x_S, x_V = ψ[1:d], ψ[d+1:d+n_skew]
    V = AxUtil.Math.cayley_orthog(x_V/10, d)
    S = AxUtil.Flux.diag0(tanh.(x_S))
    return S * V
end

Astable(s::MyLDS) = Astable(s.a, size(s, 1))


function fit_optimal_obs_params(s::MyLDS_g{T}, Y::AbstractMatrix{T},
    U::AbstractMatrix{T}; λ::T=T(1e-6), t_ahead::Int=1) where T <: AbstractFloat
    fit_optimal_obs_params(make_nograd(s), Y, U; t_ahead=t_ahead)
end

function fit_optimal_obs_params(s::MyLDS_ng{T}, Y::AbstractMatrix{T},
    U::AbstractMatrix{T}; λ::T=T(1e-6), t_ahead::Int=1) where T <: AbstractFloat
    @argcheck size(U, 2) == size(Y, 2)
    d_out = size(Y, 1)
    N = size(Y, 2) + 1 - t_ahead
    d_state = size(s, 1)

    Yhankel = reduce(vcat, [Y[:,i:N+i-1] for i in 1:t_ahead])  # = Y by default.
    unhankel(x::AbstractMatrix) = mean([x[(i-1)*d_out+1:i*d_out,:] for i in 1:t_ahead])
    unhankel(x::AbstractVector) = mean([x[(i-1)*d_out+1:i*d_out] for i in 1:t_ahead])

    CDd = _tikhonov_mrdivide(Yhankel, [state_rollout(s, U[:,1:N]); U[:,1:N]; ones(1, N)], λ)
    # CDd = Y / [state_rollout(s, U); U; ones(1, N)]

    s.C .= unhankel(CDd[:, 1:d_state])
    s.D .= unhankel(CDd[:, d_state+1:end-1])
    s.d .= unhankel(CDd[:, end]);

    return s
end

#==============================================================================
    ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ LDS Forward Pass ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
 =============================================================================#


mutable struct LDSCell{T}
    A::TrackedArray{T}
    B::TrackedArray{T}
    b::TrackedArray{T}
end

# Operation
function (m::LDSCell)(h, u)
    A, B, b = m.A, m.B, m.b
    h = A * h + B * u + b
    return h, h
end


function state_rollout(lds::MyLDS_g{T}, U::AbstractMatrix{T}) where T <: AbstractFloat
    d = size(lds, 1)
    n = size(U, 2)
    A = Astable(lds)
    ldscell = LDSCell(A, lds.B, lds.b)
    ldscell = Flux.Recur(ldscell, lds.h, lds.h)
    return hcat([ldscell(U[:,i]) for i in 1:n]...)
end


function state_rollout(lds::MyLDS_ng{T}, U::AbstractMatrix{T}) where T <: AbstractFloat
    d = size(lds, 1)
    n = size(U, 2)
    A = Astable(lds)

    X = Matrix{T}(undef, d, n);
    X[:,1] = A*lds.h + lds.B*U[:, 1] + lds.b

    for i in 2:n
        @views X[:,i] = A*X[:,i-1] + lds.B*U[:, i] + lds.b
    end
    return X
end

(lds::MyLDS_ng)(U) = let X = state_rollout(lds, U); lds.C * X + lds.D * U .+ lds.d; end
(lds::MyLDS_g)(U)  = let X = state_rollout(lds, U); lds.C * X + lds.D * U .+ lds.d; end


function unsup_predict(lds::MyLDS_ng{T}, U::AbstractMatrix{T},
        YsTrue::AbstractMatrix{T}, standardize_Y::MyStandardScaler,
        standardize_U::MyStandardScaler, ix_unsup::Int=10) where T <: AbstractFloat

    n = size(U, 2)
    A = Astable(lds)

    X = Matrix{T}(undef, size(lds, 1), n);
    Y = Matrix{T}(undef, size(lds, 2), n)
    u = Vector{T}(undef, size(lds, 3))

    # transform Y -> U
    μ, σ = standardize_U.μ[61:121], standardize_U.σ[61:121]

    X[:,1] = A*lds.h + lds.B*U[:, 1] + lds.b
    Y[:,1] = lds.C * X[:,1] + lds.D * U[:,1] .+ lds.d
    for i in 2:n
        u = U[:,i]   # I found some unexpected behaviour when using views here.
        y = (i <= ix_unsup) ? YsTrue[:,i-1] : Y[:,i-1]   # initial state is wrong (0) => need to wash out.
        y_unnorm = invert(standardize_Y, reshape(y, 1, 64)) |> vec
        u[61:121] = (y_unnorm[4:64] - μ) ./ σ            # transform to u space
        @views X[:,i] = A*X[:,i-1] + lds.B*u + lds.b
        @views Y[:,i] = lds.C * X[:,i] + lds.D * u .+ lds.d
    end
    return Y
end


function kstep_predict(lds::MyLDS_ng{T}, U::AbstractMatrix{T},
        YsTrue::AbstractMatrix{T}, standardize_Y::MyStandardScaler,
        standardize_U::MyStandardScaler, ix_unsup::Int=10, k::Int=1) where T <: AbstractFloat

    n = size(U, 2)
    A = Astable(lds)

    X = Matrix{T}(undef, size(lds, 1), n+1);
    Y = Matrix{T}(undef, size(lds, 2), n+1-k)
    u = Vector{T}(undef, size(lds, 3))

    # transform Y -> U
    μ, σ = standardize_U.μ[61:121], standardize_U.σ[61:121]

end   # module end
