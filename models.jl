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

    X[:,1] = zeros(T, size(lds, 1))  # note that X[:,i] := x_{i-1}
    for i in 1:n+1-k
        u = U[:,i]   # I found some unexpected behaviour when using views here.
        @views X[:,i+1] = A*X[:,i-1+1] + lds.B*u + lds.b
        y = lds.C * X[:,i+1] + lds.D * u .+ lds.d
        x = X[:, i+1]  # implicit copy
        for j = 2:k
            ix = i+j-1    # i already incorporates (+1)
            u = U[:,ix]   # I found some unexpected behaviour when using views here.
            y_ = (ix <= ix_unsup) ? YsTrue[:,ix-1] : y   # initial state is wrong (0) => need to wash out.
            y_unnorm = invert(standardize_Y, reshape(y_, 1, 64)) |> vec
            u[61:121] = (y_unnorm[4:64] - μ) ./ σ            # transform to u space
            copy!(x, A*x + lds.B*u + lds.b)
            y = lds.C * x + lds.D * u .+ lds.d   # no copy! o.w. cannot resize array with shared data (see ternary expr eval=> y)
        end
        Y[:, i] = y
    end
    return Y
end



# #==============================================================================
#     ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ Low Rank LDS definition ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
#  =============================================================================#
#
# abstract type LRLDS end
#
#
# mutable struct LRLDS_ng{T} <: LRLDS
#     a::AbstractVector{T}
#     B1::AbstractMatrix{T}
#     B2::AbstractMatrix{T}
#     b::AbstractVector{T}
#     C::AbstractMatrix{T}
#     D1::AbstractMatrix{T}
#     D2::AbstractMatrix{T}
#     d::AbstractVector{T}
#     h::AbstractVector{T}
# end
#
# mutable struct LRLDS_g{T} <: LRLDS
#     a::TrackedVector{T}
#     B1::TrackedMatrix{T}
#     B2::TrackedMatrix{T}
#     b::TrackedVector{T}
#     C::TrackedMatrix{T}
#     D1::TrackedMatrix{T}
#     D2::TrackedMatrix{T}
#     d::TrackedVector{T}
#     h::TrackedVector{T}
# end
#
# Base.eltype(s::LRLDS_g{T}) where T <: Real = T
# Base.eltype(s::LRLDS_ng{T}) where T <: Real = T
# Base.size(s::LRLDS) = (size(s.B, 1), size(s.C, 1), size(s.D, 2))
# Base.size(s::LRLDS, d)::Int = (size(s.B, 1), size(s.C, 1), size(s.D, 2))[d]



#==============================================================================
    ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ MTLDS definition ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
 =============================================================================#


abstract type MTLDS end

mutable struct MTLDS_ng{T, F}  <: MTLDS where {T <: Real, F <: Chain}
    nn::F
    a::AbstractVector{T}
    B::AbstractMatrix{T}
    b::AbstractVector{T}
    C::AbstractMatrix{T}
    D::AbstractMatrix{T}
    d::AbstractVector{T}
    h::AbstractVector{T}
    logσ::AbstractVector{T}
    function MTLDS_ng(nn,a,B,b,C,D,d,h,logσ)
        T = eltype(Tracker.data(a))
        @argcheck T ∈ [Float32, Float64]
        cT = (T==Float32) ? f32 : f64;
        nn = cT(nn)
        new{T,typeof(nn)}(nn,a,B,b,C,D,d,h,logσ)
    end
end

mutable struct MTLDS_g{T <: Real, F <: Chain} <: MTLDS
    nn::F
    a::TrackedVector{T}
    B::TrackedMatrix{T}
    b::TrackedVector{T}
    C::TrackedMatrix{T}
    D::TrackedMatrix{T}
    d::TrackedVector{T}
    h::TrackedVector{T}
    logσ::TrackedVector{T}
    function MTLDS_g(nn,a,B,b,C,D,d,h,logσ)
        T = eltype(Tracker.data(a))
        @argcheck T ∈ [Float32, Float64]
        cT = (T==Float32) ? f32 : f64;
        nn = cT(nn)
        new{T,typeof(nn)}(nn,a,B,b,C,D,d,h,logσ)
    end
end

Base.eltype(s::MTLDS_g{T, F}) where {T <: Real, F <: Chain} = T
Base.eltype(s::MTLDS_ng{T, F}) where {T <: Real, F <: Chain} = T
Base.size(s::MTLDS) where {T <: Int} = (size(s.B, 1), size(s.C, 1), size(s.D, 2))
Base.size(s::MTLDS, d)::Int = if d==1; size(s.B, 1); elseif d==2; size(s.C, 1); elseif d==3; size(s.D, 2); end

Flux.mapleaves(f::Function, s::MTLDS) = typeof(s)(mapleaves(f, s.nn), f(s.a), f(s.B), f(s.b),
    f(s.C), f(s.D), f(s.d), f(s.h), f(s.logσ))

Base.copy(s::MTLDS) = Flux.mapleaves(deepcopy, s)

has_grad(s::MTLDS_g) = true
has_grad(s::MTLDS_ng) = false

Flux.param(f::Function) = f  # need this to permit mapleaves of Flux.param below.
function make_grad(s::MTLDS_ng)
    f = Flux.param
    nn = mapleaves(f, s.nn)
    MTLDS_g(nn, f(s.a), f(s.B), f(s.b), f(s.C), f(s.D), f(s.d), f(s.h), f(s.logσ))
end

function make_nograd(s::MTLDS_g)
    f = Tracker.data
    nn = mapleaves(f, s.nn)
    MTLDS_ng(nn, f(s.a), f(s.B), f(s.b), f(s.C), f(s.D), f(s.d), f(s.h), f(s.logσ))
end

pars(s::MTLDS_g) = Flux.params(s.nn, s.logσ)
allpars(s::MTLDS_g) = Flux.params(s.nn, s.a, s.B, s.b, s.C, s.D, s.d, s.logσ)
ldsparvalues(s::MTLDS) = map(Tracker.data, [s.a, s.B, s.b, s.C, s.D, s.d, s.logσ])

zero_grad!(s::MTLDS_g) = zero_grad!(pars(s))

"""
    make_lds(s::MTLDS, z::AbstractVector)

Create an LDS instance from a multitask LDS given a latent hierarchical variable `z` .
Notice that the impact of `ψ = nn(z)` impacts *affinely* with the initialised
parameters a, B, b, C, D, d, logσ.
"""
function make_lds(s::Union{MTLDS_g{T,F}, MTLDS_ng{T,F}},
        z::Union{AbstractVector{T}, TrackedVector{T}}) where {T <: Real, F <: Chain}
    ψ = s.nn(z)
    return _make_lds_psi(s, ψ)
end

function _make_lds_psi(s::Union{MTLDS_g{T,F}, MTLDS_ng{T,F}},
        ψ::Union{AbstractVector{T}, TrackedVector{T}}) where {T <: Real, F <: Chain}
    d_state, d_out, d_in = size(s)
    ldsdims = _partition_ldspars_dims(d_state, d_out, d_in, length(ψ))
    a, B, b, C, D, d = partition_ldspars(ψ, ldsdims, d_state, d_out, d_in)
    h = zeros(T, d_state)
    (ldstype, state) = has_grad(s) ? (MyLDS_g{T}, Flux.param(h)) : (MyLDS_ng{T}, h)
    return ldstype(a + s.a, B + s.B, b + s.b, C + s.C, D + s.D, d + s.d, state)
end

function mtldsg_from_lds(s::MyLDS_ng{T}, nn::Chain, logσ::Vector=repeat([exp(1)], size(s,2))) where T
    Tnn = eltype(Tracker.data(nn[1].W))
    @assert (Tnn == T) "Ambiguous type. LDS is type $T, but nn is type $Tnn."
    f = Flux.param
    MTLDS_g(nn, f(s.a), f(s.B), f(s.b), f(s.C), f(s.D), f(s.d), f(s.h), f(T.(logσ)))
end

#==============================================================================
    ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ MTLDS utilities  ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
 =============================================================================#

"""
    get_pars(s::MyLDS)
    get_pars(s::MTLDS)

Extract all parameters from a given MyLDS/MTLDS (whether tracked or otherwise)
into a single parameter vector. This is useful for saving the parameter state
(for instance).

See `set_pars!` to load a given parameter state into an MyLDS/MTLDS object.
"""
function get_pars(s::MyLDS)
    ldspars = vcat(map(vec, map(Tracker.data, [s.a, s.B, s.b, s.C, s.D, s.d]))...)
    return ldspars
end

function get_pars(s::MTLDS)
    nnweights = Tracker.data.(Flux.params(s.nn))
    nnweights = vcat(map(vec, nnweights)...)
    ldspars = vcat(map(vec, map(Tracker.data, [s.a, s.B, s.b, s.C, s.D, s.d, s.logσ]))...)
    return vcat(nnweights, ldspars)
end

"""
    set_pars(s::MyLDS, p::AbstractVector)
    set_pars(s::MTLDS, p::AbstractVector)

Load a parameter vector (corresponding to all parameters in a given MyLDS/MTLDS
model -- whether tracked or otherwise) into an existing object, `s`. This is useful
to reset the state of the MTLDS to some previously extracted parameter vector.

See `get_pars` to extract a parameter vector from an MyLDS/MTLDS object.
"""
function set_pars!(s::MyLDS, p::Vector)
    lds_pars = ldsparvalues(s)
    lds_dims = map(length, lds_pars)
    @argcheck length(p) == sum(lds_dims)

    # begin setting
    csz = cumsum([0; lds_dims])
    for (nn, θ) in enumerate(lds_pars)
        θ .= reshape(p[(csz[nn]+1):csz[nn+1]], size(θ)...)
    end
end

function set_pars!(s::MTLDS, p::Vector)
    lds_pars = ldsparvalues(s)
    nnweights = Tracker.data.(Flux.params(s.nn))
    sz = map(size, nnweights)
    csz = cumsum([0; map(prod, sz)])
    lds_dims = map(length, lds_pars)
    total_d = sum(lds_dims) +  csz[end]
    @argcheck length(p) == total_d

    # begin setting
    new_nnweights = [reshape(p[(csz[nn]+1):csz[nn+1]], sz[nn]...) for nn in 1:length(sz)]
    Flux.loadparams!(s.nn, new_nnweights)
    csz = cumsum([csz[end]; lds_dims])
    for (nn, θ) in enumerate(lds_pars)
        θ .= reshape(p[(csz[nn]+1):csz[nn+1]], size(θ)...)
    end
end


"""
    partition_ldspars(s::MTLDS, p::AbstractVector)
    partition_ldspars(s::LDS, p::AbstractVector)
    partition_ldspars(p::AbstractVector, dims::AbstractVector{Int}, d_state::Int, d_out::Int, d_in::Int)

Partition an LDS parameter vector (**Note, not an MTLDS parameter vector**) into
the vector and matrices required for a `MyLDS` object. This was created
primarily as a useful utility to convert a parameter vector ψ created by the
hierarhical MTLDS prior into the relevant quantities for an LDS.

There is little overhead in using this function, and all ops are simple index /
copies that can be backprop-ed through without issue. However, in case of a large
number of calls, the parameter sizes can be cached (see `_partition_ldspars_dims`)
in source, and the long form version above may be used.
"""
function partition_ldspars(s::Union{MTLDS_g{T,F}, MTLDS_ng{T,F}}, p::Union{AbstractVector{T}, TrackedVector}) where {T,F}
    d_state, d_out, d_in = size(s)
    partition_ldspars(p, _partition_ldspars_dims(d_state, d_out, d_in, length(p)), d_state, d_out, d_in)
end

function partition_ldspars(s::Union{MyLDS_g{T}, MyLDS_ng{T}}, p::Union{AbstractVector{T}, TrackedVector{T}}) where T
    d_state, d_out, d_in = size(s)
    partition_ldspars(p, _partition_ldspars_dims(d_state, d_out, d_in, length(p)), d_state, d_out, d_in)
end

function partition_ldspars(p::Union{AbstractVector, TrackedVector}, dims::AbstractVector,
        d_state::Int, d_out::Int, d_in::Int)
    (p[1:dims[1]], reshape(p[dims[1]+1:dims[2]], d_state, d_in), p[dims[2]+1:dims[3]],
        reshape(p[dims[3]+1:dims[4]], d_out, d_state), reshape(p[dims[4]+1:dims[5]], d_out, d_in),
        p[dims[5]+1:dims[6]])
end

function _partition_ldspars_dims(d_state::Int, d_out::Int, d_in::Int, check_length::Int=-1)
    dims = Vector{Int}(undef, 7)
    dims[1] = Int((d_state*(d_state+1))/2)
    dims[2] = dims[1] + d_state * d_in
    dims[3] = dims[2] + d_state
    dims[4] = dims[3] + d_out * d_state
    dims[5] = dims[4] + d_out * d_in
    dims[6] = dims[5] + d_out
    check_length > -1 && check_length != dims[6] &&
        @warn "param vector given is different length ($check_length) to LDS params ($(dims[7]))"
    return dims
end
end   # module end
