# using Revise
using LinearAlgebra, Random
using StatsBase, Statistics
using Distributions, MultivariateStats   # Categorical, P(P)CA
using Quaternions    # For manipulating 3D Geometry
using MeshCat        # For web visualisation / animation
using AxUtil         # Cayley, skew matrices
using Flux           # Optimisation
using DSP            # convolution / low-pass (MA) filter

# small utils libraries
using ProgressMeter, Formatting, ArgCheck, Dates
using BSON


@argcheck 1 <= length(ARGS) <= 2
style_ix = parse(Int, ARGS[1])
testmode = (length(ARGS) > 1) && (ARGS[2] == "test")
testmode && println("Test mode active")
println("Style ix: $style_ix")

logname = format("log_hardeminf_{:d}", style_ix)

DIR_MOCAP_MTDS = "." 

# Data loading and transformation utils
include(joinpath(DIR_MOCAP_MTDS, "io.jl"))

# MeshCat skeleton visualisation tools
include(joinpath(DIR_MOCAP_MTDS, "mocap_viz.jl"))

# Data scaling utils
include(joinpath(DIR_MOCAP_MTDS, "util.jl"))

# Models: LDS
include(joinpath(DIR_MOCAP_MTDS, "models.jl"))

# Table visualisation
include(joinpath(DIR_MOCAP_MTDS, "pretty.jl"))


#================================================
         CUSTOM WIDELY USED FUNCTIONS
================================================#
function zero_grad!(P) 
    for x in P
        x.grad .= 0
    end
end

#https://discourse.julialang.org/t/redirect-stdout-and-stderr/13424/3
function redirect_to_files(dofunc, outfile, errfile)
    open(outfile, "a") do out
        open(errfile, "a") do err
            redirect_stdout(out) do
                redirect_stderr(err) do
                    dofunc()
                end
            end
        end
    end
end

const NoGradModels = Union{model.MyLDS_ng, model.ORNN_ng}
const _var_cache = IdDict()

mse(Δ::AbstractArray, scale=size(Δ, 1)) = mean(x->x^2, Δ)*scale

function mse(d::mocaputil.DataIterator, m::NoGradModels)
    obj = map(d) do (y, u, new_state)
        new_state && (m.h .= zeros(size(m, 1))) 
        u_ = (m isa model.ORNN_ng && length(m.inpnn)) > 0 ? vcat(u, m.inpnn(u)) : u
        mse(m(u_) - y)
    end
    m.h .= zeros(size(m, 1))
    return dot(obj, mocaputil.weights(d, as_pct=true))
end


mse(Ds::Vector{D}, m::NoGradModels) where {D <: Dict} = mse(mocaputil.DataIterator(Ds, 1000000), m)
mse(D::Dict, m::NoGradModels) = mse(m(D[:U]) - D[:Y])
mse(V::Tuple, m::NoGradModels) = mse(m(V[2]) - V[1])

# Calculate variance
function _calc_var!(cache::IdDict, d::mocaputil.DataIterator)
    Y = reduce(hcat, [y for (y, u, h) in d])
    _var_cache[d] = var(Y, dims=2)
end

function _calc_var!(cache::IdDict, d::Vector{D}) where {D <: Dict}
    Y = reduce(hcat, [dd[:Y] for dd in d])
    _var_cache[d] = var(Y, dims=2)
end

function Statistics.var(d::Union{mocaputil.DataIterator, Vector{D}}) where {D <: Dict}
    !haskey(_var_cache, d) && _calc_var!(_var_cache, d)
    return _var_cache[d]
end
Statistics.var(d::Dict) = var(d[:Y], dims=2)

# Standardised MSE
smse(Δ::AbstractArray, scale=size(Δ, 1)) = mse(Δ, scale) / sum(var(Δ, dims=2))

smse(d::mocaputil.DataIterator, m::NoGradModels) = mse(d, m) / sum(var(d))
smse(D::Dict, m::NoGradModels) = mse(m(D[:U]) - D[:Y]) / sum(var(D))
smse(Ds::Vector{D}, m::NoGradModels) where {D <: Dict} = mse(mocaputil.DataIterator(Ds, 1000000), m) / sum(var(Ds))
smse(D::Tuple, m::NoGradModels) = mse(D, m) / sum(var(D[1], dims=2))

rsmse(args...) = sqrt(smse(args...))

function mse(d::mocaputil.DataIterator, m::model.MTLDS_ng, z::AbstractArray)
    @argcheck size(z, 2) == length(d)
    obj = map(enumerate(d)) do (ii, (y, u, new_state))
        new_state && (m.h .= zeros(size(m, 1))) 
        cmodel = model.make_lds(m, z[:,ii], m.η_h)
        mse(cmodel(u) - y)
    end
    m.h .= zeros(size(m, 1))
    return dot(obj, mocaputil.weights(d, as_pct=true))
end
function mse(V::Tuple, m::model.MTLDS_ng, z::AbstractArray)
    cstate = copy(m.h)
    V[3] && (m.h .= zeros(size(m, 1))) 
    cmodel = model.make_lds(m, z, m.η_h)
    ŷ = cmodel(V[2]); m.h .= cstate;
    mse(ŷ - V[1])
end

function mse(d::mocaputil.DataIterator, m::model.ORNN_ng, z::AbstractArray, nn::Chain)
    @argcheck size(z, 2) == length(d)
    obj = map(enumerate(d)) do (ii, (y, u, new_state))
        new_state && (m.h .= zeros(size(m, 1))) 
        cmodel = model.make_rnn_psi(m, Tracker.data(nn(z[:,ii])), 1f0)
        mse(cmodel(u) - y)
    end
    m.h .= zeros(size(m, 1))
    return dot(obj, mocaputil.weights(d, as_pct=true))
end
function mse(V::Tuple, m::model.ORNN_ng, z::AbstractArray, nn::Chain)
    cstate = copy(m.h)
    V[3] && (m.h .= zeros(size(m, 1))) 
    cmodel = model.make_rnn_psi(m, Tracker.data(nn(z)), 1f0)
    ŷ = cmodel(V[2]); m.h .= cstate;
    mse(ŷ - V[1])
end

smse(d::Union{mocaputil.DataIterator}, m::model.MTLDS_ng, z::AbstractArray) = mse(d, m, z) / sum(var(d))
smse(d::Union{mocaputil.DataIterator}, m::model.ORNN_ng, z::AbstractArray, nn::Chain) = mse(d, m, z, nn) / sum(var(d))
smse(V::Tuple, m::model.MTLDS_ng, z::AbstractArray) = mse(V, m, z) / sum(var(V[1], dims=2))
smse(V::Tuple, m::model.ORNN_ng, z::AbstractArray, nn::Chain) = mse(V, m, z, nn) / sum(var(V[1], dims=2))
function smse(D::Array{T}, m::model.ORNN_ng, z::AbstractArray, nn::Chain) where T <: Tuple
    @argcheck size(z, 2) == length(d)
    mse_val, wgt = [mse(tup, m, z[:,ii]) for tup in D], [size(d[1], 2) for d in D]
    dot(mse_val, wgt / sum(wgt)) / sum(var(reduce(hcat, [d[1] for d in D]), dims=2))
end



#================================================
                  LOAD DATA
================================================#

# task descriptors
styles_lkp = BSON.load("styles_lkp")[:styles_lkp];
# Load in data
Usraw = BSON.load("edin_Xs_30fps.bson")[:Xs];
Ysraw = BSON.load("edin_Ys_30fps.bson")[:Ys];

Ysraw = [y[2:end,:] for y in Ysraw]
Usraw = [hcat(u[2:end,1:end-8], u[1:end-1,end-7:end]) for u in Usraw];

# Standardise inputs and outputs
standardize_Y = fit(mocaputil.MyStandardScaler, reduce(vcat, Ysraw),  1)
standardize_U = fit(mocaputil.MyStandardScaler, reduce(vcat, Usraw),  1)

Ys = [mocaputil.scale_transform(standardize_Y, y[2:end, :] ) for y in Ysraw];  # (1-step ahead of u)
Us = [mocaputil.scale_transform(standardize_U, u[1:end-1,:]) for u in Usraw];  # (1-step behind y)

@assert (let c=cor(Usraw[1][1:end-1, :], Ysraw[1][2:end, :], dims=1); 
        !isapprox(maximum(abs.(c[.!isnan.(c)])), 1.0); end) "some input features perfectly correlated"

expmtdata = mocapio.ExperimentData(Ysraw, [Matrix(y') for y in Ys], 
    [Matrix(u') for u in Us], styles_lkp);

# to invert: `mocaputil.invert(standardize_Y, y)`



#================================================
               SELECT EXPMT
================================================#

# Get training set for STL and pooled models.
d = d_state = 100;
batch_size = 64
min_size = 50
k = 3                 # dimension of manifold

trainPool, validPool, testPool = mocapio.get_data(expmtdata, style_ix, :split, :pooled);

# construct batch iterator
trainIter = mocaputil.DataIterator(trainPool, batch_size, min_size=min_size);

# style segment lookups
style_names = ["angry", "childlike", "depressed", "neutral", "old", "proud", "sexy", "strutting"];
segment_lkp = [length(mocaputil.DataIterator(mocapio.get_data(expmtdata, i, :train, :stl, split=[0.875,0.125]),
            batch_size, min_size=50)) for i in setdiff(1:8, style_ix)];
segment_lkp = [collect(i+1:j) for (i,j) in zip(vcat(0, cumsum(segment_lkp[1:end-1])), cumsum(segment_lkp))];

#================================================
                  LOAD MODEL
================================================#

fname = format("ornn{:d}_2L_{:d}_{:d}_pool_{:02d}{:02d}_v{:d}.bson", style_ix, d_state, 
    k, day(today()), month(today()), batch_size)
println(fname)
ornn_base, ornn_optim_ng, nn_ng, Zmap = let b=BSON.load(fname); 
    b[:m1], b[:m2], b[:nn], b[:Zmap]; end
ornn_base = model.make_grad(ornn_base)
ornn_optim = model.make_grad(ornn_optim_ng)
ornn_optim_ng = model.make_nograd(ornn_optim)
Zmap = Flux.param(Zmap);
nn = mapleaves(Tracker.param, nn_ng)
nn_ng = mapleaves(Tracker.data, nn)

orig_std = std(Zmap.data, dims=2)
Zmap = Zmap ./ orig_std;
nn_ng.layers[1].W .*= orig_std';


function get_lower_prediction(ornn_base, _Ub, _h)
    d_state, d_out, d_in = size(ornn_base)
    _Tb = size(_Ub, 2)
    rnn_ng = mapleaves(Tracker.data, RNN(d_in, d_state, ornn_base.σ))
    _h && Flux.reset!(rnn_ng)
    ornn_base_ng = model.make_nograd(ornn_base)
    model.build_rnn!(rnn_ng, ornn_base_ng)
    x̂ = reduce(hcat, [rnn_ng(_Ub[:,i]) for i in 1:_Tb])
    let m=ornn_base_ng; m.C*x̂ + m.D*_Ub .+ m.d; end
end

_newiters = map([trainPool, validPool, testPool]) do x
    x2 = deepcopy(x);
    for i in 1:length(x)
        x2[i][:U] = get_lower_prediction(ornn_base, x[i][:U], true)
    end
    xIter2 = mocaputil.DataIterator(x2, batch_size, min_size=min_size);
    (xIter2, collect(xIter2))
end |> x -> collect(Iterators.flatten(x))
trainIter2, trainIters2, validIter2, validIters2, testIter, testIters2 = _newiters;



#================================================
        DEFINE PROBABILITY DENSITIES
================================================#

function sse_batch(rnn::Flux.Recur, C::Matrix{T}, D::Matrix{T}, d_offset::Vector{T}, Y::Vector{Matrix{T}},
        U::Vector{Matrix{T}}, Λ½::Vector{T}=ones(T, size(Y, 1))) where T
    d = size(Y[1], 1)
    @argcheck length(U) == length(Y)
    @argcheck length(Λ½) == d
    n = length(Y)
    sses = Vector{T}(undef, n)
    for ix in 1:n
        y = Y[ix]
        u = U[ix]
        x̂ = reduce(hcat, [rnn(u[:,i]) for i in 1:size(u, 2)])
        ŷ = C*x̂ + D*u .+ d_offset
        sses[ix] = dot(sum(x->x^2, y - ŷ, dims=2), Λ½)
    end
    return sses
end


function sse_batch(ornn::model.ORNN_ng{T,F,C}, nn::Chain, Y::Vector{Matrix{T}}, U::Vector{Matrix{T}},
        Λ½::Vector{T}, ϵ::Matrix{T}) where {T, F, C}
    @argcheck nn.layers[1].W isa Array    # not tracked
    d_state, d_out, d_in = size(ornn)
    rnn_ng = mapleaves(Tracker.data, RNN(d_in, d_state, tanh))
    k, M = size(ϵ)
    sses = Matrix{T}(undef, length(Y), M)
    for i in 1:M
        c_ornn = model.make_rnn_psi(ornn_optim_ng, nn(ϵ[:,i]), 1f0)
        c_ornn.h .= 0
        model.build_rnn!(rnn_ng, c_ornn)
        sses[:,i] = sse_batch(rnn_ng, c_ornn.C, c_ornn.D, c_ornn.d, Y, U, Λ½)
        rnn_ng.state .= 0
    end
    return sses, ϵ
end

function sse_batch(ornn::model.ORNN_ng{T,F,C}, nn::Chain, Y::Vector{Matrix{T}}, U::Vector{Matrix{T}},
        Λ½::Vector{T}, M::Int) where {T, F, C}
    ϵ = let k = size(nn.layers[1].W, 2); convert(Array{T}, AxUtil.Random.sobol_gaussian(M, k)'); end
    sse_batch(ornn, nn, Y, U, Λ½, ϵ)
end

const prior_var = 1.0

function p_log_prior(Z::AbstractMatrix)
    n, d = size(Z)
    @assert d != 1
    ZZ = Z .* 1/sqrt(prior_var)
    exponent = -0.5*sum(ZZ.^2, dims=1)
    lognormconst = -(d/2)*log(2*pi*prior_var)
    return exponent .+ lognormconst
end

# function p_log_llh_test(X, Z::TrackedMatrix, tt::Int)
#     Y = decoder_nog_test(Z, tt)
#     Y = Tracker.collect(reduce(vcat, Y))
#     X = unsqueeze(vec(X), 1)
#     Δ = (Y .- X) ./ exp.(log_emission_std.data)
#     return -0.5*sum(Δ.^2, dims=2)
# end

function p_log_llh_test(Y::AbstractVector, U::AbstractVector, Z::AbstractMatrix{T}, Λ½::Vector{T}) where T
    d = length(Y)
    sse = sse_batch(ornn_optim_ng, nn_ng, Y, U, Λ½, Z)[1]
    logσ = -log.(Λ½)
    lognormconst = -0.5*d*log(2π) - d*mean(2*logσ/2)
    return -0.5*sse .+ lognormconst
end

function p_log_llh_test(Y::AbstractMatrix{T}, U::AbstractMatrix{T}, Z::AbstractMatrix{T}, Λ½::Vector{T}) where T
    p_log_llh_test([Y], [U], Z, Λ½)
end

function p_log_posterior_unnorm_beta_test(Z, Y, U, beta, Λ½::Vector)
    f_llh = p_log_llh_test(Y, U, Z, Λ½)
    f_prior = p_log_prior(Z)
    return f_prior + beta*f_llh, f_llh + f_prior
end



#================================================
         LOAD / DEFINE AMIS ROUTINES
================================================#

include("combinedsmcs.jl")
csmcs_opt = abcsmcs.csmcs_opt

function amis(f_log_beta, f_log_target, mu, cov, opts)
    @abcsmcs.unpack_csmcs_opt opts
    gris_opts = abcsmcs.smcs_opt(resample_every=gris_rsmp_every, sqrtdelta=gris_sqrtdelta,
        betas=ais_betas, test=diagnostics, grad_delta=gris_grad_delta, burnin=gris_burnin,
        prior_std=prior_std)
    S, logW = rand(MvNormal(mu, cov), 2000)', zeros(eltype(mu), 2000)
    S = convert(Matrix{eltype(mu)}, S)   # for whatever reason rand always returns Float64 :<
    S, W, pi, mu, cov = abcsmcs.AMIS(S, logW, amis_kcls, f_log_target, epochs=amis_epochs, 
        nodisp=!diagnostics, gmm_smps=amis_smp, IS_tilt=gmm_tilt)
    S, logW = abcsmcs.GMM_IS(gmm_smp, pi, mu, cov, f_log_target)

    return S, logW, [pi, mu, cov]
end


function smp_posterior(Y, U, Λ½, mu_prior, cov_prior; n_retry=3)
    
    p_lbeta(Z, beta) = p_log_posterior_unnorm_beta_test(Z, Y, U, beta, Λ½)
    p_lpost(Z) = p_log_posterior_unnorm_beta_test(Z, Y, U, 1, Λ½)[2]
    best = nothing
    for loop_try = 1:n_retry
        try
            S, logW, gmmcomp = amis(p_lbeta, p_lpost, mu_prior, cov_prior,
                    csmcs_opt(ais_betas=LinRange(0.01, 1., 15).^2, 
                    amis_epochs=3, diagnostics=false, amis_kcls=3, gmm_smp=3000,
                    amis_smp=3000, prior_std=3.0, gmm_tilt=2.))
            ess = abcsmcs.eff_ss(softmax(logW))
            
            if loop_try == 1 || ess > best[3]
                best = (S, logW, ess)
            end
            println(ess)
            if loop_try < n_retry && ess <= 100
                continue
            elseif ess <= 100
                @warn "low ESS"
            end
            return best
            
        catch e
            rethrow(e)
            if isa(e, InterruptException)
                rethrow(e)
            end
            @warn "failure"
            @warn e
        end
    end
end


#================================================
       SAMPLE POSTERIOR: VALIDATION SET
================================================#

ess_valid = zeros(length(validIters2))
all_S_valid = []
all_logW_valid = []

# redirect_to_files(logname * ".log", logname * ".err") do
# for i in 1:length(validIters2)
#     S, logW, ess = smp_posterior(validIters2[i][1], validIters2[i][2], ones(Float32, 64)/30, 
#         zeros(Float32, k), Matrix(I, k, k)*1f0; n_retry=4)
#     push!(all_S_valid, S)
#     push!(all_logW_valid, logW)
#     ess_valid[i] = ess
#     (i % 10 == 0) && printfmtln("Completed {:03d}", i);
# end
# end

fname = format("posterior_valid_ornn{:d}_2L_{:d}_{:d}_pool_{:02d}{:02d}_v{:d}.bson", style_ix, d_state, 
    k, day(today()), month(today()), batch_size)
# BSON.bson(fname, S=all_S_valid, logW=all_logW_valid, ess=ess_valid);
all_S_valid, all_logW_valid, ess_valid = let b=BSON.load(fname); b[:S], b[:logW], b[:ess]; end
println(fname)

err_matrixv = ones(length(validIters2), length(validIters2))*NaN
redirect_to_files(logname * ".log", logname * ".err") do
for (ii, (s, logw)) in enumerate(zip(all_S_valid, all_logW_valid))
    for jj in ii:length(validIters2)
        rn = rand(Categorical(softmax(logw[:])), 10)
        errs = [smse(validIters2[jj], ornn_optim_ng, vec(s[rn[mm],:]), nn_ng) for mm in 1:10]
        err_matrixv[ii,jj] = mean(errs)
    end
    println(ii)
end
end

fname = format("posterior_valid_ornn{:d}_2L_{:d}_{:d}_pool_{:02d}{:02d}_v{:d}.bson", style_ix, d_state, 
    k, day(today()), month(today()), batch_size)
BSON.bson(fname, S=all_S_valid, logW=all_logW_valid, ess=ess_valid, err_matrix=err_matrixv);
println(fname)


#================================================
       SAMPLE POSTERIOR: TEST SET
================================================#

ess_test = zeros(length(testIters2))
all_S_test = []
all_logW_test = []

redirect_to_files(logname * ".log", logname * ".err") do
for i in 1:length(testIters2)
    S, logW, ess = smp_posterior(testIters2[i][1], testIters2[i][2], ones(Float32, 64)/30, 
    zeros(Float32, k), Matrix(I, k, k)*1f0; n_retry=3)
    push!(all_S_test, S)
    push!(all_logW_test, logW)
    ess_test[i] = ess
    (i % 10 == 0) && printfmtln("Completed {:03d}", i);
end
end

fname = format("posterior_test_ornn{:d}_2L_{:d}_{:d}_pool_{:02d}{:02d}_v{:d}.bson", style_ix, d_state, 
    k, day(today()), month(today()), batch_size)
BSON.bson(fname, S=all_S_test, logW=all_logW_test, ess=ess_test);
println(fname)

err_matrix = ones(length(testIters2), length(testIters2))*NaN
redirect_to_files(logname * ".log", logname * ".err") do
for (ii, (s, logw)) in enumerate(zip(all_S_test, all_logW_test))
    for jj in ii:length(testIters2)
        rn = rand(Categorical(softmax(logw[:])), 10)
        errs = [smse(testIters2[jj], ornn_optim_ng, vec(s[rn[mm],:]), nn_ng) for mm in 1:10]
        err_matrix[ii,jj] = mean(errs)
    end
    println(ii)
end
end

fname = format("posterior_test_ornn{:d}_2L_{:d}_{:d}_pool_{:02d}{:02d}_v{:d}.bson", style_ix, d_state, 
    k, day(today()), month(today()), batch_size)
BSON.bson(fname, S=all_S_test, logW=all_logW_test, ess=ess_test, err_matrix=err_matrix);
println(fname)