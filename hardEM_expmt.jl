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


function mse(d::mocaputil.DataIterator, m::model.ORNN_ng, z::AbstractArray, nn::Chain)
    @argcheck size(z, 2) == length(d)
    obj = map(enumerate(d)) do (ii, (y, u, new_state))
        new_state && (m.h .= zeros(size(m, 1))) 
        cmodel = model.make_rnn_psi(m, Tracker.data(nn(z[:,ii])), 1f0)
        u_ = length(m.inpnn) > 0 ? vcat(u, m.inpnn(u)) : u
        mse(cmodel(u_) - y)
    end
    m.h .= zeros(size(m, 1))
    return dot(obj, mocaputil.weights(d, as_pct=true))
end

smse(d::mocaputil.DataIterator, m::model.MTLDS_ng, z::AbstractArray) = mse(d, m, z) / sum(var(d))
smse(d::mocaputil.DataIterator, m::model.ORNN_ng, z::AbstractArray, nn::Chain) = mse(d, m, z, nn) / sum(var(d))



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
d_in₊ = 30
use_inpnn = false
k = 3                 # dimension of manifold
d_nn = 200            # "complexity" of manifold
d_subspace = 30;      # dim of subspace (⊆ parameter space) containg the manifold


@argcheck use_inpnn == false "below code needs to be changed, esp refs to us(...) and inpnn."

trainPool, validPool, testPool = mocapio.get_data(expmtdata, style_ix, :split, :pooled);

# construct batch iterator
trainIter = mocaputil.DataIterator(trainPool, batch_size, min_size=min_size);

# style segment lookups
style_names = ["angry", "childlike", "depressed", "neutral", "old", "proud", "sexy", "strutting"];
segment_lkp = [length(mocaputil.DataIterator(mocapio.get_data(expmtdata, i, :train, :stl, split=[0.875,0.125]),
            batch_size, min_size=50)) for i in setdiff(1:8, style_ix)];
segment_lkp = [collect(i+1:j) for (i,j) in zip(vcat(0, cumsum(segment_lkp[1:end-1])), cumsum(segment_lkp))];


#================================================
               SETUP BASE RNN-1
================================================#

model_1_ixs = 1:28;

# base *LDS*
train_neutral = mocapio.get_data(expmtdata, 4, :train, :stl, concat=true, simplify=true);
train_neutral[:Y] = train_neutral[:Y][model_1_ixs, :]
clds_orig = model.init_LDS_spectral(train_neutral[:Y], train_neutral[:U], d_state, t_ahead=4);

# init a:
# extract cθ from a spectral LDS fit for ORNN initialisation (see Henaff et al. for block diag init motivation)
lds_evs = eigvals(model.Astable(clds_orig));
blkvals = vcat([sqrt((1-ct)/(1+ct)) for ct in real(lds_evs[2:2:end])]', 
                zeros(Float32, floor(Int, d_state/2))')[1:end-1]
a = AxUtil.Math.unmake_lt_strict(diagm(-1=>blkvals), d_state)
a = vcat(ones(Float32, 10)*atanh(0.5f0), ones(Float32, 10)*atanh(0.75f0), ones(Float32, d_state-20)*atanh(0.9f0), a);


_d_state, d_out, d_in = size(clds_orig)
@argcheck d_state == _d_state
_U = train_neutral[:U]
cN = size(_U, 2)


d_in_total = d_in + d_in₊
inpnn = Chain(Dense(d_in, 50), Dense(50, d_in₊))
inpnn_ng = mapleaves(Tracker.data, inpnn)
u_s(u, nn) = vcat(u, nn(u))
u_s(u) = u

# construct init RNN
rnn = RNN(d_in_total, d_state, tanh)
rnn = RNN(d_in, d_state, tanh)
rnn.cell.Wh.data .= model.Astable(a, d_state)

# initialise emissions
x̂ = reduce(hcat, let _rnn=mapleaves(Tracker.data, rnn); [_rnn(u_s(_U[:,i])) for i in 1:cN]; end)
CDd = model._tikhonov_mrdivide(train_neutral[:Y], [x̂; _U; ones(1, cN)], 1e-3);
C = param(CDd[:, 1:d_state]) |> f32
D = param(CDd[:, d_state+1:end-1]) |> f32
d_offset = param(CDd[:,end]) |> f32;

# initialise base model
ornn_base = model.ORNN_g(param(a), copy(rnn.cell.Wi), copy(rnn.cell.b), copy(rnn.cell.h),
                    copy(C), copy(D), copy(d_offset), tanh, Chain()); #inpnn);


#================================================
                    OPTIMISE
================================================#

opt = ADAM(1e-4)
pars = model.pars(ornn_base);   # includes inpnn if available

ω = ones(Float32, length(model_1_ixs))
ω[1:3]*=5;

function optimise_lower!(ornn_base, trainIter, opt, η, n_epochs, shuffle_examples, train_ixs, 
        ω = ones(Float32, length(train_ixs)))
    opt.eta = η
    nB = length(trainIter)
    W = mocaputil.weights(trainIter; as_pct=false) ./ trainIter.batch_size

    history = ones(n_epochs*nB) * NaN

    for ee in 1:n_epochs
        rnn = RNN(size(ornn_base,3), size(ornn_base,1), ornn_base.σ)

        if shuffle_examples
            mtl_ixs, trainData = mocaputil.indexed_shuffle(trainIter)
        else
            mtl_ixs, trainData = 1:length(trainIter), trainIter
        end
        for (ii, (Yb, Ub, h0)) in zip(mtl_ixs, trainData)
            h0 && Flux.reset!(rnn)
            Tb = size(Yb, 2)      # not constant

            model.build_rnn!(rnn, ornn_base)
            x̂ = reduce(hcat, [rnn(u_s(Ub[:,i])) for i in 1:Tb])  |> Tracker.collect
            ŷ = let m=ornn_base; m.C*x̂ + m.D*Ub .+ m.d; end                 # adapt C, D, d too.
            obj = mean(x->x^2, ω .* (Yb[train_ixs,:] - ŷ)) * 8^2 * W[ii]

            Tracker.back!(obj)
            history[(ee-1)*nB + ii] = obj.data

            if ii % 34 == 0
                obj = 1e-3*sum(abs, ornn_base.B)
                obj += 1e-3*sum(abs, ornn_base.C)
                obj += 1e-3*sum(abs, ornn_base.D)
                Tracker.back!(obj)

                for p in pars
                    Tracker.update!(opt, p, Tracker.grad(p))
                end
            end

            rnn.cell.h.data .= 0       # initial state is a param :/. Easier to reset here.
            Flux.truncate!(rnn);
        end
        printfmtln("{:02d}: {:.5f}", ee, sqrt(mean(history[(1:nB) .+ nB*(ee-1)]))); flush(stdout)

    end
    return history
end

logname = format("log_hardem_{:d}", style_ix)
redirect_to_files(logname * ".log", logname * ".err") do
    println("Style ix: $style_ix")
    history1 = optimise_lower!(ornn_base, trainIter, opt, 1e-3, testmode ? 1 : 200, true, model_1_ixs, ω);
    history2 = optimise_lower!(ornn_base, trainIter, opt, 5e-4, testmode ? 1 : 200, true, model_1_ixs, ω);
    history3 = optimise_lower!(ornn_base, trainIter, opt, 2e-4, testmode ? 1 : 200, true, model_1_ixs, ω);
    history4 = optimise_lower!(ornn_base, trainIter, opt, 1e-4, testmode ? 1 : 250, true, model_1_ixs, ω);
end

trainPool_1 = deepcopy(trainPool);
for i in 1:length(trainPool)
    trainPool_1[i][:Y] = trainPool[i][:Y][model_1_ixs, :]
end
trainIter_1 = mocaputil.DataIterator(trainPool_1, batch_size, min_size=min_size);
redirect_to_files(logname * ".log", logname * ".err") do
    println(smse(trainIter_1, model.make_nograd(ornn_base)))
    fname = format("ornn{:d}_lowermodel_{:d}_pool_{:02d}{:02d}_v{:d}.bson", style_ix, d_state, 
                day(today()), month(today()), batch_size)
    BSON.bson(fname, m1=model.make_nograd(ornn_base));
    println(fname)
end

#================================================
         PREDICT INPUTS FOR NEXT LAYER
================================================#

trainIters = collect(trainIter);
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

trainPool2 = deepcopy(trainPool);
for i in 1:length(trainPool)
    trainPool2[i][:U] = get_lower_prediction(ornn_base, trainPool[i][:U], true)
end
trainIter2 = mocaputil.DataIterator(trainPool2, batch_size, min_size=min_size);

#================================================
               SETUP Multi-Task RNN-2
================================================#
ornn_optim = copy(ornn_base);   # copy to avoid over-writing initialisation
ornn_optim.C = Tracker.param(Flux.glorot_normal(64, size(ornn_optim,1)));
ornn_optim.B = Tracker.param(Flux.glorot_normal(size(ornn_optim, 1),28));
ornn_optim.D = Tracker.param(Flux.glorot_normal(size(ornn_optim, 2),28));
ornn_optim.D.data[model_1_ixs, model_1_ixs] = AxUtil.Arr.eye(length(model_1_ixs))
ornn_optim.d = Tracker.param(zeros(Float32, size(ornn_optim, 2)));
ornn_optim_ng = model.make_nograd(ornn_optim);


d_par = [length(x) for x in model.pars_no_inpnn(ornn_optim)] |> sum
nn = Chain(Dense(k, d_nn, tanh), Dense(d_nn, d_subspace, identity), 
    Dense(d_subspace, d_par, identity, initW = ((dims...)->Flux.glorot_uniform(dims...)*0.05f0)))
nn_ng = mapleaves(Tracker.data, nn)
Zmap = Flux.param(randn(Float32, k, length(trainIter))*0.01f0);


opt = ADAM(1e-4)
pars = Flux.params(nn, Zmap);

#================================================
               OPTIMISE (BEGIN)
================================================#

function optimise_upper!(ornn_optim, Zmap, trainIter, opt, η, n_epochs, shuffle_examples, lower_ixs)
    opt.eta = η
    nB = length(trainIter)
    W = mocaputil.weights(trainIter; as_pct=false) ./ batch_size
    history = ones(n_epochs*nB) * NaN

    for ee in 1:n_epochs
        rnn = RNN(size(ornn_optim,3), size(ornn_optim,1), ornn_optim.σ)

        if shuffle_examples
            mtl_ixs, trainData = mocaputil.indexed_shuffle(trainIter)
        else
            mtl_ixs, trainData = 1:length(trainIter), trainIter
        end
        for (ii, (Yb, Ub, h0)) in zip(mtl_ixs, trainData)
            h0 && Flux.reset!(rnn)
            Tb = size(Yb, 2)      # not constant
            Zs_post = Zmap[:,ii]  #.+ convert(Array{Float32}, AxUtil.Random.sobol_gaussian(m_bprop,2)'*0.01)

            c_ornn = model.make_rnn_psi(ornn_optim, nn(Zs_post), 1f0)

            model.build_rnn!(rnn, c_ornn)
            x̂ = reduce(hcat, [rnn(Ub[:,i]) for i in 1:Tb])  |> Tracker.collect
            #         ŷ = let m=ornn_optim; m.C*x̂ + m.D*Ub .+ m.d; end   # keep same C, D, d ∀ tasks
            DU = vcat(Ub, c_ornn.D[lower_ixs[end]+1:end,:]*Ub)       # strictly residual connection to lower
            ŷ = let m=c_ornn; m.C*x̂ + DU .+ m.d; end                 # adapt C, D, d too.
            obj = mean(x->x^2, Yb - ŷ) * 8^2 * W[ii]

            # Prior penalty
            obj += 0.5*sum(Zs_post .* Zs_post)

            Tracker.back!(obj)
            history[(ee-1)*nB + ii] = obj.data

            if ii % 34 == 0
                for layer in nn.layers
                    obj += 1e-3*sum(abs, layer.W)
                    obj += 1e-3*sum(abs, layer.b)
                end

                for p in pars
                    Tracker.update!(opt, p, Tracker.grad(p))
                end
            end

            rnn.cell.h.data .= 0       # initial state is a param :/. Easier to reset here.
            Flux.truncate!(rnn);
        end
        printfmtln("{:02d}: {:.5f}", ee, sqrt(mean(history[(1:nB) .+ nB*(ee-1)]))); flush(stdout)

    end
end

redirect_to_files(logname * ".log", logname * ".err") do
    history1 = optimise_upper!(ornn_optim, Zmap, trainIter2, opt, 1e-3, testmode ? 1 : 200, true, model_1_ixs)
end


#================================================
      GLOBAL SEARCH OVER Z - AVOID LOCAL MIN
================================================#

nsmp = testmode ? 10 : 350
_Zsmp = cholesky(cov(Zmap.data')).U * f32(AxUtil.Random.sobol_gaussian(nsmp, k)');

# populate error matrix with above samples
res = ones(Float32, length(trainIter), nsmp)

ornn_optim_ng = model.make_nograd(ornn_optim);
rnn_ng = mapleaves(Tracker.data, RNN(d_in, d_state, ornn_optim.σ))
@time for i in 1:nsmp
    _ψ = nn_ng(_Zsmp[:,i]);
    c_ornn = model.make_rnn_psi(ornn_optim_ng, _ψ, 1f0)
    model.build_rnn!(rnn_ng, c_ornn)
    for (n, (Yb, Ub, h0)) in enumerate(trainIter2)
        h0 && Flux.reset!(rnn_ng)
        Tb = size(Yb, 2)
        x̂ = reduce(hcat, [rnn_ng(Ub[:,i]) for i in 1:Tb])
        ŷ = let m=c_ornn; m.C*x̂ + m.D*Ub .+ m.d; end
        res[n,i] = mean(x->x^2, Yb - ŷ)
    end
end

# sample from implicit posterior (SNIS)
pz = softmax(-32*(res')) 
z_smpopt = copy(Zmap.data)
for i in 1:length(trainIter2)
    z_smpopt[:,i] = _Zsmp[:, argmax(pz[:,i])]
end

Zmap.data .= z_smpopt .+ randn(Float32, k, length(trainIter))*0.005;


#================================================
               OPTIMISE (FINISH)
================================================#
redirect_to_files(logname * ".log", logname * ".err") do
    history2 = optimise_upper!(ornn_optim, Zmap, trainIter2, opt, 5e-4, testmode ? 1 : 150, true, model_1_ixs)
    history3 = optimise_upper!(ornn_optim, Zmap, trainIter2, opt, 2e-4, testmode ? 1 : 150, true, model_1_ixs)
    history4 = optimise_upper!(ornn_optim, Zmap, trainIter2, opt, 1e-4, testmode ? 1 : 150, true, model_1_ixs)
    history5 = optimise_upper!(ornn_optim, Zmap, trainIter2, opt, 3e-5, testmode ? 1 : 20, true, model_1_ixs)
end


# => Enforce D as residual structure for root/legs
model_1_ixsD = reshape(1:size(ornn_optim, 2)*size(ornn_optim, 3), size(ornn_optim, 2), size(ornn_optim, 3)
    )[model_1_ixs,:] .+ model._partition_ldspars_dims(size(ornn_optim)...)[4] |> vec
nn_ng.layers[end].W[model_1_ixsD, :] .= 0
nn_ng.layers[end].b[model_1_ixsD] .= 0
nn_ng.layers[end].b[diag(reshape(model_1_ixsD, 28, 28))] .= 1

redirect_to_files(logname * ".log", logname * ".err") do
    printfmtln("Training SMSE: {:.4f}", smse(trainIter2, model.make_nograd(ornn_optim), Zmap.data, nn_ng))
    fname = format("ornn{:d}_2L_{:d}_{:d}_pool_{:02d}{:02d}_v{:d}.bson", style_ix, d_state, 
    k, day(today()), month(today()), batch_size)
    BSON.bson(fname, m1=model.make_nograd(ornn_base), m2=ornn_optim_ng, nn=nn_ng, Zmap=Zmap.data);
    println(fname)
end