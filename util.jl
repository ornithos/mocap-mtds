module mocaputil

using StatsBase, Statistics, MultivariateStats
using ArgCheck

export transform, scale_transform
export fit
export invert
export MyStandardScaler

"""
    fit(MyStandardScaler, X, dims)

Fit a standardisation to a matrix `X` s.t. that the rows/columns have mean zero and standard deviation 1.
This operation calculates the mean and standard deviation and outputs a MyStandardScaler object `s` which
allows this same standardisation to be fit to any matrix using `transform(s, Y)` for some matrix `Y`. Note
that the input matrix `X` is *not* transformed by this operation. Instead use the above `transform` syntax
on `X`.

Note a couple of addendums:

1. Any columns/rows with constant values will result in a standard deviation of 1.0, not 0. This is to avoid NaN errors from the transformation (and it is a natural choice).
2. If only a subset of the rows/columns should be standardised, an additional argument of the indices may be given as:

    `fit(MyStandardScaler, X, operate_on, dims)`

    The subset to operate upon will be maintained through all `transform` and `invert` operations.
"""
mutable struct MyStandardScaler{T}
    μ::Array{T,1}
    σ::Array{T,1}
    operate_on::Array{L where L <: Int,1}
    dims::Int
end

Base.copy(s::MyStandardScaler) = MyStandardScaler(copy(s.μ), copy(s.σ), copy(s.operate_on), copy(s.dims))

StatsBase.fit(::Type{MyStandardScaler}, X::Matrix, dims::Int=1) = MyStandardScaler(vec(mean(X, dims=dims)),
    vec(std(X, dims=dims)), collect(1:size(X,3-dims)), dims) |> post_fit

StatsBase.fit(::Type{MyStandardScaler}, X::Matrix, operate_on::Array{L where L <: Int}, dims::Int=1) =
    MyStandardScaler(vec(mean(X, dims=dims))[operate_on],
    vec(std(X, dims=dims))[operate_on],
    operate_on, dims) |> post_fit

function post_fit(s::MyStandardScaler)
    bad_ixs = (s.σ .== 0)
    s.σ[bad_ixs] .= 1
    s
end

function scale_transform(s::MyStandardScaler, X::Matrix, dims::Int=s.dims)
    (dims != s.dims) && @warn "dim specified in transform is different to specification during fit."
    tr = dims == 1 ? transpose : identity
    tr2 = dims == 2 ? transpose : identity
    if s.operate_on == 1:size(X, 3-dims)
        out = (X .- tr(s.μ)) ./ tr(s.σ)
    else
        out = tr2(copy(X))
        s_sub = copy(s)
        s_sub.operate_on = 1:length(s.operate_on)
        s_sub.dims = 1
        out[:, s.operate_on] = transform(s_sub, out[:, s.operate_on], 1)
        out = tr2(out)
    end
    out
end

function invert(s::MyStandardScaler, X::Matrix, dims::Int=s.dims)
    (dims != s.dims) && @warn "dim specified in inversion is different to specification during fit."
    tr = dims == 1 ? transpose : identity
    tr2 = dims == 2 ? transpose : identity
    if s.operate_on == 1:size(X, 3-dims)
        out = (X .* tr(s.σ)) .+ tr(s.μ)
    else
        out = tr2(copy(X))
        s_sub = copy(s)
        s_sub.operate_on = 1:length(s.operate_on)
        s_sub.dims = 1
        out[:, s.operate_on] = invert(s_sub, out[:, s.operate_on], 1)
        out = tr2(out)
    end
    out
end



"""
    fit(OutputDifferencer, Y)

Differences the joint positions in an output matrix `Y`, size N x d, with d
either {64, 68}, depending on whether foot contacts were appended. This utility
essentially just performs differencing of the middle columns 4:64, but saves
the initial frame, since this is lost with differencing. The transformation can
then be inverted using the `invert` command to modelled data. Note however that
because we're using diff/cumsum on potentially large arrays, the accumulated
error can be nontrivial, especially if using Float32s.

Methods:
    s = fit(OutputDifferencer, Y)                   # fit object (i.e. check dims, and save first frame)
    Ydiff = difference_transform(s, Y)              # perform transformation
    s, Ydiff = fit_transform(OutputDifferencer, Y)  # do both at once.
    Yhat = invert(s, Ydiff)                         # invert transformation.
"""
mutable struct OutputDifferencer{T}
    first_frame::Array{T, 1}
    operate_on::AbstractArray{L where L <: Int,1}
end

Base.length(s::OutputDifferencer) = length(s.first_frame, 1)
Base.copy(s::OutputDifferencer) = OutputDifferencer(copy(s.first_frame), copy(s.operate_on))

function StatsBase.fit(::Type{OutputDifferencer}, Y)
    @argcheck size(Y, 2) in [64, 68]
    return OutputDifferencer(Y[1,:], 4:64)
end

function difference_transform(s::OutputDifferencer, Y)
    @assert Y[1,:] == s.first_frame
    return hcat(Y[2:end,1:(s.operate_on[1] - 1)],
        diff(Y[:,s.operate_on], dims=1),
        Y[2:end,(s.operate_on[end] + 1):end])
end

function fit_transform(::Type{OutputDifferencer}, Y)
    s = fit(OutputDifferencer, Y)
    return s, difference_transform(s, Y)
end

function invert(s::OutputDifferencer, Y)
    tr_first = reshape(s.first_frame, 1, :)
    inv = cumsum(vcat(tr_first[:,s.operate_on], Y[:,s.operate_on]), dims=1)
    return vcat(tr_first,
                hcat(Y[:,1:(s.operate_on[1] - 1)],
                    inv[2:end,:],
                    Y[:,(s.operate_on[end] + 1):end])
                )
end

"""
    no_pos(X)

where X is an *INPUT DATA* matrix. X will be checked to be the right input
matrix size, and then discard the joint position elements. Note that this
returns a view of `X`, rather than a copy.

`no_poscp` is a shortcut for the copy version, otherwise do:

    no_pos(X; doCopy = true)
"""
function no_pos(X::AbstractMatrix; doCopy=false)
    @argcheck size(X, 2) == 121
    return doCopy ? X[:,1:60] : view(X, :, 1:60)
end
"""
    no_poscp(X)

removes joint positions from an input array X (returns a copy). See `no_pos`.
"""
no_poscp(X::AbstractMatrix) = no_pos(X; doCopy=true)
end # module end
