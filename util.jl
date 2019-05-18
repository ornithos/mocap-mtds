module mocaputil

using StatsBase, Statistics, MultivariateStats
export transform, scale_transform
export fit
export invert
export MyStandardScalar

"""
    fit(MyStandardScaler, X, dims)

Fit a standardisation to a matrix `X` s.t. that the rows/columns have mean zero and standard deviation 1.
This operation calculates the mean and standard deviation and outputs a MyStandardScalar object `s` which
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


end # module end
