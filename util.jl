module mocaputil

using StatsBase, Statistics, MultivariateStats
using ArgCheck

import Random: shuffle

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


"""
    generate_synthetic_inputs(N::Int; ω=0.177, speed=0.43, Δt=5)

Generate an input matrix `U` for an MTDS model (using the 35-d param), for
`N` timesteps. Currently we only have a **forward motion** implemented;
however, one only need change the `strj` and `s∇` (tangent) in the code here.
"""
function generate_synthetic_inputs(N::Int; ω=0.177, speed=0.43, Δt=5)
    X = zeros(35, N)
    N_extra = N + 6*Δt + 1

    # Trajectory generated
    strj = hcat(zeros(N_extra), (1:N_extra).* speed) # trajectory (straight)
    s∇ = hcat(zeros(N_extra), ones(N_extra))         # normalised tangent

    # ================ Smoothed Lagrangian trajectory ===========
    # (1) Inverse Kinematics convert to => Lagrange representation
    slagr_r, slagr_x, slagr_z = mocapio._traj_invk(strj, s∇);
    slagr_r = mocapio.fix_atan_jumps(slagr_r)
    root_x, root_z, root_r = slagr_x, slagr_z, slagr_r

    # (2) Convert to lagged form for use in inputs
    for i = 1:N
        rs = i:(i+(Δt*6))
        t_x, t_z = mocapio._traj_fk(root_x[rs], root_z[rs], root_r[rs])
        X[1, i], X[2, i], X[3, i] = root_r[i+1], root_x[i+1], root_z[i+1]
        X[4:9, i] = t_x[Δt:Δt:6*Δt]
        X[10:15, i] = t_z[Δt:Δt:6*Δt]
    end

    # ==================== Absolute differenced trajectory ==================
    # (1) apply difference operator to Eulerian traj
    abs_trj = diff(vcat(zeros(eltype(strj), 2)', strj), dims=1)

    # (2) Convert to lagged form for use in inputs
    for i in 1:N
        X[15 .+ (1:6), i] = abs_trj[(Δt:Δt:6*Δt) .+ i, 1]'
        X[15 .+ (7:12), i] = abs_trj[(Δt:Δt:6*Δt) .+ i, 2]'
    end

    # ======================= Reverse turn indicators =======================

    # (All reverse turn indicators are zero)

    # ============================== Phase ==================================
    X[34, :] = cos.((1:N) .* ω)
    X[35, :] = sin.((1:N) .* ω);

    return convert(Matrix{Float32}, X')
end

#==============================================================================
    ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ Data Iterator  ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
 =============================================================================#


struct DataIterator
    data::Array{D}  where {D <: Dict}
    batch_size::Int
    min_size::Int
    start::Int
    DataIterator(d, b_sz, m_sz, start) = begin; @assert m_sz < b_sz; new(d, b_sz, m_sz, start); end
end
DataIterator(data, batch_size; min_size=1, start=0) = DataIterator(data, batch_size, min_size, start)
Base.copy(di::DataIterator) = DataIterator(deepcopy(di.data), copy(di.batch_size), copy(di.min_size), copy(di.start))

function Base.iterate(iter::DataIterator, state=(1, iter.start))
    element, ix = state

    (element > length(iter.data)) && return nothing
    while ix + iter.min_size > size(iter.data[element][:Y], 2)
        element += 1
        ix = iter.start
        (element > length(iter.data)) && return nothing
    end
    new_state = ix == iter.start   # not in while, since 1st iterate in general won't use this loop.

    chunk  = iter.data[element]
    cur_length = size(chunk[:Y], 2)
    eix  = min(ix + iter.batch_size, cur_length)
    ix += 1

    return ((chunk[:Y][:,ix:eix], chunk[:U][:,ix:eix], new_state), (element, eix))
end

function weights(iter::DataIterator; as_pct::Bool=true)
    w = [size(y,2) for (y, u, new_state) in iter]
    return as_pct ? w / sum(w) : w
end

function Base.length(iter::DataIterator)
    map(iter.data) do x
        d, r = divrem(size(x[:Y],2)-iter.start, iter.batch_size);
        d + (r >= iter.min_size);
        end |> sum
end

function shuffle(di::DataIterator)
    newdata = shuffle([Dict(:Y=>cY, :U=>cU) for (cY, cU, h0) in di])
    mocaputil.DataIterator(newdata, di.batch_size, 1, di.start)
end


function indexed_shuffle(di::DataIterator)
    order = shuffle(1:length(di))
    dataArray = [Dict(:Y=>cY, :U=>cU) for (cY, cU, h0) in di]
    newdata = [dataArray[i] for i in order]
    order, mocaputil.DataIterator(newdata, di.batch_size, 1, di.start)
end


function get_file_pos_from_iter(di::DataIterator, styles_lkp::Vector,
        style_ix::Int, ix::Int)
    ffs = vcat([s for (i,s) in enumerate(styles_lkp) if i != style_ix]...)
    e, file_offset = 1, di.start
    for i in 1:ix-1
        _, (e, file_offset) = iterate(di, (e, file_offset))
    end
    return ffs[e], file_offset+1
end

function get_file_pos_from_iter_test(di::DataIterator, styles_lkp::Vector,
        style_ix::Int, ix::Int)
    mocaputil.get_file_pos_from_iter(di, [styles_lkp[style_ix]], 0,ix)
end
#### TESTING
# tmpiter = mocaputil.DataIterator(trainSTL, 256, min_size=32, start=102);
# tmp_ii = Int[]
# for (ii, (_y, _u, new_state)) in enumerate(tmpiter)
#     @argcheck size(_y, 2) == size(_u, 2)
#     printfmtln("{:2d}. {:s}Chunk size is: {:d}", ii, new_state ? "(new state) " : "", size(_y, 2))
#     push!(tmp_ii, ii)
# end
# @assert maximum(tmp_ii) == length(tmpiter)

end # module end
