module torchutil

using Flux
using Flux.Tracker: istracked, TrackedArray, TrackedReal
using Distributions: MvNormal
using LinearAlgebra: Diagonal
using Formatting: format
using Random: seed!
using PyCall: pyimport
using ArgCheck


pytorch = pyimport("torch")

function get_psi_decoder(mtorch)
    W1, h1 = let l=get(mtorch.psi_decoder, 0); l.weight, l.bias; end
    W2, h2 = let l=get(mtorch.psi_decoder, 2); l.weight, l.bias; end  # (1)=Tanh
    W3, h3 = let l=get(mtorch.psi_decoder, 3); l.weight, l.bias; end  # (1)=Tanh
    W1, h1, W2, h2, W3, h3 = map(x->f32(x.detach().numpy()), (W1, h1, W2, h2, W3, h3))
    Flux.Chain(Flux.Dense(W1, h1, tanh), Flux.Dense(W2, h2, identity), Flux.Dense(W3, h3))
end

from_torch(x) = f32(x.detach().numpy())
from_torch(x::Array) = x

function get_gru_encoder(mtorch)
    d(x) = f32(x.detach().numpy())
    gru = mtorch.encoder_cell
    n = gru.hidden_size
    Wh, Wi = map(d, (gru.weight_hh, gru.weight_ih))
    b = d(gru.bias_ih) + vcat(d(gru.bias_hh)[1:2*n], zeros(Float32, n))
    b̃ = d(gru.bias_hh)[2*n+1:3*n]
    Wh, Wi, b, b̃ = map(Flux.param, (Wh, Wi, b, b̃))
    Flux.Recur(GRUCellv2(Wi, Wh, b, b̃, Flux.param(zeros(Float32, n))))
end

# GRU2
const gate = Flux.gate

mutable struct GRUCellv2{A,V}
  Wi::A
  Wh::A
  b::V
  b̃::V
  h::V
end

GRUCellv2(in, out; init = glorot_uniform) =
  GRUCellv2(param(init(out*3, in)), param(init(out*3, out)),
          param(init(out*3)), param(zeros(out)))

function (m::GRUCellv2)(h, x)
    b, o = m.b, size(h, 1)
    gx, gh = m.Wi*x, m.Wh*h
    r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ r .* m.b̃ .+ gate(b, o, 3))
    h′ = (1 .- z).*h̃ .+ z.*h
    return h′, h′
end

Flux.hidden(m::GRUCellv2) = m.h

Flux.@treelike GRUCellv2

Base.show(io::IO, l::GRUCellv2) =
  print(io, "GRUCellv2(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ")")

"""
    GRUv2(in::Integer, out::Integer)

See Flux.jl for (a little!) more documentation. Most is self-explanatory; the
function def (m::GRUCellv2)(h, x) is essentially the entire procedure.

(vs. Original): reset gate introduces a different bias into h̃. This is so we
have a direct translation from PyTorch, but is essentially just a different
way of writing it. It would seem Mike Innes preferred the elegance of the other
representation.
"""
GRUv2(a...; ka...) = Flux.Recur(GRUCellv2(a...; ka...))



# encoder + emission
mutable struct MTGRU{R, V1, V2, S}
  encoder::R
  M_μ::V2
  M_lσ::V2
  b_lσ::V1
  psi_decoder::S
  gru_Wih::V2
  gru_Whh::V2
  gru_bias::V1
  output_dim::Int
  residual::Bool
end

Base.show(io::IO, s::MTGRU) =
  print(io, format("MTGRU(k={:d}); (enc={:d}, task={:d}, out={:d})",
  size(s.psi_decoder[1].W, 2), encoder_size(s)[2], decoder_size(s)[2], s.output_dim))

tracked_eltype(T) = T
tracked_eltype(x::TrackedArray{T,N,A}) where {T,N,A} = T
tracked_eltype(x::TrackedReal{T}) where {T} = T
tracked_eltype(::Type{Tracker.TrackedReal{T}}) where T = T
tracked_eltype(::Type{Tracker.TrackedArray{T,N,A}}) where {T,N,A} = T
Base.eltype(s::MTGRU{R, V1, V2, S}) where {R, V1, V2, S} = tracked_eltype(eltype(V1))

encoder_size(s::MTGRU) = (size(s.encoder.cell.Wi, 2), size(s.encoder.cell.Wi, 1) ÷ 3)
decoder_size(s::MTGRU) = (size(s.gru_Wih, 1), size(s.gru_Wih, 2) ÷ 2)

Flux.children(s::MTGRU) = (s.encoder, s.M_μ, s.M_lσ, s.b_lσ, s.psi_decoder, s.gru_Wih, s.gru_Whh, s.gru_bias)
Flux.mapchildren(f, s::MTGRU) = MTGRU(f.(Flux.children(s))..., s.output_dim, s.residual)
Flux.reset!(s::MTGRU) = begin; h = Tracker.data(s.encoder.state); h .= 0; end

function get_MTGRU(mtorch; tracked=true)
    d(x) = f32(x.detach().numpy())
    psi_dec = get_psi_decoder(mtorch)
    enc = get_gru_encoder(mtorch)
    M_μ, M_lσ, b_lσ = map(d, (mtorch.to_mu, mtorch.to_lsigma, mtorch.to_lsigma_bias))
    Wih, Whh, bias = map(d, (mtorch.gru_Wih, mtorch.gru_Whh, mtorch.gru_bias))
    mtgru = MTGRU(enc, M_μ, M_lσ, b_lσ, psi_dec, Wih, Whh, bias, mtorch.HUMAN_SIZE,
           mtorch.residual_output)
    return tracked ? mapleaves(Flux.param, mtgru) : mtgru
end

function posterior(s::MTGRU{R,V1,V2,S}, Y::V2, as_dist=true) where {R, V1, V2, S}
    h = Tracker.data(s.encoder.state); (h .= 0)
    [s.encoder(Y[:,t]) for t in 1:size(Y,2)];
    h = s.encoder.state
    if as_dist
        return MvNormal(vec(h' * s.M_μ), Diagonal(exp.(2*(vec(h' * s.M_lσ) .+ s.b_lσ))))
    else
        return h' * s.M_μ, (vec(h' * s.M_lσ) .+ s.b_lσ)
    end
end

function get_task_gruCD(s::MTGRU{R,V1,V2,S}, z::V1) where {R, V1, V2, S}
    @argcheck sum(size(z) .!= 1) == 1
    T = eltype(V1)
    ψ = s.psi_decoder(z)

    n = decoder_size(s)[2]
    Whh, bh, Wih, C, D = _decoder_par_reshape(s, ψ)
    Wi, Wh = Matrix(hcat(s.gru_Wih, Wih)'), Matrix(hcat(s.gru_Whh, Whh)')
    b = vcat(s.gru_bias, bh)
    b̃, state = zeros(T, n),  zeros(T, n)
    istracked(Wh) &&
        begin; b̃, state = map(Flux.param, (b̃, state)); end
    gru = Flux.Recur(GRUCellv2(Wi, Wh, b, b̃, state))
    return gru, C, D
end

get_task_gru(s::MTGRU{R,V1,V2,S}, z::V1) where {R, V1, V2, S} = get_task_gruCD(s, z)[1]

function decode(s::MTGRU{R, V1, V2, S}, z::V1, U::V2,
        state::V1=zeros(eltype(V1), decoder_size(s)[2])) where {R, V1, V2, S}
    T = eltype(V1)
    gru, C, D = get_task_gruCD(s, z)
    h = Tracker.data(gru.state); (h .= state)
    dec = reduce(hcat, [gru(U[:,t]) for t in 1:size(U,2)])

    if s.residual
        yhat = Matrix(C') * dec + vcat(U, Matrix(D') * U)
    else
        yhat = Matrix(C') * dec + Matrix(D') * U
    end
    return yhat
end


function _decoder_par_shape(s::MTGRU)
    sz_in, sz_dec, sz_out = (decoder_size(s)..., s.output_dim)
    Whh = (sz_dec, sz_dec)
    bh = (sz_dec,)
    Wih = (sz_in, sz_dec)
    C = (sz_dec, sz_out)
    D = s.residual ? (sz_in, (sz_out - sz_in)) : (sz_in, sz_out)
    return Whh, bh, Wih, C, D
end

_decoder_par_size(s::MTGRU) = [prod(x) for x in _decoder_par_shape(s)]

function _decoder_par_slice(s::MTGRU)
    csizes = cumsum(vcat(0, _decoder_par_size(s)))
    return [(csizes[i]+1):csizes[i+1] for i in 1:(length(csizes)-1)]
end

function _decoder_par_reshape(s::MTGRU, ψ)
    shapes, slices = _decoder_par_shape(s), _decoder_par_slice(s)
    return [reshape_rmaj(ψ[slice], shape) for (shape, slice) in zip(shapes, slices)]
end


reshape_rmaj(x, sz) = permutedims(reshape(x, reverse(sz)), reverse(1:length(sz)))


#= ⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅
        Test equality of torch model with Flux equivalent
⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅⋅ =#

unsqueeze(x, d) = reshape(x, (size(x)[1:d-1]..., 1, size(x)[d:end]...))

function test_posteriors(mtgru_ng, mtorch, Ys)
    max_μ, max_lσ = (0., 0), (0., 0)
    for (i,y) in enumerate(Ys)
        μ, logσ = posterior(mtgru_ng, y, false)
        tμ, tlogσ = mtorch.encode(pytorch.tensor(unsqueeze(y', 1)))
        μ, logσ = map(vec, (μ, logσ))
        tμ, tlogσ = map(y->vec(y.detach().numpy()), (tμ, tlogσ))
        errμ, errlσ = sum(x->x^2, μ - tμ)/length(μ), sum(x->x^2, logσ - tlogσ)/length(μ)
        max_μ = (max(max_μ[1], errμ), errμ > max_μ[1] ? i : max_μ[2])
        max_lσ = (max(max_lσ[1], errlσ), errlσ > max_lσ[1] ? i : max_lσ[2])
    end
    max_μ[1] > 1e-4 && @warn "max_μ failed the test"
    max_lσ[1] > 1e-4 && @warn "max_lσ failed the test"
    return max_μ, max_lσ   # max err, ix
end

function test_decoders(mtgru_ng, mtorch, Us, zdist::MvNormal)
    seed!(12345678)
    max_Δ = (0., 0)
    T = tracked_eltype(eltype(mtgru_ng))
    pytusq(x, d) = pytorch.tensor(unsqueeze(x, d))
    for (i,u) in enumerate(Us)
        z = vec(T.(rand(zdist)))
        Fhat = decode(mtgru_ng, z, u)
        tcU, tcz, state = pytusq(u', 1), pytusq(z, 1), pytorch.zeros(1,256)
        That = mtorch.forward_given_z(tcU, tcz, state)[1].detach().numpy()[1,:,:]'
        err  = sum(x->x^2, That - Fhat) / length(That)
        max_Δ = (max(max_Δ[1], err), err > max_Δ[1] ? i : max_Δ[2])
    end
    max_Δ[1] > 1e-6 && @warn format("Failed test. Worst deviation at {:d}", max_Δ[2])
    return max_Δ # max err, ix
end

end
