using AxUtil         # spline basis stuff in Math

#= =======================================================================
                Smoothing the trajectory (esp. omit snaking)
   ======================================================================= =#


#= -----------------------------------------------------------------------
               Ramer-Douglas-Peucker line simplification
  ----------------------------------------------------------------------- =#

   # Ramer-Douglas-Peucker line simplification algorithm copied from Rosetta
   # code. Source:
   # https://rosettacode.org/wiki/Ramer-Douglas-Peucker_line_simplification
   #
   # Very happy to acknowledge the author's contribution to this, but I don't
   # have time to chase up necessary compatibility between licenses as this is
   # not a core part of my project. If necessary, I'll rewrite or obtain from
   # elsewhere. It appears this is GFDL licensed:
   # https://www.gnu.org/licenses/old-licenses/fdl-1.2.html

const Point = Vector{Float64}

function perpdist(pt::Point, lnstart::Point, lnend::Point)
   d = normalize!(lnend .- lnstart)

   pv = pt .- lnstart
   # Get dot product (project pv onto normalized direction)
   pvdot = dot(d, pv)
   # Scale line direction vector
   ds = pvdot .* d
   # Subtract this from pv
   return norm(pv .- ds)
end

function rdp(plist::Vector{Point}, ϵ::Float64 = 1.0)
   if length(plist) < 2
       throw(ArgumentError("not enough points to simplify"))
   end

   # Find the point with the maximum distance from line between start and end
   distances  = collect(perpdist(pt, plist[1], plist[end]) for pt in plist)
   dmax, imax = findmax(distances)

   # If max distance is greater than epsilon, recursively simplify
   if dmax > ϵ
       fstline = plist[1:imax]
       lstline = plist[imax:end]

       recrst1 = rdp(fstline, ϵ)
       recrst2 = rdp(lstline, ϵ)

       out = vcat(recrst1, recrst2)
   else
       out = [plist[1], plist[end]]
   end

   return unique(out)
end

rdp(X::AbstractMatrix{T}, ϵ = 1.0) where T = convert(Matrix{T}, reduce(hcat,
    rdp([convert(Point, X[i,:]) for i in 1:size(X,1)], ϵ))')

# => A few utilities for merging boolean vectors of knots. To be honest, these
# were often less useful than originally hoped, but this was hacked together
# and I'm not going to risk breaking it by attempting to remove some stages.


function merge_knots!(u; take_first=true, merge_triplets=true)
   i = 1
   while i < length(u)-2
       if u[i] == 1
           if merge_triplets && u[i+2] > 0
               u[i], u[i+1], u[i+2] = 0, 1, 0
               i += 3
               continue
           elseif u[i+1] > 0
               if take_first
                   u[i+1] = 0
               else
                   u[i] = 0
               end
               i += 2
               continue
           end
       end
       i += 1
   end
   return u
end
merge_knots(u; take_first=true, merge_triplets=true) = merge_knots!(copy(u);
   take_first=take_first, merge_triplets=merge_triplets)

function reinforce_knots(u)
   n = length(u)
   out = copy(u)
   for i in 1:n
       if u[i] == 1
           i > 3 && (out[i-3] = 1)
           i < n - 2 && (out[i+3] = 1)
       end
   end
   return out
end



#= -----------------------------------------------------------------------
            Perform smoothing using RDP => Cubic B-Splines
  ----------------------------------------------------------------------- =#

    # RDP is used to get a base description of the trajectory in terms of
    # a small number of line segments. We can extract knots for the spline
    # approximation from this.
    #
    # (Simply relying on regularisation for simplification typically oversmooths
    # sharp corners. There are two violations to naïve application of
    # regularisation: firstly the noise is substantially not iid, it has
    # ~periodicity and other regularities. Secondly, the underlying function has
    # a non-stationary smoothness parameter. RDP knot choices seems a nice way
    # to work around this.)
    #
    # This isn't perfect, and you'll see a lot of custom knot removals and
    # additions below. Nevertheless the lion's share are algorithmic, and the
    # raw fits are really quite impressive (imo).

# defaults (see get below: 0.7, 0.3 respectively)
epsilons = Dict(10=>0.6, 14=>0.5, 15=>0.9, 16=>0.9, 17=>0.5, 18=>1.2, 19=>1.2,
    20=>1.2, 21=>1.2, 25=>1.2, 26=>2.5, 27=>2.5, 28=>3.5, 31=>1.5)
θ2d_thrsh = Dict(10=>0.28, 18=>0.25, 19=>0.25, 26=>Inf, 27=>0.4);

# additional knots, treated as breaching the curvature threshold.
addl_pts = Dict(10 => [462], 18 => [1570], 19=>[880,920, 1022], 22=>[510],
    26=>[946, 2890])
# remove points discovered by curvature condition
rm_pts = Dict(10=>81:88,
    18=>vcat(670:700, 1250:1290),
    19=>887:895,
    20=>1293:1314,
    21=>vcat(659:661, 1724:1732),
    22=>509:512,
    23=>991:998,
    24=>1040:1058,
    25=>vcat(41:47, 146:170, 590:600, 800:820,1440:1460, 1720:1740),
    27=>vcat(69:74, 284:290, 2060:2070),
    28=>vcat(2543:2560, 3448:3460),
    28=>vcat(3829:3835, 3940:3960, 4065:4080),
    29=>vcat(2427:2440),
    30=>vcat(257:270, 1120:1130),
    31=>vcat(962:980, 1060:1080, 1200:1210, 1322:1330, 1532:1540, 1600:1630, 2096:2107))
# additional knots, but not treated as per the curvature ones: not triplicated.
addl_sgl = Dict(3 => [60],
    20=>[1293, 1314, 1330],
    21=>[650,660,670,1710,1728],
    23=>[980,1010],
    24=>[650,1040,1060,1210,1220,1230,1240],
    25=>[30, 50, 1440, 1460, 1490],
    26=>[1945, 1970, 2860, 2920],
    27=>[70, 100, 1530],
    28=>[3455],
    30=>[264],
    31=>vcat([60,276], [1060, 1070, 1090, 1100]))
# gradient discontinuity at knot
addl_discont = Dict(28=>[3455], 30=>[264])
# file length (assert)
file_length=[1472,1627,1393,1422,1111,781,3016,652,2071,3071,
 1877,1909,3662,1822,3484,1688,1586,1843,1974,1952,1942, 922,
 1446,1568,2217,2955,2856,4525,2440,2334,2976]

function smooth_trajectory(start::Vector{T}, root_x::Vector{T}, root_z::Vector{T},
        root_r::Vector{T}; file_ix::Int=0, turn_thrsh::Float64=0.15, check=true) where T

    trj = _traj_fk(root_x, root_z, root_r; start=start);
    ϵ = get(epsilons, file_ix, 0.7)  # default 0.7, o.w. see above for epsilon
    traj_mat = hcat(trj[1], trj[2])

    # Do RDP transform and extract knots.
    simplified = rdp(traj_mat, ϵ)

    # Is there a better way to find indices of the returned? Yes. But here we are...
    ts = map(1:size(simplified,1)) do i
            findfirst((simplified[i,1] .≈ traj_mat[:,1]) .& (simplified[i,2] .≈ traj_mat[:,2]))
    end

    # Find additional turning points by looking at large large second "derivatives" (differences)
    θ = atan.(root_x, root_z);
    thrsh = file_ix > 0 ? get(θ2d_thrsh, file_ix, 0.3) : 0.3
    knots = abs.(diff(vcat(0, θ))) .> thrsh;
    knots = vcat(knots, false)    # output of FK is n+1 since velocity "predicts" next frame.

    # add/remove custom points specified above
    if file_ix > 0
        rm_knots = vcat(1, get(rm_pts, file_ix, []))   # always rm initial knot
        check && @assert (length(root_x) == file_length[file_ix]) format(
            "Expecting Mason file of length {:d}. Got {:d}.", file_length[file_ix], length(root_x))
        knots[rm_knots] .= 0
        addl_knots = get(addl_pts, file_ix, [])
        length(addl_knots) > 0 && (knots[addl_knots] .= 1)

        # reinforce these addl knots and merge with RDP knots
        knots = reinforce_knots(knots)
        extreme_turns = vcat(findall(merge_knots(knots; take_first=false, merge_triplets=false)),
            get(addl_sgl, file_ix, []))
        knots[ts] .= 1
        addl_knots = get(addl_sgl, file_ix, [])
        length(addl_knots) > 0 && (knots[addl_knots] .= 1)
        knots_ix = findall(merge_knots(knots; take_first=false, merge_triplets=false));
        knots_ix = sort(vcat(knots_ix, get(addl_discont, file_ix, [])));
    else
        knots[ts] .= 1
        knots[1], knots[end] = 1, 1    # just in case start/end not already there.
        knots_ix = findall(merge_knots(knots; take_first=false, merge_triplets=false));
    end

    # Calculate spline coefficients via OLS
    N = length(root_x)+1

    #  -- spline calc needs knots at begin/end of seq
    (knots_ix[1] != 1) && (knots_ix = vcat(1, knots_ix))
    (knots_ix[end] != N) && (knots_ix = vcat(knots_ix, N))

    #  -- calculate spline basis / deriv basis.
    knots = knots_ix .- 1
    eval_ts = 0:(N-1)
    splBasis = AxUtil.Math.bsplineM(eval_ts, knots, 3+1)
    derivBasis = AxUtil.Math.bsplineM(eval_ts, knots, 3+1, 1)
    # deriv2Basis = AxUtil.Math.bsplineM(eval_ts, knots, 3+1, 2)  # 2nd deriv not used

    #  -- perform regression
    A = traj_mat' / splBasis';
    smoothed_trj = splBasis * A';

    # gradient of forward angle
    smoothed_deriv = derivBasis * A';
    g = atan_dt.(diff(smoothed_trj[:, 2]), diff(smoothed_trj[:, 1]),
                 diff(smoothed_deriv[:, 2]), diff(smoothed_deriv[:, 1]))

    # Calculate sharp turns via threshold on angular velocity
    sharp_turns = abs.(g) .> turn_thrsh
    turn_window = 10
    for ix in findall(sharp_turns)   # smoosh...
        sharp_turns[max(1, ix-turn_window):min(N-1, ix+turn_window)] .= true
    end

    return smoothed_trj, smoothed_deriv, findall(sharp_turns), g
end


#= -----------------------------------------------------------------------
                      Foot contact smoothing
   ----------------------------------------------------------------------- =#
using Lasso
"""
    foot_anomalies(ixs; knot_spacing::Int=20, thrsh::Float64=2)

Fit a B-spline model to the deltas between foot contacts using L1
regression (robust). Return any indices which are g.t. `thrsh` times
the average deviation.
"""
function foot_anomalies(ixs; knot_spacing::Int=20, thrsh::Float64=5.0, order::Int=1,
        do_plot=false, λ=1.0)
    n = length(ixs)
    Δs = diff(ixs)
    knots = 1:knot_spacing:(ceil(n/knot_spacing)*knot_spacing+1)
    basis = AxUtil.Math.bsplineM(1:n-1, knots, order+1)[:,1:end-1]
    β = coef(fit(LassoModel, basis, Δs, intercept=true, λ=[λ]))
    modelled = [ones(n-1,1) basis] * β
    if do_plot
        plot(Δs); plot(modelled);
    end
    _ads = abs.(Δs - modelled)
    _mad = mean(_ads)
    anomalies = findall((_ads ./ _mad) .> thrsh)
    do_plot && scatter(anomalies .-1, Δs[anomalies], marker="x", color="r")
    return anomalies, (Δs - modelled)[anomalies]
end

function foot_anomalies_process(ixs::Vector, ads)
    n = length(ixs)
    i = 1
    out = []
    while i <= n
        if i < n && (ixs[i] + 1 == ixs[i+1]) && ads[i] < 0
            push!(out, ixs[i+1])
            i += 2
            continue
        end
        # @warn "no matching pair for $i, $ixs[i]"
        i += 1
    end
    return out
end

function make_phase_from_contacts(ixsL, ixsR, L)
    # Merge the Left and Right foot indices, marking which came from which
    # =====================================================================

    n, n_1, n_2 = sum(length, [ixsL, ixsR]), length(ixsL), length(ixsR)
    m_ixs, m_left = Vector{Int}(undef, n), Vector{Bool}(undef, n)
    i_1, i_2 = 1, 1
    for j in 1:n
        if i_1 > n_1
            m_ixs[j], m_left[j] = ixsR[i_2], false
            i_2 += 1
            continue
        elseif i_2 > n_2
            m_ixs[j], m_left[j] = ixsL[i_1], true
            i_1 += 1
            continue
        end

        if ixsL[i_1] < ixsR[i_2]
            m_ixs[j], m_left[j] = ixsL[i_1], true
            i_1 += 1
        elseif ixsL[i_1] > ixsR[i_2]
            m_ixs[j], m_left[j] = ixsR[i_2], false
            i_2 += 1
        else
            print("i_1 is $i_1, i_2 is $i_2")
            error("Unreachable error")
        end
    end

    # Interpolate between these indices
    # ================================================
    out = zeros(L);

    vals = [0, π]
    prv_ix, prv_isleft = m_ixs[1], m_left[1]

    for j in 2:n
        ix, isleft = m_ixs[j], m_left[j]
        _interp = range(0, π, length=(ix - prv_ix + 1)) .+ (isleft == 1 ? 0 : π)
        out[prv_ix:ix] = _interp
        prv_ix, prv_isleft = ix, isleft
    end

    # deal with beginning / end
    # ================================================
    δ₋ = diff(out[m_ixs[1]:m_ixs[1]+1])[1] |> mod2pi
    for j in (m_ixs[1]-1):-1:1
        out[j] = mod2pi(out[j+1] - δ₋)
    end

    δ₊ = diff(out[m_ixs[end]-1:m_ixs[end]])[1] |> mod2pi
    for j in (m_ixs[end]+1):1:L
        out[j] = mod2pi(out[j-1] + δ₊)
    end

    # turn into angular components
    # ================================================
    out = [cos.(out) sin.(out)];
    return out
end


#= -----------------------------------------------------------------------
         Arctan -- rm jumps at 2π boundary, and define derivatives.
   ----------------------------------------------------------------------- =#


function fix_atan_jumps(x)
    """
    `atan(y, x)` or `atan2` is super useful, but the returned angle is in
    [-π, π], which results in 2π jumps at the boundary. This function assumes
    continuity and will grow unboundedly →∞, →-∞ to get the most continuous fn.
    """
    offset = 0
    out = similar(x); out[1] = x[1]
    for i in 2:length(x)
        if x[i] - x[i-1] > π
            offset -= 2π
        elseif x[i] - x[i-1] < -π
            offset += 2π
        end
        out[i] = x[i] + offset
    end
    return out
end

atan_dt(y, x, ẏ, ẋ) = (ẏ * x - ẋ * y) / (x^2  + y^2)
atan_d2t(y, x, ẏ, ẋ, ÿ, ẍ) = let num=(ẏ * x - ẋ * y); denom=(x^2  + y^2); num_dt=(ÿ * x - ẍ * y)
    denom_dt= 2*(x*ẋ  + y*ẏ); (num_dt*denom - num*denom_dt) / denom^2; end


function full_body_rotation_vs_forward(body_sin::Vector{T}, body_cos::Vector{T}) where T
    # get body angle with forward
    x = fix_atan_jumps(atan.(body_sin, body_cos))

    # get piecewise approx, choosing ϵ ≈ π
    pwise = rdp(hcat(1:length(body_sin), x), 3.0)
    ixs = pwise[:,1]

    # find the multiple of pi of each breakpoint to determine if true change.
    pi_mult = round.(Int, x[Int.(ixs)] / 2π)
    ix_change = findall(diff(pi_mult) .!= 0) .+ 1

    # obtain start/end ixs for all true changepoints and create binary mask.
    start_ixs, end_ixs = Int.(ixs[ix_change .- 1]), Int.(ixs[ix_change])
    out = zeros(T, length(body_sin))
    for (s, e) in zip(start_ixs, end_ixs)
        # println([s,e])
        out[s:e] .= 1
    end
    return out
end
