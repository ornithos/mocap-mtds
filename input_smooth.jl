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

rdp(X::AbstractMatrix, ϵ = 1.0) = reduce(hcat, rdp([X[i,:] for i in 1:size(X,1)], ϵ))'

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

function smooth_trajectory(start::Vector{T}, root_x::Vector{T}, root_z::Vector{T},
        root_r::Vector{T}, file_ix::Int=0, turn_thrsh::Float64=0.15) where T
    trj = _traj_fk(start, root_x, root_z, root_y);
    ϵ = get(epsilons, file_ix, 0.7)  # default 0.7, o.w. see above for epsilon
    ix_rng = 1:length(root_x)    # neccesary ∵ bug atm in mocapio which triples length of output
    traj_mat = hcat(trj[1][ix_rng], trj[2][ix_rng])

    # Do RDP transform and extract knots
    simplified = rdp(traj_mat, ϵ)
    ts = map(1:size(x,1)) do i
            findfirst((x[i,1] .≈ traj_mat[:,1]) .& (x[i,2] .≈ traj_mat[:,2]))
    end

    # Find additional turning points by looking at large large second "derivatives" (differences)
    θ = atan.(root_x, root_z);
    thrsh = file_ix > 0 ? get(θ2d_thrsh, file_ix, 0.3) : 0.3
    knots = abs.(diff(vcat(0, θ))) .> thrsh;

    # add/remove custom points specified above
    if file_ix > 0
        rm_knots = vcat(1, get(rm_pts, file_ix, []))   # always rm initial knot
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
        knots_ix = findall(merge_knots(knots; take_first=false, merge_triplets=false));
    end

    # Calculate spline coefficients via OLS
    N = length(root_x)

    knots = knots_ix .- 1
    eval_ts = 0:(N-1)
    splBasis = AxUtil.Math.bsplineM(eval_ts, knots, 3+1)
    derivBasis = AxUtil.Math.bsplineM(eval_ts, knots, 3+1, 1)
    # deriv2Basis = AxUtil.Math.bsplineM(eval_ts, knots, 3+1, 2)  # 2nd deriv not used

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
        sharp_turns[max(0, ix-turn_window):min(N, ix+turn_window)] .= true
    end

    return smoothed_trj, findall(sharp_turns), g[sharp_turns]
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
