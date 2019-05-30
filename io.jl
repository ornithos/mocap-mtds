module mocapio

using LinearAlgebra, Statistics, Random
using Quaternions    # For manipulating 3D Geometry
using ProgressMeter, Formatting, ArgCheck # small utils libraries


# Read a string from STDIN. The trailing newline is stripped. https://stackoverflow.com/a/23342933
function input(prompt::String="")::String
   print(prompt)
   return chomp(readline())
end

# ##############################################################################
# Python interface to reuse Dan Holden's code
# ##############################################################################

using PyCall
pysys = pyimport("sys")

# add python directory to python path if available
pushfirst!(PyVector(pysys."path"), joinpath("."));
if isdir(joinpath(pwd(), "pyfiles/"))
    pushfirst!(PyVector(pysys."path"), joinpath(pwd(), "pyfiles"));
end

# check we can find the relevant py files.
while !any([isfile(joinpath(p, "BVH.py")) for p in pysys."path"])
    @warn "Dan Holden python processing scripts not found."
    println("Please run mocapio from a directory which contains these files,\n" *
             "or contains a 'pyfiles' directory with them.")
    println("\nAlternatively please provide a path on which python can find them:")
    userinput = chomp(readline())
    if isdir(userinput)
        pushfirst!(PyVector(pysys."path"), userinput);
    else
        throw(ErrorException("Cannot find file BVH.py"))
    end
end

# imports
BVHpy = pyimport("BVH");
Animpy = pyimport("Animation");
Quatpy = pyimport("Quaternions");
Pivotspy = pyimport("Pivots")
filterspy = pyimport("scipy.ndimage.filters")   # note scipy not dl by default by Julia


# PyCall Utility Functions
_toArray(x) = convert(Array, x)   # Quatpy class operations return Python objects (numpy). Forces convert.
_collapseDim3Jl(x) = reshape(x, size(x, 1), :)
_collapseDim3Npy(x) = (x=permutedims(x, [1,3,2]); reshape(x, size(x, 1), :);)

# class method __getitem__ def in Quaternions.py doesn't play nicely. Let's write our own:
_QuatpyGetItem(pyo, ixs...) = Quatpy.Quaternions(pyo.qs[(ixs..., :, :, :)[1:ndims(pyo.qs)]...])
# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-


# ##############################################################################
# Julia utility functions
# ##############################################################################

# global utils
_squeeze(x) = dropdims(x, dims=tuple(findall(size(x) .== 1)...))  # may be an official fn, but not sure where
_unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...))  # <= from Flux.jl
_rowmaj_reshape_3d(x, ix, iy, iz) = (x=reshape(x, ix, iz, iy); permutedims(x, [1,3,2]);)

# Quaternion related utils
_qimag = Quaternions.imag
_quat_list(x) = [quat(x[i,:]) for i in 1:size(x,1)]
_quat_list_to_mat(x) = reduce(vcat, [_qimag(xx)' for xx in x])
_quaterion_angle_axis_w_y(θ) = quat(cos(θ/2), 0, sin(θ/2), 0)
_apply_rotation(x, qrot) = qrot * x * conj(qrot)


# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-


"""
    bvh_frames_approx(filename(s))

Count the expected number of frames in (array of) BVH file(s) without loading
them. The calculation is based on linear regression between file length and
number of frames in a collection of processed files. Due to different whitespace
etc. the answer is only approximate, but it is within ± 2 frames in the
200 files it was trained on.

# Examples
```jldoctest
julia> a = bvh_frames_approx("foo.bvh")
1-element Array{Int64,1}:
 2946

julia> a = bvh_frames_approx(["foo.bvh", "bar.bvh"])
2-element Array{Int64,1}:
 2946
 3256
```
"""
bvh_frames_approx(files::String) = bvh_frames_approx([files])

function bvh_frames_approx(files::Array)
    # Approximate, based on LR to countlines->processed
    # but it's very high accuracy
    map(files) do f
        open(f) do file
            Int(floor(countlines(file)*0.5 - 95))
        end
    end
end


"""
    _trigvecs(A, B)

This is the row-wise signed angle between two 3D vectors a_{i:} and b_{i:}. Some
simplifications have been made since it is only used in the context where the
second co-ordinate (vertical) is zero. Hence it assumes both input matrices have
two columns, and allows us to simplify the cross product considerably. Note that
this is more challenging than simply using the cosine rule, because we need the
*signed* angle, which is usually not given. This version uses
https://stackoverflow.com/a/33920320 which was neater than my original version.
"""
function _trigvecs(A::Matrix{T}, B::Matrix{T}) where T <: Number
    inner_prod(X, Y) = sum(X .* Y, dims=2)
#     cross_rows(X, Y) = [cross(view(A, i,:), view(B, i,:))' for i in 1:size(A,1)] |>  x->reduce(vcat, x)
    norm_rows(X) = sqrt.(sum(x->x^2, X, dims=2))
    norm_A, norm_B = norm_rows(A), norm_rows(B)
    !all(isapprox.(norm_A, one(T), atol=1e-4)) && (A = A ./ norm_A)
    !all(isapprox.(norm_B, one(T), atol=1e-4)) && (B = B ./ norm_B)
    cosθ = inner_prod(A, B)

    # **SIGNED** angle θ https://stackoverflow.com/a/33920320
#     θ = atan2(inner_prod(cross_rows(B, A), Vn), cosθ)
    # and hence sinθ = inner_prod(cross_rows(B, A), Vn)
    # since y is our normal vector, and this component of A, B is
    # always zero, we can simplify
    sinθ = A[:,1] .* B[:,2] - A[:,2] .* B[:,1]
    return cosθ, sinθ
end


"""
    process_file(filename)

Load file of type bvh (mocap data) into raw format used by Holden and Mason.
Data are presumed to be targeted to a normalized 31-joint skeleton consonant
with the CMU dataset (I have not thought about what difference an unnormalized
input would make since all my data has been re-targeted beforehand. Caveat
emptor). The output will be an (N x 196) Matrix with the following columns:

* (1) Root rotational velocity,
* (2) Root (x,z) velocity,
* (4) Foot ground contacts (L Heel, L Toe, R Heel, R Toe),
* (63) Joint positions (Lagrangian frame, z=forward, x=orthogonal, y=vertical)
* (63) Joint velocities (Lagrangian frame, z=forward, x=orthogonal, y=vertical)
* (63) Joint rotations (?? I've never used these).

The named argument `smooth_footc` may be used to smooth the input prior to
performing the thresholding Dan Holden applies to extract foot contacts. This
is not done in the original code, and hence its default is 0. The (Int) value
corresponds to the filter width of the smoother. I recommend ≤ 5.

The default frame rate is 60 fps which is used by Holden and Mason (and
presumably the vision community). However, it may be more convenient for ML work
to use fps=30. A frame rate in {30,60,120} may be supplied via `fps=...`. Note
that this should also be supplied to inputs and outputs for the traj. resolution.
"""
function process_file(filename; fps::Int=60, smooth_footc::Int=0)

    @argcheck fps in [30,60,120]
    anim, names, frametime = BVHpy.load(filename)

    # Subsample to 60 fps
    fps_step = Dict(120=>1, 60=>2, 30=>4)[fps]
    anim = get(anim,  range(0, length(anim)-1, step=fps_step))

    # Do FK
    global_xforms = Animpy.transforms_global(anim)  # intermediate
    global_positions = global_xforms[:,:,1:3,4] ./ global_xforms[:,:,4:end,4]
    global_rotations = Quatpy.Quaternions.from_transforms(global_xforms)


    # Remove Uneeded Joints
    used_joints = [0, 2,  3,  4,  5, 7,  8,  9, 10, 12, 13, 15, 16, 18, 19, 20, 22, 25, 26, 27, 29] .+ 1

    positions = global_positions[:, used_joints,:]
    global_rotations = _QuatpyGetItem(global_rotations,:,used_joints,:)
    N = size(positions, 1)
    # ________________________________________________________

    # Put on Floor
    positions[:,:,2] .-= minimum(positions[:,:,2])

    # Get Foot Contacts
    velfactor, heightfactor = [0.05,0.05], [3.0, 2.0]
    fid_l, fid_r = [3,4] .+1, [7,8] .+1

    pos_l = positions[:, fid_l, 1:3]
    (smooth_footc > 0) && (pos_l = filterspy.gaussian_filter1d(pos_l, smooth_footc, axis=0, mode="nearest"))
    feet_l_vmag_sq = sum(x->x^2, diff(pos_l, dims=1), dims=3) |> _squeeze
    feet_l_h = pos_l[1:end-1, :, 2]
    feet_l = (feet_l_vmag_sq .< velfactor') .& (feet_l_h .< heightfactor')

    pos_r = positions[:, fid_r, 1:3]
    (smooth_footc > 0) && (pos_r = filterspy.gaussian_filter1d(pos_r, smooth_footc, axis=0, mode="nearest"))
    feet_r_vmag_sq = sum(x->x^2, diff(pos_r, dims=1), dims=3) |> _squeeze
    feet_r_h = pos_r[1:end-1, :, 2]
    feet_r = (feet_r_vmag_sq .< velfactor') .& (feet_r_h .< heightfactor')

    # Get Root Velocity
    velocity = diff(positions[:,1:1,:], dims=1)

    # Remove translation
    positions[:,:,1] .-= positions[:,1:1,1]
    positions[:,:,3] .-= positions[:,1:1,3]

    # Get Forward Direction
    sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6  #13, 17, 1, 5
    across1 = positions[:,hip_l,:] - positions[:,hip_r,:]
    across0 = positions[:,sdr_l,:] - positions[:,sdr_r,:]
    across = across0 + across1
    across = across ./ sqrt.(sum(x->x^2, across, dims=2))

    direction_filterwidth = 20
#     forward = [cross(view(across, i,:), [0., 1, 0])' for i in 1:N] |>  x->reduce(vcat, x)  # crossprod
    forward = hcat(-across[:,3], zeros(size(across, 1), 1), across[:,1])  # crossprod (easy as spec. case)
    forward = filterspy.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode="nearest")
    forward = forward ./ sqrt.(sum(x->x^2, forward, dims=2))

    # Get Root Rotation
    target = repeat([0,0,1]', N, 1)
    root_rotation = Quatpy.Quaternions.between(forward, target)
    root_rotation.qs = _unsqueeze(root_rotation.qs, 2);
    root_rot_omitlast = _QuatpyGetItem(root_rotation, 1:(N-1))
    rvelocity = (_QuatpyGetItem(root_rotation, 2:N) * -root_rot_omitlast).to_pivots()

    # Local Space  # NEW: define position of joints relative to root
    local_positions = positions  # copy(positions)
    local_positions[:,:,1] .-= local_positions[:,1:1,1]  # x rel to root x
    local_positions[:,:,3] .-= local_positions[:,1:1,3]  # z rel to root z

    local_positions = root_rot_omitlast * local_positions[1:end-1,:,:]  |> _toArray # remove Y rotation from pos
    local_velocities = diff(local_positions, dims=1)
    local_rotations = abs((root_rot_omitlast * _QuatpyGetItem(global_rotations, 1:(N-1)))).log()

    root_rvelocity = Pivotspy.Pivots.from_quaternions(_QuatpyGetItem(root_rotation, 2:N) * -root_rot_omitlast).ps
    global_velocities = root_rot_omitlast * velocity    |> _toArray                # remove Y rotation from vel


    @assert (size(global_velocities, 2) == 1) "output assumes global_velocities dim2 = 1."
    omit_end = 1:(N-2)
    out = hcat(root_rvelocity[omit_end,:]
                ,global_velocities[omit_end,1,1]
                ,global_velocities[omit_end,1,3]
                ,feet_l[omit_end,:], feet_r[omit_end,:]
                ,_collapseDim3Npy(local_positions[omit_end,:,:])
                ,_collapseDim3Npy(local_velocities)
                ,_collapseDim3Npy(local_rotations[omit_end,:,:])
#                 ,forward[omit_end,:]
        )
    return out
end




"""
    reconstruct_raw(Y)

This reconstructs the absolute positions of joints in a contiguous set of frames
from the raw matrix output of `process_file`. This proceeds by applying forward
kinematics using the root rotation from the Lagrangian representation, and the
x-z root velocities of the first few dims of the processed matrix.
"""
function reconstruct_raw(Y::Matrix)
    Y = convert(Matrix{Float64}, Y)   # reduce error propagation from iterative scheme

    root_r, root_x, root_z, joints = Y[:,1], Y[:,2], Y[:,3], Y[:,8:(63+7)]
    return _joints_fk(joints, root_x, root_z, root_r)
end

function _joints_fk(joints::Matrix{T}, root_x::Vector{T}, root_z::Vector{T},
        root_r::Vector{T}) where T <: Number

    n = size(joints, 1)
    joints = _rowmaj_reshape_3d(joints, n, 21, 3)
#     joints = reshape(joints, n, 3, 21)
#     joints = permutedims(joints, [1,3,2])
    rotation = Quaternion(1.0)
    offsets = []
    translation = zeros(3)

    for i = 1:n
        joints[i,:,:] = _apply_rotation(_quat_list(joints[i,:,:]), rotation) |> _quat_list_to_mat
        joints[i,:,1] = joints[i,:,1] .+ translation[1]
        joints[i,:,3] = joints[i,:,3] .+ translation[3]

        rotation = _quaterion_angle_axis_w_y(-root_r[i]) * rotation
        append!(offsets, _apply_rotation(quat(0.,0,0,1), rotation))
        translation = translation + _qimag(_apply_rotation(quat(0., root_x[i], 0, root_z[i]), rotation))
    end

    return joints
end


"""
    construct_inputs(raw [; direction=:relative, joint_pos=true])

Construct the input matrix for the mocap models. The input `raw` is the raw
output from the `process_file` function. The function outputs the following
matrix, which contains only the range of frames: [start+69, end-60] (i.e.)
excluding approx. the first and last second. This is in order to construct
trajectories that extend ± 60 frames of the current position. (The additional is
due to needing a bit extra to calculate velocity, plus some historical baggage.)
Note also that the trajectory is centered at every frame at the current position
and hence `(trajectory_x(7), trajectory_z(7)) == (0.0, 0.0)`.


The following columns are contained in the matrix:

* (12): ± 60 frame trajectory x-cood at step 10 intervals
* (12): ± 60 frame trajectory z-cood at step 10 intervals
* (12): ± 60 frame trajectory angle sin(θ) to forward
* (12): ± 60 frame trajectory angle cos(θ) to forward
* (12): ± 60 frame trajectory magnitude of velocity
* (61): joint positions in Lagrangian frame (optional)

The angle θ is expressed in both sine and cosine components to avoid a
discontinuity when it wraps around 2π (which it sometimes does). This angle
is Lagrangian in nature too: that is, θ = 0 when the skeleton is facing in
exactly the same direction as it is walking (i.e. the direction of the
velocity). One might prefer a Eulerian (absolute) representation instead, in
which case, pass in the named argument `direction=:absolute`.

Note that there are only 61 dimensions of the joint positions as the root x,z
are excluded, as they are always zero. They're excluded from the output too,
which is more important: we don't want to waste strength on predicting zero. In
most of my experiments, I have found that including the joint positions in the
input tends to make it too easy for the model to obtain trivial predictions. To
avoid returning any joint_positions in the input matrix, select:

    joint_pos=false

If `fps=30` or `fps=120` used in processing, this should also be supplied.
"""
function construct_inputs(raw; direction=:relative, joint_pos=true, fps::Int=60)
    X = reconstruct_raw(raw)
    construct_inputs(X, raw; direction=direction, joint_pos=joint_pos, fps=fps)
end

function construct_inputs(X, raw; direction=:relative, joint_pos=true, fps::Int=60)
    @argcheck direction in [:relative, :absolute]
    @argcheck fps in [30,60,120]
    @argcheck size(X)[2:3] == (21, 3)
    @argcheck size(raw, 2) == 196
    @argcheck size(X, 1) == size(raw, 1)

    # proc = (N-2)   *  [ rvel (1), xvel (1), zvel (1), feet (4),  pos (63),  vel (63),  rot (63) ]
    ϝ = fps/60
    t₀, t₁, Δt = Int(70 * ϝ),  size(X, 1) - Int(60 * ϝ), Int(10 * ϝ)
    use_ixs = range(t₀, stop=t₁)
    N = length(use_ixs)
    T = eltype(raw)

    # traj_pos (12x2), abs./rel. direction (12x2), abs. velocity (12), joint positions (63)
    jpnum = joint_pos ? 61 : 0
    Xs = Matrix{T}(undef, N, 48 + 12 + jpnum)

    # add rel. pos from raw
    if joint_pos
        Xs[:, 61] = raw[use_ixs,9]          # x,z value of root are always zero
        Xs[:, 62:end] = raw[use_ixs,11:70]
    end

    # Extract -60:10:59 (or -30:5:29 etc.) trajectory on a rolling basis
    # ---------------------------------------
    for i in 1:12
        Xs[:,i]    = X[(Δt:N+Δt-1) .+ (i-1)*Δt, 1, 1]
        Xs[:,i+12] = X[(Δt:N+Δt-1) .+ (i-1)*Δt, 1, 3]
    end
    Xs[:,1:12] .-= Xs[:,7]
    Xs[:,13:24] .-= Xs[:,19]

    # convert Euler trajectory to Lagrangian frame
    prev_dir = hcat(-Xs[:,6], -Xs[:,18])   # ∵ [7] == 0, hence diff is -[6]
    cθ, sθ   = _trigvecs(prev_dir, hcat(zeros(T, N, 1), ones(T, N, 1))) # calc angle to z-axis (2nd cood)
    cθ, sθ   = cθ, sθ  # reverse sign of θ (note cos is unchanged ∵ even fn)
    euler_traj  = Xs[:,1:24]  # implicit copy
    Xs[:,1:12]  = euler_traj[:,1:12] .* cθ - euler_traj[:,13:24] .* sθ
    Xs[:,13:24] = euler_traj[:,1:12] .* sθ + euler_traj[:,13:24] .* cθ


    # Calculate body direction and velocity
    # ---------------------------------------
    # Calculate forward direction (same as process_file, but different reference frame)
    sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6  #13, 17, 1, 5
    normalize_rows(x) = x ./ sqrt.(sum(z->z^2, x, dims=2))
    across = (X[:,sdr_l,:] - X[:,sdr_r,:]) + (X[:,hip_l,:] - X[:,hip_r,:]) |> normalize_rows
    forward = hcat(-across[:,3], zeros(size(across, 1), 1), across[:,1])  # crossprod (easy as spec. case)
    forward = filterspy.gaussian_filter1d(forward, 5, axis=0, mode="nearest")[:,[1,3]] |> normalize_rows

    # Calculate diff/velocity (i.e. instantaneous dir of traj) / compare with body forward dir
    traj_pos_smooth = filterspy.gaussian_filter1d(X[:,1,[1,3]], 5, axis=0, mode="nearest")
    Δ_½  = Dict(120=>10, 60=>5, 30=>2)[fps]
    Δ_½r = Dict(120=>0, 60=>0, 30=>1)[fps]
    # traj_pos_smooth[15:end, :] - traj_pos_smooth[5:end-10, :]
    traj_vel_xz = traj_pos_smooth[Δt+Δ_½:end, :] - traj_pos_smooth[Δ_½:end-Δt, :]
    # traj_vel_xz[1:end-5,:], forward[9:end-11, :]
    # display(traj_vel_xz[1:end-Δ_½,:])
    # display(forward[(Δt-1):(end-Δt+Δ_½r-1), :])
    rel_angle = hcat(reverse(_trigvecs(traj_vel_xz[1:end-Δ_½,:], forward[(Δt-1):(end-Δt+Δ_½r-1), :]))...)  # sinθ, cosθ

    # display("Rel angle2     ")
    # display(size(rel_angle))
    for (r, ix) in enumerate(use_ixs)
        v_ixs = (ix+Int(-70 * ϝ)+1):Δt:(ix+Int(40 * ϝ)+1) # note vel is +10--> due to +5/-5 differencing
        cvel = view(traj_vel_xz, v_ixs, :)

        if direction == :relative
            cangle = view(rel_angle, v_ixs,:)
            Xs[r, 25:48] = vec(cangle)
        else
            a_ixs = (ix+Int(-60 * ϝ)):Δt:(ix+Int(50 * ϝ))
            cforward = view(forward, a_ixs, :)
            Xs[r, 25:48] = vec(cforward)
        end

        Xs[r, 49:60] = sqrt.(sum(x->x^2, cvel, dims=2)[:])
    end
    return Xs
end


"""
    construct_outputs(raw [; include_ftcontact])

Construct the output matrix for the mocap models. The input `raw` is the raw
output from the `process_file` function. The function outputs the following
matrix, which contains only the range of frames: [start+69, end-60] (i.e.)
excluding approx. the first and last second. This is in order to match with
the input matrix which needs these boundaries in order to construct trajectories
consistently. This function outputs a matrix with the following columns:

* (3): rotational velocity, x-velocity, z-velocity
* (61): (Lagrangian) joint positions (excl. root x/z ∵ always zero)
* (4): [optional] feet contacts

These are the minimal requirements to reconstruct an animation of human motion
on the target skeleton. The feet contacts are in {0,1} and may be challenging
for some linear models, therefore they are optional (see named argument
    `include_ftcontact`
(Bool)). Further, if a different frame rate is used for processing, a change in
trajectory resolution will be required, and hence the amount of padding will
change. To account for this difference in inputs, supply a `fps=` named
argument.

"""
function construct_outputs(raw; include_ftcontact=true, fps::Int=60)
    @argcheck size(raw, 2) == 196
    @argcheck fps in [30,60,120]
    ϝ = fps/60
    t₀, t₁, Δt = Int(70 * ϝ),  size(raw, 1) - Int(60 * ϝ), Int(10 * ϝ)

    ixs = range(t₀, stop=t₁)
    if !include_ftcontact
        return reduce(hcat, (raw[ixs, 1:3], raw[ixs, 9:9], raw[ixs, 11:(63+7)]))
    else
        return reduce(hcat, (raw[ixs, 1:3], raw[ixs, 9:9], raw[ixs, 11:(63+7)],
                             raw[ixs, 4:7]))
    end
end


"""
    reconstruct_modelled(Y)

This reconstructs the absolute positions of joints in a contiguous set of frames
from a model matrix with the same columns as the output of `construct_outputs`.
This proceeds by applying forward kinematics using the root rotation from the
Lagrangian representation, and the x-z root velocities of the first few dims of
the `Y` matrix.
"""
function reconstruct_modelled(Y::Matrix)
    Y = convert(Matrix{Float64}, Y)   # reduce error propagation from iterative scheme
    T = eltype(Y)
    N = size(Y, 1)

    root_r, root_x, root_z, joints = Y[:,1], Y[:,2], Y[:,3], Y[:,5:(60+4)]
    rootjoint = reduce(hcat,  (zeros(T, N, 1), Y[:,4:4], zeros(T, N, 1)))
    joints = hcat(rootjoint, joints)
    return _joints_fk(joints, root_x, root_z, root_r)
end


#= -----------------------------------------------------------------------
                Utilities for accessing data
   ----------------------------------------------------------------------- =#

mutable struct ExperimentData{MT <: AbstractMatrix}
    YsRaw::Vector{MT}
    Ys::Vector{MT}
    Us::Vector{MT}
    ix_lkp
    function ExperimentData(YsRaw, Ys, Us, ix_lkp)
        MT = unique([typeof(x) for x in YsRaw])[1]
        YsRaw = [y[2:end,:] for y in YsRaw]  # note that Y, U take 2:end, 1:end-1.
        YsRaw = convert(Vector{MT}, YsRaw)
        Ys = convert(Vector{MT}, Ys)
        Us = convert(Vector{MT}, Us)
        new{MT}(YsRaw, Ys, Us, ix_lkp)
    end
end


"""
    get_data(s::ExperimentData, ix, splittype, tasktype)

Convenience utility for accessing data stored in an ExperimentData struct.
Specify the index of the target task, and then select from:

splittype:
* **:all**        - return the concatentation of all training/validation/test data.
* **:trainvalid** - return the concatentation of all training/validation data.
* **:split**      - return individual (3x) outputs for training/validation/test data.
* **:test**       - return only the test data
* **:train**      - return only the train data
* **:valid**      - return only the validation data.

tasktype:
* **:stl**   - single task model. Return train/validation/test data from this task's data.
* **:pool**  - pooled/cohort model. Here, training and validation data are from the
         complement of the selected index, returned in individual wrappers.

Note that in all cases, the output will be (a) Dict(s) containing the following fields:

* **:Y**    - the observation matrix (each column is an observation).
* **:U**    - the input matrix (each column is a datapoint).
* **:Yraw** - the raw data before standardisation and another manipulation. (Possibly legacy?)

Othe kwargs:

* `concat`  - By default, each boundary encountered between files will result in
a separate Dict, so the return values will be a vector of Dicts. However, for
more basic models (such as linear regression) with no assumption of temporal
continuity, it may be simpler to operate on a standard input and output data
matrix. Setting `concat = true` will return just a single Dict in an array with
all the data. Choosing `simplify=true` will further remove the array, returning
only Dicts.
* `stratified`  - (STL only) stratify the validation/test sets across files in
each style. By default, the test set will come at the end of the concatenation
of all files. Stratifying will mean there are L test sets from each of L files.
For the pooled dataset, the test set is partially stratified, that is, it is
stratified over the *types* (i.e. a % of each style), but not over the *files*
within the types. Given that our goal is MTL, this seems appropriate.
* `split`  - The train/validation/test split as a simplicial 3-dim vector.
* `simplify` - See `concat`. Used without `concat` this option does nothing.
"""
function get_data(s::ExperimentData, ix::Int, splittype::Symbol, tasktype::Symbol;
        concat::Bool=false, stratified=false, split=[0.7,0.1,0.2], simplify::Bool=false)
    @argcheck splittype ∈ [:all, :trainvalid, :split, :test, :train, :valid]
    @argcheck tasktype ∈ [:stl, :pool, :pooled]
    @assert !(stratified && splittype != :stl) "stratified only available for STL. Pooled is 'semi-stratified'(!)"
    if tasktype == :stl
        get_data_stl(s, ix, splittype; concat=concat, stratified=stratified,
            split=split, simplify=simplify)
    else
        splitpool = split[1:2]./sum(split[1:2])
        get_data_pooled(s, ix, splittype; concat=concat, split=splitpool, simplify=simplify)
    end
end


function get_data_stl(s::ExperimentData, ix::Int, splittype::Symbol;
        concat::Bool=false, stratified=false, split=[0.7,0.1,0.2], simplify=false)
    @argcheck splittype ∈ [:all, :trainvalid, :split, :test, :train, :valid]
    ixs = s.ix_lkp[ix]
    stratified = false

    # Get STL data (needed for everything)
    cYsraw = s.YsRaw[ixs]
    cYs    = s.Ys[ixs]
    cUs    = s.Us[ixs]

    if splittype == :all
        if concat
            cYsraw, cYs, cUs = _concat(cYsraw, cYs, cUs, simplify);
        end
        return _create_y_u_raw_dict(cYs, cUs, cYsraw)
    end

    Ns     = [size(y, 2) for y in cYs]
    !stratified && begin; cYsraw, cYs, cUs = _concat(cYsraw, cYs, cUs, false); end

    train, valid, test = create_data_split(cYs, cUs, cYsraw, split);
    if !stratified && !concat
        train, valid, test = _unconcatDicts(train, valid, test, Ns)
    elseif concat
        train, valid, test = _concatDicts(train, valid, test)
        if simplify
            rmv(x) = (length(x) == 1) ? x[1] : x
            train, valid, test = rmv(train), rmv(valid), rmv(test)
        end
    end

    if splittype == :split
        return train, valid, test
    elseif splittype == :test
        return test
    elseif splittype == :train
        return train
    elseif splittype == :valid
        return valid
    elseif splittype == :trainvalid
        return _concatDict([train, valid])
    else
        error("Unreachable error")
    end
end


function get_data_pooled(s::ExperimentData, ix::Int, splittype::Symbol;
        concat::Bool=false, split=[0.875, 0.125], simplify=false)
    @argcheck splittype ∈ [:all, :split, :test, :train, :valid]
    @argcheck length(split) == 2

    # Get STL data (needed for everything)
    test = get_data_stl(s, ix, :all; concat=concat, stratified=false, simplify=simplify)

    if splittype == :test
        return test
    end

    train = Dict[]
    valid = Dict[]

    for i in setdiff(1:length(s.ix_lkp), ix)
        _train, _valid, _test = get_data_stl(s, i, :split; concat=concat,
            stratified=false, split=vcat(split, 0))
        train = vcat(train, _train)
        valid = vcat(valid, _valid)
    end

    if concat
        train, valid = _concatDicts(train), _concatDicts(valid)
        if simplify
            rmv(x) = (length(x) == 1) ? x[1] : x
            train, valid, test = rmv(train), rmv(valid), rmv(test)
        end
    end

    if splittype == :split
        return train, valid, test
    elseif splittype == :train
        return train
    elseif splittype == :valid
        return valid
    else
        error("Unreachable error")
    end

end

"""
    create_data_split(Y, U, Yraw, split=[0.7,0.1,0.2])

Create a partition of the data into train/validation/test components. The
default size of these partitions is 70%/10%/20% respectively. This can be
switched up by supplying the argument `split=[a, b, c]` s.t. sum(a,b,c) = 1.

It is assumed that *columns* of Y and U contain datapoints, and that *rows* of
Yraw contain datapoints. The output will be a Dict containing the following fields:

* **:Y**    - the observation matrix (each column is an observation).
* **:U**    - the input matrix (each column is a datapoint).
* **:Yrawv - the raw data before standardisation and another manipulation. (Possibly legacy?)
"""
function create_data_split(Y::Matrix{T}, U::Matrix{T}, Yraw::Matrix{T}, split=[0.7,0.1,0.2]) where T
    N = size(Y,2)
    @argcheck size(Y, 2) == size(U, 2)
    @argcheck sum(split) == 1
    int_split = zeros(Int, 3)
    int_split[1] = Int(round(N*split[1]))
    int_split[2] = Int(round(N*split[2]))
    int_split[3] = N - sum(int_split[1:2])
    split = vcat(0, cumsum(int_split), N) .+1
    train, valid, test = (Dict(:Y=>Y[:,split[i]:split[i+1]-1],
                               :U=>U[:,split[i]:split[i+1]-1],
                               :Yraw=>Yraw[split[i]:split[i+1]-1,:]) for i in 1:3)
    return train, valid, test
end

function create_data_split(Y::Vector{MT}, U::Vector{MT}, Yraw::Vector{MT},
    split=[0.7,0.1,0.2]) where MT <: AbstractMatrix
    @argcheck length(Y) == length(U) == length(Yraw)
    train, valid, test = Dict[], Dict[], Dict[]
    for i = 1:length(Y)
        _train, _valid, _test = create_data_split(Y[i], U[i], Yraw[i], split);
        push!(train, _train)
        push!(valid, _valid)
        push!(test, _test)
    end
    return train, valid, test
end


#= -----------------------------------------------------------------------
                Utilities for utilities for accessing data  :S
    This whole thing got way out of hand. I'm sure this can be tidied up =>
    is a function of setting out without a plan, under the assumption that
    "it wouldn't take long". You know what I'm talking about.
   ----------------------------------------------------------------------- =#

function _concat(Ysraw::Vector{MT}, Ys::Vector{MT}, Us::Vector{MT}, simplify::Bool) where MT <: AbstractMatrix
    v = simplify ? identity : x -> [x]
    return v(reduce(vcat, Ysraw)), v(reduce(hcat, Ys)), v(reduce(hcat, Us))
end

function _concat(trainsplit::Vector{D}, simplify::Bool) where D <: Dict
    v = simplify ? identity : x -> [x]
    Ysraw = reduce(vcat, [x[:Yraw] for x in trainsplit])
    Ys = reduce(hcat, [x[:Y] for x in trainsplit])
    Us = reduce(hcat, [x[:U] for x in trainsplit])
    return v(Ysraw), v(Ys), v(Us)
end

function _concatDicts(trainsplit::Vector{D}) where D <: Dict
    Ysraw, Ys, Us = _concat(trainsplit, true)
    return Dict(:Y=>Ys, :U=>Us, :Yraw=>Ysraw)
end

function _concatDicts(train::Vector{D}, valid::Vector{D}, test::Vector{D}) where D <: Dict
    train = _concatDicts(train)
    valid = _concatDicts(valid)
    test  = _concatDicts(test)
    return train, valid, test
end

function _unconcat(Yraw::AbstractMatrix{T}, Y::AbstractMatrix{T}, U::AbstractMatrix{T},
    breaks::Vector{I}) where {T <: Real, I <: Int}

    N = size(Y,2)
    @argcheck size(Y, 2) == size(U, 2) == size(Yraw, 1) == sum(breaks)
    nsplit = length(breaks)
    split = cumsum([0; breaks]) .+ 1
    Ys = [Y[:,split[i]:split[i+1]-1] for i in 1:nsplit]
    Us = [U[:,split[i]:split[i+1]-1] for i in 1:nsplit]
    Ysraw = [Yraw[split[i]:split[i+1]-1, :] for i in 1:nsplit]
    return Ysraw, Ys, Us
end

function _unconcatDict(d::Dict, breaks::Vector{I}) where I <: Int
    Ysraw, Ys, Us = _unconcat(d[:Yraw], d[:Y], d[:U], breaks)
    return [Dict(:Y=>y, :U=>u, :Yraw=>yraw) for (y,u,yraw) in zip(Ys, Us, Ysraw)]
end

# TODO: DRY fail here.

function _unconcatDicts(ds::Vector{D}, breaks::Vector{I}) where {D <: Dict, I <: Int}
    N = length(ds)
    Ls = [size(d[:Yraw], 1) for d in ds]
    @assert sum(Ls) == sum(breaks)
    bs  = copy(breaks)
    cbs = cumsum(breaks)
    out = Dict[]
    for nn in 1:N
        ix = something(findlast(cbs .<= Ls[nn]), 0)
        (ix > 0 && ix < length(bs) && cbs[ix] != Ls[nn]) && (ix += 1)
        ix = max(ix, 1)
        cbreaks = bs[1:ix]
        cbreaks[end] = Ls[nn] - (ix > 1 ? cbs[ix-1] : 0)
        bs = bs[ix:end]
        bs[1] -= cbreaks[end]
        cbs = cbs[ix:end] .- Ls[nn]

        out = vcat(out, _unconcatDict(ds[nn], cbreaks))
    end
    return out
end

function _unconcatDicts(train::Vector{D}, valid::Vector{D}, test::Vector{D},
    breaks::Vector{I}) where {D <: Dict, I <: Int}
    Ls = [sum([size(d[:Y], 2) for d in data]) for data in [train, valid, test]]
    bs  = copy(breaks)
    cbs = cumsum(breaks)
    out = []
    for (nn, data) in enumerate([train, valid, test])
        ix = something(findlast(cbs .<= Ls[nn]), 0)
        (ix > 0 && ix < length(bs) && cbs[ix] != Ls[nn]) && (ix += 1)
        ix = max(ix, 1)
        cbreaks = bs[1:ix]
        cbreaks[end] = Ls[nn] - (ix > 1 ? cbs[ix-1] : 0)
        bs = bs[ix:end]
        bs[1] -= cbreaks[end]
        cbs = cbs[ix:end] .- Ls[nn]
        push!(out, _unconcatDicts(data, cbreaks))
    end
    return out[1], out[2], out[3]
end


function _create_y_u_raw_dict(Ys::Vector{MT}, Us::Vector{MT}, Ysraw::Vector{MT}
    ) where MT <: AbstractMatrix
    [Dict(:Y=>y, :U=>u, :Yraw=>yraw) for (y,u,yraw) in zip(Ys, Us, Ysraw)]
end

function _create_y_u_raw_dict(Ys::AbstractMatrix, Us::AbstractMatrix, Ysraw::AbstractMatrix)
    [Dict(:Y=>Ys, :U=>Us, :Yraw=>Ysraw)]
end



end   # module end
