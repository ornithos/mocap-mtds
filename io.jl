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
"""
function process_file(filename; smooth_footc::Int=0)

    anim, names, frametime = BVHpy.load(filename)

    # Subsample to 60 fps
    anim = get(anim,  range(0, length(anim)-1, step=2))

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
"""
function construct_inputs(raw; direction=:relative, joint_pos=true)
    X = reconstruct_raw(raw)
    construct_inputs(X, raw; direction=direction, joint_pos=joint_pos)
end

function construct_inputs(X, raw; direction=:relative, joint_pos=true)
    @argcheck direction in [:relative, :absolute]
    @argcheck size(X)[2:3] == (21, 3)
    @argcheck size(raw, 2) == 196
    @argcheck size(X, 1) == size(raw, 1)

    # proc = (N-2)   *  [ rvel (1), xvel (1), zvel (1), feet (4),  pos (63),  vel (63),  rot (63) ]
    use_ixs = range(70, stop=size(X, 1) - 60)
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

    # Extract -60:10:59 trajectory on a rolling basis
    # ---------------------------------------
    for i in 1:12
        Xs[:,i]    = X[(10:N+9) .+ (i-1)*10, 1, 1]
        Xs[:,i+12] = X[(10:N+9) .+ (i-1)*10, 1, 3]
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
    traj_vel_xz = traj_pos_smooth[15:end, :] - traj_pos_smooth[5:end-10, :]
    rel_angle = hcat(reverse(_trigvecs(traj_vel_xz[1:end-5,:], forward[9:end-11, :]))...)  # sinθ, cosθ

    for (r, ix) in enumerate(use_ixs)
        cvel = view(traj_vel_xz, ix-69:10:ix+41, :)  # note v is +10--> due to +5/-5 differencing

        if direction == :relative
            cangle = view(rel_angle, ix-69:10:ix+41,:)
            Xs[r, 25:48] = vec(cangle)
        else
            cforward = view(forward, ix-60:10:ix+50, :)
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
    `include_ftcontact` (Bool)).
"""
function construct_outputs(raw; include_ftcontact=true)
    @argcheck size(raw, 2) == 196
    ixs = range(70, stop=size(raw, 1) - 60)
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


end   # module end
