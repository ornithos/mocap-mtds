module mocapviz

using Quaternions
using MeshCat
using CoordinateTransformations: LinearMap, AffineMap, Translation, rotation_between
using GeometryTypes: Cylinder, HyperSphere, Point
using LinearAlgebra: norm
using DSP   # conv
using Colors: RGBA, RGB
using Formatting, ArgCheck


const greymesh = MeshPhongMaterial(color=RGBA(0.3, 0.3, 0.3, 0.7))
const yellowmesh = MeshPhongMaterial(color=RGBA(204/255, 204/255, 0., 0.7))
const redmesh = MeshPhongMaterial(color=RGBA(224/255, 131/255, 94/255, 0.7))

# initialise lines and dots
mutable struct LineCollection{T, R}
    data::Array{T}
    vnames::Array{String}
    attributes::Array{R}
end
mutable struct JointCollection{T, R}
    data::Array{T}
    vnames::Array{String}
    attributes::Array{R}
end

Base.length(x::LineCollection) = length(x.data)
Base.length(x::JointCollection) = length(x.data)

function update_pos!(lines::LineCollection, origin::Matrix, extremity::Matrix)
    @argcheck length(lines) == size(origin, 1) == size(extremity, 1)
    lines.data = [Cylinder(Point{3}(origin[i,:]), Point{3}(extremity[i,:]), l.r)
                  for (i,l) in enumerate(lines.data)]
end

function update_pos!(lines::LineCollection, data::Matrix, parents::Vector)
    parents = parents[1:length(lines)]
    parent = data[parents,:]
    data = data[2:end, :]  # remove root (has no line)
    update_pos!(lines, parent, data)
end

function update_pos!(joints::JointCollection, data::Matrix)
    @argcheck length(joints) == size(data, 1)
    joints.data = [HyperSphere(Point{3}(data[i,:]), l.r) for (i,l) in enumerate(joints.data)]
end

# 3D plotting utilities: map line from x₁ → x₂ into affine transformation of e₁.
function line_into_transform(x1, x2)
    scaling, position, rotation = line_into_transforms_indiv(x1, x2)
    return AffineMap(rotation, position) ∘ LinearMap(diagm(0=>scaling))
end

function line_into_transforms_indiv(x1, x2)
    scaling = [norm(x2-x1), 1., 1]
    position = x1
    rotation = rotation_between([1., 0, 0], x2 - x1)
    return scaling, position, rotation
end

line_into_transform(x::Cylinder) = line_into_transform(x.origin, x.extremity)
line_into_transforms_indiv(x::Cylinder) = line_into_transforms_indiv(x.origin, x.extremity)

function setobj_collection!(v::AbstractVisualizer, L::LineCollection, J::JointCollection)
    # lines at back, joints at front
    for i in 1:length(L)
        setobject!(v[L.vnames[i]], L.data[i], L.attributes[i])
    end
    for i in 1:length(J)
        setobject!(v[J.vnames[i]], J.data[i], J.attributes[i])
    end
end

function settransform_collection!(v::AbstractVisualizer, L::LineCollection, J::JointCollection, anim=false)
    # lines at back, joints at front
    if anim
        for i in 1:length(L)
            scaling, position, rotation = line_into_transforms_indiv(L.data[i])
            anim_settransform!(v[L.vnames[i]], scaling, position, rotation)
        end
    else
        for i in 1:length(L)
            tform = line_into_transform(L.data[i])
            settransform!(v[L.vnames[i]], tform)
        end
    end
    for i in 1:length(J)
        settransform!(v[J.vnames[i]], Translation(J.data[i].center))
    end
end

# HACK: alternative method for rotation∘scale because MeshCat *animation* lowering is broken
function anim_settransform!(vis::MeshCat.AnimationFrameVisualizer, scaling, position, rotation)
    clip = MeshCat.getclip!(vis)
    MeshCat._setprop!(clip, vis.current_frame, "scale", "vector3", scaling)
    MeshCat._setprop!(clip, vis.current_frame, "position", "vector3", position)
    MeshCat._setprop!(clip, vis.current_frame, "quaternion", "quaternion", MeshCat.js_quaternion(rotation))
end


function create_animation(data::Vector, names::Union{String, Array{String}}="dataset";
    vis=nothing, parents=[1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20],
    jointmesh::Union{AbstractMaterial, Vector{T} where T <: AbstractMaterial}=greymesh,
    linemesh::Union{AbstractMaterial, Vector{T} where T <: AbstractMaterial}=yellowmesh, scale=0.1,
    camera::Symbol=:front)

    Ts = [size(d,1) for d in data]
    ls = [size(d,2) for d in data]
    n  = length(ls)
    (names isa String) && (names = [names * string(i) for i in 1:n])
    (jointmesh isa AbstractMaterial) && (jointmesh = [jointmesh for i in 1:n])
    (linemesh isa AbstractMaterial) && (linemesh = [linemesh for i in 1:n])
    @argcheck length(data) == length(names)
    @assert (all([ndims(d) for d in data] .== 3) && all([size(d,3) for d in data] .==3)) "Need NxJx3 matrices."
    @argcheck camera in [:front, :back, :static]
    # @assert all(ls .<= length(parents)-1) "more joints given (dim 1) than are assigned parents."
    # any(ls .> length(parents)) && @warn format("({:d}/{:d}) datasets have fewer joints than specified in parents.",
    #     sum(ls .> length(parents)), n)

    if vis === nothing
        vis = Visualizer()
        open(vis)
    else
        @assert (vis isa Visualizer) "Don't know what to do with vis::$(typeof(vis)). Expecting Visualizer."
    end

    dotstr(i) = "dot_" * format("{:02d}", i);
    linestr(i) = "line_" * format("{:02d}", i);

    # Initialise objects for each dataset
    objs = map(1:n) do i
        nj, nl = ls[i], ls[i]-1

        objs = (LineCollection([Cylinder(zero(Point{3, Float64}), Point{3}([1.,0,0]), 0.03)
                                for j in 1:nl], [linestr(j) for j in 1:nl], [linemesh[i] for j in 1:nl]),
        JointCollection([HyperSphere(Point{3}(0.), 0.04) for j in 1:nj],
                                [dotstr(j) for j in 1:nj], [jointmesh[i] for j in 1:nj]))

        setobj_collection!(vis[names[i]], objs[1], objs[2])
        objs
    end


    anim = Animation()
    maxT = maximum(Ts)

    cam_ix = 1
    pos = hcat([data[cam_ix][tt,1,1]*scale for tt in 1:maxT], [data[cam_ix][tt,1,3]*scale for tt in 1:maxT])
    pos = hcat(conv(pos[:,1], Windows.rect(4)/4), conv(pos[:,2], Windows.rect(4)/4))[1:maxT,:]
    cc = [scale, scale] * (camera == :front ? -1 : +1)

    reord = [1,3,2] # need to permute z/y axis for three.js setup
    for tt in 1:maxT
        atframe(anim, vis, tt-1) do frame
            for i in 1:n
                if Ts[i] >= tt
                    lines, joints = objs[i]
                    update_pos!(lines, data[i][tt,:,reord]*scale, parents)
                    update_pos!(joints, data[i][tt,:,reord]*scale);
                    settransform_collection!(frame[names[i]], lines, joints, true)

                    if camera != :static
                        settransform!(frame["/Cameras/default"], Translation(pos[tt,1] + cc[1], pos[tt,2] + cc[2], 1))
                    end
                end
            end
        end
    end

    setanimation!(vis, anim)
    return vis
end

end
