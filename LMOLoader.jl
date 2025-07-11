# LMOLoader.jl
module LMOLoader

export ObjectModel, Frames, Frame, load_scene

using FileIO, ImageIO, Images, MeshIO, JSON3
using Printf, ProgressMeter

const OBJECT_IDS = [1, 5, 6, 8, 9, 10, 11, 12]
const FRAME_IDS = collect(0:1213)
const MASK_TO_OBJECT_ID = Dict(
    0 => 1,   # 000000 -> object 1
    1 => 5,   # 000001 -> object 5
    2 => 6,   # 000002 -> object 6
    3 => 8,   # 000003 -> object 8
    4 => 9,   # 000004 -> object 9
    5 => 10,  # 000005 -> object 10
    6 => 11,  # 000006 -> object 11
    7 => 12   # 000007 -> object 12
)
const OBJECT_ID_TO_MASK = Dict(
    1 => 0,   # object 1 -> 000000
    5 => 1,   # object 5 -> 000001
    6 => 2,   # object 6 -> 000002
    8 => 3,   # object 8 -> 000003
    9 => 4,   # object 9 -> 000004
    10 => 5,  # object 10 -> 000005
    11 => 6,  # object 11 -> 000006
    12 => 7   # object 12 -> 000007
)

struct ObjectModel
    mesh  # GeometryBasics.Mesh
    diameter::Float32  # in meters
    min_x::Float32     # in meters
    min_y::Float32     # in meters
    min_z::Float32     # in meters
    size_x::Float32    # in meters
    size_y::Float32    # in meters
    size_z::Float32    # in meters

    function ObjectModel(model_file::String, model_info)
        # Load mesh using trimesh
        mesh = load(model_file)

        # Convert from millimeters to meters
        diameter = model_info[:diameter] / 1000.0
        min_x = model_info[:min_x] / 1000.0
        min_y = model_info[:min_y] / 1000.0
        min_z = model_info[:min_z] / 1000.0
        size_x = model_info[:size_x] / 1000.0
        size_y = model_info[:size_y] / 1000.0
        size_z = model_info[:size_z] / 1000.0

        return new(mesh, diameter, min_x, min_y, min_z, size_x, size_y, size_z)
    end
end

struct Frames
    rgbs::Array{Float32,4} # (frame x height x width x 3)
    depths::Array{Float32,3} # (frame x height x width) - in meters
    masks::BitArray{4} # (frame x objects x height x width)
    ks::Array{Float32,3} # (frame x 3x3) - focal lengths in pixels
    poses::Array{Float32,4} # (frame x objects x 4x4) - translation in meters
    valid_idxs::Vector{Int}
end

struct Frame
    rgb::Array{Float32,3} # (height x width x 3)
    depth::Array{Float32,2} # (height x width) - in meters
    mask::BitArray{3} # (objects x height x width)
    K::Array{Float32,2} # (3x3) - focal lengths in pixels
    pose::Array{Float32,3} # (objects x 4x4) - translation in meters
    idx::Int
end

"""
    load_depth_raw(filename)

Load depth image as raw 16-bit values, not normalized.
"""
function load_depth_raw(filename)
    img = load(filename)
    raw_values = reinterpret(UInt16, img)
    return map(Float32, raw_values)
end

function load_scene(dataset_root::String="datasets/lmo", scene_id::String="000002")
    objects = Dict{Int,ObjectModel}()
    models_dir = joinpath(dataset_root, "models")
    models_info = JSON3.read(joinpath(models_dir, "models_info.json"))

    for id in OBJECT_IDS
        model_file = joinpath(models_dir, @sprintf("obj_%06d.ply", id))
        model_info = models_info[id]
        objects[OBJECT_ID_TO_MASK[id]+1] = ObjectModel(model_file, model_info)
    end

    rgbs = Array{Float32,4}(undef, length(FRAME_IDS), 480, 640, 3)
    depths = Array{Float32,3}(undef, length(FRAME_IDS), 480, 640)
    masks = Array{Bool,4}(undef, length(FRAME_IDS), length(OBJECT_IDS), 480, 640)
    ks = Array{Float32,3}(undef, length(FRAME_IDS), 3, 3)
    poses = Array{Float32,4}(undef, length(FRAME_IDS), length(OBJECT_IDS), 4, 4)
    invalid_frames = Set{Int}()

    scene_dir = joinpath(dataset_root, "scenes", scene_id)
    rgb_dir = joinpath(scene_dir, "rgb")
    depth_dir = joinpath(scene_dir, "depth")
    mask_dir = joinpath(scene_dir, "mask_visib")
    scene_camera = JSON3.read(joinpath(scene_dir, "scene_camera.json"))
    scene_gt = JSON3.read(joinpath(scene_dir, "scene_gt.json"))

    @printf "Loading scene %s\n" scene_id
    @showprogress for (frame_idx, frame_id) in enumerate(FRAME_IDS)
        if length(scene_gt[frame_id]) < length(OBJECT_IDS)
            push!(invalid_frames, frame_idx)
            continue
        end

        # Load RGB image
        rgb = load(joinpath(rgb_dir, @sprintf("%06d.png", frame_id)))
        rgbs[frame_idx, :, :, :] .= map(Float32, permutedims(channelview(rgb), (2, 3, 1)))

        # Load depth image as raw values and convert to meters
        depth_file = joinpath(depth_dir, @sprintf("%06d.png", frame_id))
        depth_raw = load_depth_raw(depth_file)
        depth_scale = scene_camera[frame_id][:depth_scale] / 1000.0  # convert from mm to meters
        depths[frame_idx, :, :] .= depth_raw .* depth_scale

        # Load camera intrinsics (in pixels) - Fix the transpose issue
        cam_K = scene_camera[frame_id][:cam_K]
        # cam_K is stored as row-major [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        ks[frame_idx, :, :] .= [cam_K[1] cam_K[2] cam_K[3];
            cam_K[4] cam_K[5] cam_K[6];
            cam_K[7] cam_K[8] cam_K[9]]

        for i in 1:length(OBJECT_IDS)
            cam_R_m2c = scene_gt[frame_id][i][:cam_R_m2c]
            cam_t_m2c = scene_gt[frame_id][i][:cam_t_m2c]
            obj_id = scene_gt[frame_id][i][:obj_id]
            mask_id = OBJECT_ID_TO_MASK[obj_id]
            obj_idx = mask_id + 1

            mask_file = joinpath(mask_dir, @sprintf("%06d_%06d.png", frame_id, mask_id))
            if !isfile(mask_file)
                push!(invalid_frames, frame_idx)
                continue
            end
            masks[frame_idx, obj_idx, :, :] .= load(mask_file)

            # Initialize pose matrix
            poses[frame_idx, obj_idx, :, :] .= 0.0
            # Rotation matrix - Fix the transpose issue
            poses[frame_idx, obj_idx, 1:3, 1:3] .= [cam_R_m2c[1] cam_R_m2c[2] cam_R_m2c[3];
                cam_R_m2c[4] cam_R_m2c[5] cam_R_m2c[6];
                cam_R_m2c[7] cam_R_m2c[8] cam_R_m2c[9]]
            # Translation vector converted from millimeters to meters
            poses[frame_idx, obj_idx, 1:3, 4] .= cam_t_m2c ./ 1000.0
            # Homogeneous coordinate
            poses[frame_idx, obj_idx, 4, 4] = 1.0
        end
    end
    valid_idxs = collect(setdiff(1:length(FRAME_IDS), invalid_frames))
    return Frames(rgbs, depths, BitArray(masks), ks, poses, valid_idxs), objects
end

Base.iterate(frames::Frames) = iterate(frames, 1)
function Base.iterate(frames::Frames, state::Int)
    if state > length(frames.valid_idxs)
        return nothing
    end

    idx = frames.valid_idxs[state]
    frame = Frame(
        frames.rgbs[idx, :, :, :],
        frames.depths[idx, :, :],
        frames.masks[idx, :, :, :],
        frames.ks[idx, :, :],
        frames.poses[idx, :, :, :],
        idx
    )
    return (frame, state + 1)
end
Base.length(frames::Frames) = length(frames.valid_idxs)
Base.eltype(::Type{Frames}) = Frame

function Base.getindex(frames::Frames, i::Int)
    if i < 1 || i > length(frames.valid_idxs)
        throw(BoundsError(frames, i))
    end

    idx = frames.valid_idxs[i]
    return Frame(
        frames.rgbs[idx, :, :, :],
        frames.depths[idx, :, :],
        frames.masks[idx, :, :, :],
        frames.ks[idx, :, :],
        frames.poses[idx, :, :, :],
        idx
    )
end

function Base.getindex(frames::Frames, indices::AbstractVector{Int})
    return [frames[i] for i in indices]
end

end # module LMOLoader
