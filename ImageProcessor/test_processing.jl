# test_processing.jl

include("../LMOLoader.jl")
using .LMOLoader

include("../ImageProcessor.jl")
using .ImageProcessor

# Load data
frames, objects = load_scene()

# Test single image processing (using utility functions)
println("=== Testing Single Image Processing ===")
frame = frames[1]
object_id = 2
mask_single = frame.mask[object_id, :, :]
pose_single = frame.pose[object_id, :, :]
obj = objects[object_id]

# Convert single to batch format
rgb_batch = to_batch(frame.rgb)
depth_batch = to_batch(frame.depth)
mask_batch = to_batch(mask_single)
pose_batch = to_batch(pose_single)
K_batch = to_batch(frame.K)
mesh_radius_batch = Float32[obj.diameter/2]
mesh_diameter_batch = Float32[obj.diameter]

# Process
mask_result = process_mask_crop(
    rgb_batch, depth_batch, mask_batch, K_batch;
    mesh_radii=mesh_radius_batch,
    estimate_centers=true
)

pose_result = process_pose_crop(
    rgb_batch, depth_batch, pose_batch, K_batch, mesh_diameter_batch;
    mesh_radii=mesh_radius_batch
)

println("Estimated object center: $(round.(mask_result.object_centers[1, :], digits=3))")
println("Output size: $(size(mask_result.rgb, 2))×$(size(mask_result.rgb, 3))")

# Visualize single results
visualize_preprocessing_stages(
    to_single(rgb_batch),
    to_single(depth_batch),
    to_single(mask_batch),
    mask_result,
    pose_result
)

# Test batch processing
println("\n=== Testing Batch Processing ===")
batch_size = 252
frame_indices = 1:batch_size

# Prepare batch data - use Float32 arrays since LMOLoader provides Float32
rgb_batch_multi = zeros(Float32, batch_size, 480, 640, 3)
depth_batch_multi = zeros(Float32, batch_size, 480, 640)
mask_batch_multi = falses(batch_size, 480, 640)
pose_batch_multi = zeros(Float32, batch_size, 4, 4)
K_batch_multi = zeros(Float32, batch_size, 3, 3)
mesh_radius_batch_multi = Float32[]
mesh_diameter_batch_multi = Float32[]

for (i, idx) in enumerate(frame_indices)
    local frame = frames[idx]
    rgb_batch_multi[i, :, :, :] = frame.rgb
    depth_batch_multi[i, :, :] = frame.depth
    mask_batch_multi[i, :, :] = frame.mask[object_id, :, :]
    pose_batch_multi[i, :, :] = frame.pose[object_id, :, :]
    K_batch_multi[i, :, :] = frame.K
    push!(mesh_radius_batch_multi, Float32(obj.diameter/2))
    push!(mesh_diameter_batch_multi, Float32(obj.diameter))
end

# Time the batch processing
println("Processing batch of $batch_size images...")
t_start = time()

mask_result_batch = process_mask_crop(
    rgb_batch_multi, depth_batch_multi, mask_batch_multi, K_batch_multi;
    mesh_radii=mesh_radius_batch_multi,
    estimate_centers=true
)

pose_result_batch = process_pose_crop(
    rgb_batch_multi, depth_batch_multi, pose_batch_multi, K_batch_multi, mesh_diameter_batch_multi;
    mesh_radii=mesh_radius_batch_multi
)

t_elapsed = time() - t_start
println("Batch processing took $(round(t_elapsed, digits=3)) seconds")
println("Average per image: $(round(t_elapsed/batch_size*1000, digits=1)) ms")

# Visualize batch results
visualize_batch_results(mask_result_batch, "mask")
visualize_batch_results(pose_result_batch, "pose")

# Verify consistency between single and batch processing
println("\n=== Verifying Consistency ===")
single_rgb = mask_result.rgb[1, :, :, :]
batch_first_rgb = mask_result_batch.rgb[1, :, :, :]
rgb_diff = maximum(abs.(single_rgb - batch_first_rgb))
println("Max RGB difference: $rgb_diff")

single_xyz = mask_result.xyz[1, :, :, :]
batch_first_xyz = mask_result_batch.xyz[1, :, :, :]
valid_mask = (single_xyz[:, :, 3] .> 0.001) .& (batch_first_xyz[:, :, 3] .> 0.001)
if any(valid_mask)
    xyz_diff = maximum(abs.(single_xyz[valid_mask, :] - batch_first_xyz[valid_mask, :]))
    println("Max XYZ difference: $xyz_diff")
else
    println("No valid XYZ points to compare")
end

println("\nProcessing complete! Check figures/ directory for visualizations.")
