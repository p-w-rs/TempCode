# test_processing.jl

include("LMOLoader.jl")
using .LMOLoader

include("ImageProcessor.jl")
using .ImageProcessor

# Load data
frames, objects = load_scene()
frame = frames[1]

# Get first object that's visible
object_id = 2
mask_single = frame.mask[object_id, :, :]
pose_single = frame.pose[object_id, :, :]

# Get object info
obj = objects[object_id]

println("Processing object $object_id with diameter $(round(obj.diameter, digits=3))m")

# Test 1: Mask-based cropping (for initial pose estimation)
println("\n=== Mask-based cropping (initial pose estimation) ===")
mask_result = process_mask_crop(
    frame.rgb, frame.depth, mask_single, frame.K;
    mesh_radius=obj.diameter/2,
    estimate_center=true  # Estimate center from mask/depth since we don't have pose
)

println("Estimated object center: $(round.(mask_result.object_center, digits=3))")
println("Output size: $(size(mask_result.rgb, 1))×$(size(mask_result.rgb, 2))")

# Test 2: Pose-based cropping (for tracking/refinement)
println("\n=== Pose-based cropping (tracking/refinement) ===")
pose_result = process_pose_crop(
    frame.rgb, frame.depth, pose_single, frame.K, obj.diameter;
    mesh_radius=obj.diameter/2
)

println("Pose center: $(round.(pose_single[1:3, 4], digits=3))")
println("Output size: $(size(pose_result.rgb, 1))×$(size(pose_result.rgb, 2))")

# Optional: Visualize if needed
if true  # Set to true when debugging
    visualize_preprocessing_stages(frame.rgb, frame.depth, mask_single, mask_result, pose_result)
    println("\nSaved visualizations to figures/")
end

# Show XYZ statistics for mask-based result
valid_xyz = mask_result.xyz[:, :, 3] .> 0.001
if any(valid_xyz)
    println("\nMask-based XYZ ranges (normalized):")
    for (i, dim) in enumerate(["X", "Y", "Z"])
        vals = mask_result.xyz[valid_xyz, i]
        println("$dim: $(round(minimum(vals), digits=3)) to $(round(maximum(vals), digits=3))")
    end
end
