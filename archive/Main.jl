# Main.jl

include("LMOLoader.jl")
using .LMOLoader

include("ImageProcessor.jl")
using .ImageProcessor

# Load data
frames, objects = load_scene()
frame = frames[1]  # Try a different frame

# Get first object that's visible
object_id = 2
mask_single = frame.mask[object_id, :, :]
pose_single = frame.pose[object_id, :, :]

# Get object info
obj = objects[object_id]

println("Processing object $object_id with diameter $(round(obj.diameter, digits=3))m")
println("Depth range: $(round(minimum(frame.depth[frame.depth .> 0]), digits=3)) - $(round(maximum(frame.depth), digits=3))m")
println("\nPose matrix:")
display(pose_single)
println("\nCamera intrinsics K:")
display(frame.K)

# Preprocess with debug enabled
processed = preprocess_image(
    frame.rgb, frame.depth, mask_single, frame.K, pose_single;
    mesh_diameter=obj.diameter,
    mesh_radius=obj.diameter/2,
    debug=true
)

# Visualize all stages
visualize_preprocessing_stages(frame.rgb, frame.depth, mask_single, processed)

println("\nSaved visualizations:")
println("- preprocessing_mask_crop.png")
println("- preprocessing_pose_crop.png")

# Print statistics
println("\nPreprocessing Statistics:")
println("Original image size: $(size(frame.rgb, 1))×$(size(frame.rgb, 2))")
println("Cropped image size: $(size(processed.rgb_mask, 1))×$(size(processed.rgb_mask, 2))")

# Check XYZ data
valid_xyz_mask = processed.xyz_mask[:, :, 3] .> 0.001
if any(valid_xyz_mask)
    println("\nMask-based XYZ ranges (normalized):")
    for (i, dim) in enumerate(["X", "Y", "Z"])
        vals = processed.xyz_mask[valid_xyz_mask, i]
        println("$dim: $(round(minimum(vals), digits=3)) to $(round(maximum(vals), digits=3))")
    end

    # Check corners
    println("\nChecking XYZ values at corners of mask crop:")
    H, W = size(processed.xyz_mask, 1), size(processed.xyz_mask, 2)
    corners = [(1,1,"top-left"), (1,W,"top-right"), (H,1,"bottom-left"), (H,W,"bottom-right")]
    for (v,u,name) in corners
        if processed.xyz_mask[v,u,3] > 0.001
            x,y,z = processed.xyz_mask[v,u,:]
            println("  $name: X=$(round(x,digits=3)), Y=$(round(y,digits=3)), Z=$(round(z,digits=3))")
        end
    end
else
    println("\nNo valid XYZ data in mask-based crop")
end
