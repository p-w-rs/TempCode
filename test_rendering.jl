# test_renderer.jl

include("LMOLoader.jl")
include("MeshRenderer.jl")
include("PoseHandler.jl")

using .LMOLoader
using .MeshRenderer
using .PoseHandler
using Statistics
using Plots

# Load scene data
println("Loading scene data...")
frames, objects = load_scene()
frame = frames[1]

# Get first visible object
object_id = 2
mask_single = frame.mask[object_id, :, :]
pose_gt = frame.pose[object_id, :, :]
obj = objects[object_id]

println("Testing with object $object_id")
println("Object diameter: $(obj.diameter) m")

# Get image dimensions
H, W = size(frame.rgb, 1), size(frame.rgb, 2)
println("Image size: $H x $W")

# Test 1: Render at ground truth pose
println("\nTest 1: Rendering at ground truth pose...")
println("This may take a moment...")

rgb_rendered, depth_rendered, mask_rendered = render_mesh(obj.mesh, pose_gt, frame.K, H, W)

# Compare with actual frame
println("\nComparing rendered vs actual:")
println("- RGB shape: $(size(rgb_rendered)) vs $(size(frame.rgb))")
println("- Depth shape: $(size(depth_rendered)) vs $(size(frame.depth))")
println("- Mask shape: $(size(mask_rendered)) vs $(size(mask_single))")

# Calculate mask IoU
intersection = sum(mask_rendered .& mask_single)
union = sum(mask_rendered .| mask_single)
iou = intersection / max(union, 1)
println("- Mask IoU: $(round(iou, digits=3))")

# Calculate depth error where both masks are true
valid_pixels = mask_rendered .& mask_single .& (frame.depth .> 0.001)
if any(valid_pixels)
    depth_error = mean(abs.(depth_rendered[valid_pixels] - frame.depth[valid_pixels]))
    println("- Mean depth error (m): $(round(depth_error, digits=4))")
end

# Save comparison images
if !isdir("figures")
    mkdir("figures")
end

# Create comparison plot
p1 = heatmap(frame.rgb[:,:,1], title="Original RGB (R channel)",
             aspect_ratio=:equal, yflip=true, color=:grays)
p2 = heatmap(rgb_rendered[:,:,1], title="Rendered RGB (R channel)",
             aspect_ratio=:equal, yflip=true, color=:grays)
p3 = heatmap(frame.depth, title="Original Depth",
             aspect_ratio=:equal, yflip=true, clims=(0, maximum(frame.depth[frame.depth .> 0])))
p4 = heatmap(depth_rendered, title="Rendered Depth",
             aspect_ratio=:equal, yflip=true, clims=(0, maximum(depth_rendered[depth_rendered .> 0])))
p5 = heatmap(Float64.(mask_single), title="Original Mask",
             aspect_ratio=:equal, yflip=true, color=:grays)
p6 = heatmap(Float64.(mask_rendered), title="Rendered Mask",
             aspect_ratio=:equal, yflip=true, color=:grays)

plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(800, 900))
savefig("figures/renderer_comparison.png")
println("Saved comparison to figures/renderer_comparison.png")

# Test 2: Generate and render a few poses
println("\nTest 2: Testing pose generation...")
poses = make_rotation_grid(3, 120)  # 3 viewpoints, 120-degree in-plane steps
println("Generated $(length(poses)) poses")

# Test 3: Apply pose delta
println("\nTest 3: Testing pose updates...")
# Small translation and rotation
trans_delta = [0.01, 0.0, 0.0]  # 1cm in x direction
rot_angle = 0.1  # radians
rot_delta = [1.0 0.0 0.0;
             0.0 cos(rot_angle) -sin(rot_angle);
             0.0 sin(rot_angle) cos(rot_angle)]

updated_pose = apply_pose_delta(pose_gt, trans_delta, rot_delta)
println("Applied translation: $trans_delta")
println("Applied rotation angle: $(rot_angle) radians")

println("\nAll tests completed! Check figures/ directory for results.")
