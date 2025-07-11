# ImageProcessor.jl
module ImageProcessor

export process_mask_crop, process_pose_crop
export visualize_preprocessing_stages, visualize_batch_results
export guess_object_center
export to_batch, to_single

using Plots, Images

# Check if CUDA is available
const USE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

# Load the appropriate implementation
if USE_CUDA
    @info "CUDA is available, using GPU acceleration"
    include("ImageProcessor/CUDAImageProcessor.jl")
else
    @info "CUDA not available, using CPU implementation"
    include("ImageProcessor/CPUImageProcessor.jl")
end

# Utility functions for single/batch conversion
"""
    to_batch(data)

Convert single image data to batch format by adding a batch dimension at the front.
- (H, W) -> (1, H, W)
- (H, W, C) -> (1, H, W, C)
"""
function to_batch(data)
    return reshape(data, 1, size(data)...)
end

"""
    to_single(data)

Convert batch data with single element to non-batch format by removing the batch dimension.
- (1, H, W) -> (H, W)
- (1, H, W, C) -> (H, W, C)
"""
function to_single(data)
    @assert size(data, 1) == 1 "Can only convert batch size 1 to single"
    return dropdims(data, dims=1)
end

# Visualization functions
function draw_crop_box(img, crop_box; color=[1.0, 0.0, 0.0], thickness=3)
    img_with_box = copy(img)
    top, bottom, left, right = crop_box

    H, W = size(img)[1:2]

    top = max(1, floor(Int, top))
    bottom = min(H, ceil(Int, bottom))
    left = max(1, floor(Int, left))
    right = min(W, ceil(Int, right))

    for t in 0:thickness-1
        for u in left:right
            if top + t <= H && u <= W
                if ndims(img) == 3
                    img_with_box[top + t, u, :] = color
                else
                    img_with_box[top + t, u] = maximum(img)
                end
            end
            if bottom - t >= 1 && bottom - t <= H && u <= W
                if ndims(img) == 3
                    img_with_box[bottom - t, u, :] = color
                else
                    img_with_box[bottom - t, u] = maximum(img)
                end
            end
        end

        for v in top:bottom
            if left + t <= W && v <= H
                if ndims(img) == 3
                    img_with_box[v, left + t, :] = color
                else
                    img_with_box[v, left + t] = maximum(img)
                end
            end
            if right - t >= 1 && right - t <= W && v <= H
                if ndims(img) == 3
                    img_with_box[v, right - t, :] = color
                else
                    img_with_box[v, right - t] = maximum(img)
                end
            end
        end
    end

    return img_with_box
end

function visualize_preprocessing_stages(rgb, depth, mask, mask_result, pose_result=nothing)
    # Convert from batch format if needed
    if ndims(rgb) == 4
        rgb = rgb[1, :, :, :]
    end
    if ndims(depth) == 3
        depth = depth[1, :, :]
    end
    if ndims(mask) == 3
        mask = mask[1, :, :]
    end

    fig = plot(layout=(3, 3), size=(1400, 1200), titlefontsize=8, dpi=100)

    # Row 1: Original data
    plot!(fig[1], colorview(RGB, permutedims(rgb, (3, 1, 2))),
          title="Original RGB", axis=false)

    valid_depth = depth .> 0
    if any(valid_depth)
        depth_range = (minimum(depth[valid_depth]), maximum(depth[valid_depth]))
        heatmap!(fig[2], depth,
                 title="Original Depth\n$(round(depth_range[1], digits=3))-$(round(depth_range[2], digits=3))m",
                 aspect_ratio=:equal, yflip=true, colorbar=true, clim=(0, depth_range[2]))
    else
        heatmap!(fig[2], depth, title="Original Depth",
                 aspect_ratio=:equal, yflip=true, colorbar=true)
    end

    heatmap!(fig[3], mask, title="Original Mask",
             aspect_ratio=:equal, yflip=true, colorbar=false)

    # Row 2: Mask-based crop
    rgb_with_mask_box = draw_crop_box(rgb, mask_result.crop_infos[1].crop_box;
                                     color=[1.0, 0.0, 0.0])
    plot!(fig[4], colorview(RGB, permutedims(rgb_with_mask_box, (3, 1, 2))),
          title="RGB with Mask Crop Box", axis=false)

    mask_rgb = mask_result.rgb[1, :, :, :]
    plot!(fig[5], colorview(RGB, permutedims(mask_rgb, (3, 1, 2))),
          title="Mask Cropped RGB", axis=false)

    mask_depth = mask_result.depth[1, :, :]
    valid_depth_mask = mask_depth .> 0
    if any(valid_depth_mask)
        depth_range = extrema(mask_depth[valid_depth_mask])
        heatmap!(fig[6], mask_depth,
                 title="Mask Cropped Depth\n$(round(depth_range[1], digits=3))-$(round(depth_range[2], digits=3))m",
                 aspect_ratio=:equal, yflip=true, colorbar=true, clim=(0, depth_range[2]))
    else
        heatmap!(fig[6], mask_depth,
                 title="Mask Cropped Depth",
                 aspect_ratio=:equal, yflip=true, colorbar=true)
    end

    # Row 3: XYZ channels from mask crop
    mask_xyz = mask_result.xyz[1, :, :, :]
    for (i, (channel, name)) in enumerate(zip(1:3, ["X", "Y", "Z"]))
        data = mask_xyz[:, :, channel]
        valid = mask_xyz[:, :, 3] .> 0.001

        if any(valid)
            valid_data = data[valid]
            if !isempty(valid_data)
                vrange = extrema(valid_data)
                title_str = "Mask $name\n$(round(vrange[1], digits=2)) to $(round(vrange[2], digits=2))"
                heatmap!(fig[6 + i], data,
                         title=title_str,
                         aspect_ratio=:equal, yflip=true, colorbar=true,
                         clim=vrange, color=:RdBu)
            else
                heatmap!(fig[6 + i], data,
                         title="Mask $name (no data)",
                         aspect_ratio=:equal, yflip=true, colorbar=true)
            end
        else
            heatmap!(fig[6 + i], data,
                     title="Mask $name (no data)",
                     aspect_ratio=:equal, yflip=true, colorbar=true)
        end
    end

    savefig(fig, "figures/preprocessing_mask_crop.png")

    # Create second figure for pose-based crop if available
    if pose_result !== nothing
        fig2 = plot(layout=(2, 3), size=(1400, 800), titlefontsize=8, dpi=100)

        rgb_with_pose_box = draw_crop_box(rgb, pose_result.crop_infos[1].crop_box;
                                         color=[0.0, 1.0, 0.0])
        plot!(fig2[1], colorview(RGB, permutedims(rgb_with_pose_box, (3, 1, 2))),
              title="RGB with Pose Crop Box", axis=false)

        pose_rgb = pose_result.rgb[1, :, :, :]
        plot!(fig2[2], colorview(RGB, permutedims(pose_rgb, (3, 1, 2))),
              title="Pose Cropped RGB", axis=false)

        pose_depth = pose_result.depth[1, :, :]
        valid_depth_pose = pose_depth .> 0
        if any(valid_depth_pose)
            depth_range = extrema(pose_depth[valid_depth_pose])
            heatmap!(fig2[3], pose_depth,
                     title="Pose Cropped Depth\n$(round(depth_range[1], digits=3))-$(round(depth_range[2], digits=3))m",
                     aspect_ratio=:equal, yflip=true, colorbar=true, clim=(0, depth_range[2]))
        else
            heatmap!(fig2[3], pose_depth,
                     title="Pose Cropped Depth",
                     aspect_ratio=:equal, yflip=true, colorbar=true)
        end

        pose_xyz = pose_result.xyz[1, :, :, :]
        for (i, (channel, name)) in enumerate(zip(1:3, ["X", "Y", "Z"]))
            data = pose_xyz[:, :, channel]
            valid = pose_xyz[:, :, 3] .> 0.001

            if any(valid)
                valid_data = data[valid]
                if !isempty(valid_data)
                    vrange = extrema(valid_data)
                    title_str = "Pose $name\n$(round(vrange[1], digits=2)) to $(round(vrange[2], digits=2))"
                    heatmap!(fig2[3 + i], data,
                             title=title_str,
                             aspect_ratio=:equal, yflip=true, colorbar=true,
                             clim=vrange, color=:RdBu)
                else
                    heatmap!(fig2[3 + i], data,
                             title="Pose $name (no data)",
                             aspect_ratio=:equal, yflip=true, colorbar=true)
                end
            else
                heatmap!(fig2[3 + i], data,
                         title="Pose $name (no data)",
                         aspect_ratio=:equal, yflip=true, colorbar=true)
            end
        end

        savefig(fig2, "figures/preprocessing_pose_crop.png")
    end

    return fig
end

function visualize_batch_results(batch_result, batch_type="mask"; max_samples=4)
    N = min(size(batch_result.rgb, 1), max_samples)
    fig = plot(layout=(N, 4), size=(1200, 300*N), titlefontsize=8, dpi=100)

    for n in 1:N
        # RGB
        rgb = batch_result.rgb[n, :, :, :]
        plot!(fig[(n-1)*4 + 1], colorview(RGB, permutedims(rgb, (3, 1, 2))),
              title="Sample $n RGB", axis=false)

        # Depth
        depth = batch_result.depth[n, :, :]
        valid_depth = depth .> 0
        if any(valid_depth)
            depth_range = extrema(depth[valid_depth])
            heatmap!(fig[(n-1)*4 + 2], depth,
                     title="Depth $(round(depth_range[1], digits=3))-$(round(depth_range[2], digits=3))m",
                     aspect_ratio=:equal, yflip=true, colorbar=true, clim=(0, depth_range[2]))
        else
            heatmap!(fig[(n-1)*4 + 2], depth,
                     title="Depth",
                     aspect_ratio=:equal, yflip=true, colorbar=true)
        end

        # XYZ (show Z channel)
        xyz_z = batch_result.xyz[n, :, :, 3]
        valid = xyz_z .> 0.001
        if any(valid)
            vrange = extrema(xyz_z[valid])
            heatmap!(fig[(n-1)*4 + 3], xyz_z,
                     title="Z $(round(vrange[1], digits=2))-$(round(vrange[2], digits=2))",
                     aspect_ratio=:equal, yflip=true, colorbar=true,
                     clim=vrange, color=:RdBu)
        else
            heatmap!(fig[(n-1)*4 + 3], xyz_z,
                     title="Z (no data)",
                     aspect_ratio=:equal, yflip=true, colorbar=true)
        end

        # Mask (if available)
        if haskey(batch_result, :mask) && batch_result.mask !== nothing
            mask = batch_result.mask[n, :, :]
            heatmap!(fig[(n-1)*4 + 4], mask,
                     title="Mask",
                     aspect_ratio=:equal, yflip=true, colorbar=false)
        else
            # Empty plot
            plot!(fig[(n-1)*4 + 4], legend=false, grid=false, axis=false)
        end
    end

    savefig(fig, "figures/batch_$(batch_type)_processing.png")
    return fig
end

end # module ImageProcessor
