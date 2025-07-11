# ImageProcessor.jl
module ImageProcessor

export preprocess_image, visualize_preprocessing_stages
export compute_pose_based_crop

using Images, ImageTransformations, ImageFiltering
using CoordinateTransformations, Interpolations
using Statistics, LinearAlgebra
using Plots

"""
    erode_depth(depth; radius=2, depth_diff_threshold=0.001, ratio_threshold=0.8, zfar=100.0)

Erode depth values near edges/discontinuities.
"""
function erode_depth(depth; radius=2, depth_diff_threshold=0.001, ratio_threshold=0.8, zfar=100.0)
    H, W = size(depth)
    output = copy(depth)

    for h in 1:H, w in 1:W
        d_ori = depth[h, w]
        if d_ori < 0.001 || d_ori >= zfar
            output[h, w] = 0.0
            continue
        end

        bad_cnt = 0.0
        total = 0.0

        for u in max(1, w-radius):min(W, w+radius)
            for v in max(1, h-radius):min(H, h+radius)
                cur_depth = depth[v, u]
                total += 1.0
                if cur_depth < 0.001 || cur_depth >= zfar || abs(cur_depth - d_ori) > depth_diff_threshold
                    bad_cnt += 1.0
                end
            end
        end

        if bad_cnt / total > ratio_threshold
            output[h, w] = 0.0
        end
    end

    return output
end

"""
    bilateral_filter_depth(depth; radius=2, sigma_d=2.0, sigma_r=100000.0, zfar=100.0)

Apply bilateral filter to depth image.
"""
function bilateral_filter_depth(depth; radius=2, sigma_d=2.0, sigma_r=100000.0, zfar=100.0)
    H, W = size(depth)
    output = zeros(Float64, H, W)

    for h in 1:H, w in 1:W
        if depth[h, w] < 0.001
            continue
        end

        mean_depth = 0.0
        num_valid = 0

        for u in max(1, w-radius):min(W, w+radius)
            for v in max(1, h-radius):min(H, h+radius)
                cur_depth = depth[v, u]
                if cur_depth >= 0.001 && cur_depth < zfar
                    num_valid += 1
                    mean_depth += cur_depth
                end
            end
        end

        if num_valid == 0
            continue
        end
        mean_depth /= Float64(num_valid)

        depthCenter = depth[h, w]
        sum_weight = 0.0
        sum = 0.0

        for u in max(1, w-radius):min(W, w+radius)
            for v in max(1, h-radius):min(H, h+radius)
                cur_depth = depth[v, u]
                if cur_depth >= 0.001 && cur_depth < zfar && abs(cur_depth - mean_depth) < 0.01
                    weight = exp(-Float64((u-w)^2 + (h-v)^2) / (2.0*sigma_d*sigma_d) -
                                (depthCenter-cur_depth)^2/(2.0*sigma_r*sigma_r))
                    sum_weight += weight
                    sum += weight * cur_depth
                end
            end
        end

        if sum_weight > 0 && num_valid > 0
            output[h, w] = sum / sum_weight
        end
    end

    return output
end

"""
    compute_crop_transform_mask(mask; crop_ratio=1.2, target_size=(160, 160))

Compute transformation matrix for cropping based on mask.
"""
function compute_crop_transform_mask(mask; crop_ratio=1.2, target_size=(160, 160))
    mask_indices = findall(mask)
    if isempty(mask_indices)
        throw(ArgumentError("Empty mask provided"))
    end

    # Get bounding box in Julia's 1-indexed coordinates
    vs = [idx[1] for idx in mask_indices]  # rows
    us = [idx[2] for idx in mask_indices]  # cols

    umin, umax = extrema(us)
    vmin, vmax = extrema(vs)

    # Center of bounding box (still 1-indexed)
    uc = (umin + umax) / 2.0
    vc = (vmin + vmax) / 2.0

    # Convert to 0-indexed for calculations
    uc_0 = uc - 1
    vc_0 = vc - 1

    # Size of bounding box
    height = vmax - vmin
    width = umax - umin

    # Apply crop ratio to get square crop
    crop_size = max(height, width) * crop_ratio
    half_size = crop_size / 2.0

    # Define crop box in 0-indexed coordinates
    left = uc_0 - half_size
    top = vc_0 - half_size

    # Transformation matrices
    scale = crop_size / target_size[1]

    # Transform from output (cropped) to input (original) - for warping
    T_inv = [
        scale  0.0    left;
        0.0    scale  top;
        0.0    0.0    1.0
    ]

    # Transform from input to output - for updating K
    T_forward = [
        1/scale  0.0    -left/scale;
        0.0      1/scale  -top/scale;
        0.0      0.0     1.0
    ]

    crop_info = (
        center = (vc, uc),
        bbox = (vmin, vmax, umin, umax),
        crop_box = (top+1, top+crop_size+1, left+1, left+crop_size+1),  # Back to 1-indexed for display
        crop_size = crop_size,
        T_inv = T_inv,
        T_forward = T_forward
    )

    return T_inv, crop_info
end

"""
    compute_pose_based_crop(pose, K, mesh_diameter; crop_ratio=1.5, target_size=(160, 160), debug=false)

Compute transformation for pose-based cropping.
"""
function compute_pose_based_crop(pose, K, mesh_diameter; crop_ratio=1.5, target_size=(160, 160), debug=false)
    # Get object center in camera frame (should be in meters)
    center_3d = pose[1:3, 4]

    if debug
        println("Object center in camera frame: ", center_3d)
        println("Mesh diameter: ", mesh_diameter)
    end

    # Create 3D bounding box points
    radius = mesh_diameter * crop_ratio / 2.0

    # Create points for projection
    points_3d = [
        center_3d,                    # center
        center_3d + [radius, 0, 0],   # +x
        center_3d + [-radius, 0, 0],  # -x
        center_3d + [0, radius, 0],   # +y
        center_3d + [0, -radius, 0],  # -y
    ]

    # Project all points to image plane
    us = Float64[]
    vs = Float64[]

    for pt_3d in points_3d
        # Project point
        pt_2d_homo = K * pt_3d
        u = pt_2d_homo[1] / pt_2d_homo[3] + 1  # Convert to 1-indexed
        v = pt_2d_homo[2] / pt_2d_homo[3] + 1  # Convert to 1-indexed
        push!(us, u)
        push!(vs, v)

        if debug
            println("3D point: ", pt_3d, " -> 2D: (", u, ", ", v, ")")
        end
    end

    # Use the projected center
    uc = us[1]
    vc = vs[1]

    # Get radius from max distance to center
    radius_px = 0.0
    for i in 2:length(us)
        dist = sqrt((us[i] - uc)^2 + (vs[i] - vc)^2)
        radius_px = max(radius_px, dist)
    end

    if debug
        println("Center in image (1-indexed): (", uc, ", ", vc, ")")
        println("Radius in pixels: ", radius_px)
    end

    # Convert to 0-indexed for transformation
    uc_0 = uc - 1
    vc_0 = vc - 1

    # Make square crop box
    left = uc_0 - radius_px
    top = vc_0 - radius_px
    crop_size = 2 * radius_px

    # Transformation matrices
    scale = crop_size / target_size[1]

    T_inv = [
        scale  0.0    left;
        0.0    scale  top;
        0.0    0.0    1.0
    ]

    T_forward = [
        1/scale  0.0    -left/scale;
        0.0      1/scale  -top/scale;
        0.0      0.0     1.0
    ]

    crop_info = (
        center = (vc, uc),
        crop_box = (top+1, top+crop_size+1, left+1, left+crop_size+1),  # Back to 1-indexed
        crop_size = crop_size,
        T_inv = T_inv,
        T_forward = T_forward
    )

    return T_inv, crop_info
end

"""
    warp_perspective(image, T_inv, target_size; mode=:linear)

Apply perspective warp using transformation matrix.
"""
function warp_perspective(image, T_inv, target_size; mode=:linear)
    H_out, W_out = target_size

    if ndims(image) == 3
        output = zeros(eltype(image), H_out, W_out, size(image, 3))
    else
        output = zeros(eltype(image), H_out, W_out)
    end

    # For each output pixel
    for v_out in 1:H_out, u_out in 1:W_out
        # Transform to input coordinates (0-indexed)
        pt_out = [u_out - 1; v_out - 1; 1.0]
        pt_in = T_inv * pt_out
        u_in = pt_in[1] / pt_in[3]
        v_in = pt_in[2] / pt_in[3]

        # Convert back to 1-indexed for Julia arrays
        u_in += 1
        v_in += 1

        if mode == :nearest
            u_nearest = round(Int, u_in)
            v_nearest = round(Int, v_in)

            if 1 <= u_nearest <= size(image, 2) && 1 <= v_nearest <= size(image, 1)
                if ndims(image) == 3
                    output[v_out, u_out, :] = image[v_nearest, u_nearest, :]
                else
                    output[v_out, u_out] = image[v_nearest, u_nearest]
                end
            end
        else
            u0 = floor(Int, u_in)
            v0 = floor(Int, v_in)
            u1 = u0 + 1
            v1 = v0 + 1

            if u0 >= 1 && u1 <= size(image, 2) && v0 >= 1 && v1 <= size(image, 1)
                wu1 = u_in - u0
                wv1 = v_in - v0
                wu0 = 1 - wu1
                wv0 = 1 - wv1

                if ndims(image) == 3
                    for c in 1:size(image, 3)
                        val = wv0 * wu0 * image[v0, u0, c] +
                              wv0 * wu1 * image[v0, u1, c] +
                              wv1 * wu0 * image[v1, u0, c] +
                              wv1 * wu1 * image[v1, u1, c]
                        output[v_out, u_out, c] = val
                    end
                else
                    output[v_out, u_out] = wv0 * wu0 * image[v0, u0] +
                                          wv0 * wu1 * image[v0, u1] +
                                          wv1 * wu0 * image[v1, u0] +
                                          wv1 * wu1 * image[v1, u1]
                end
            end
        end
    end

    return output
end

"""
    depth_to_xyz(depth, K; debug=false)

Convert depth image to XYZ coordinates in camera frame.
Camera frame: +X right, +Y down, +Z forward
"""
function depth_to_xyz(depth, K; debug=false)
    H, W = size(depth)
    xyz = zeros(Float64, H, W, 3)

    fx = K[1,1]
    fy = K[2,2]
    cx = K[1,3]
    cy = K[2,3]

    if debug
        println("Camera intrinsics:")
        println("fx=", fx, ", fy=", fy)
        println("cx=", cx, ", cy=", cy)
    end

    # Check a few sample points for debugging
    debug_points = debug ? [(1,1), (1,W), (H,1), (H,W), (H÷2, W÷2)] : []

    for v in 1:H, u in 1:W
        z = depth[v, u]
        if z > 0.001
            # Standard pinhole camera model
            # u,v are 1-indexed in Julia, convert to 0-indexed for camera model
            x = (Float64(u - 1) - cx) * z / fx
            y = (Float64(v - 1) - cy) * z / fy
            xyz[v, u, 1] = x
            xyz[v, u, 2] = y
            xyz[v, u, 3] = z

            if debug && (v,u) in debug_points
                println("Pixel (", v, ",", u, ") -> X=", round(x, digits=3),
                       ", Y=", round(y, digits=3), ", Z=", round(z, digits=3))
            end
        end
    end

    return xyz
end

"""
    normalize_xyz(xyz, pose_center, mesh_radius; bound=2.0)

Normalize XYZ relative to object center and scale.
"""
function normalize_xyz(xyz, pose_center, mesh_radius; bound=2.0)
    normalized = zeros(size(xyz))
    H, W = size(xyz, 1), size(xyz, 2)

    # Invalid mask
    invalid = xyz[:, :, 3] .< 0.001

    # Subtract object center and scale
    for i in 1:3
        normalized[:, :, i] = (xyz[:, :, i] .- pose_center[i]) ./ mesh_radius
    end

    # Apply bound check
    invalid = invalid .| (abs.(normalized) .>= bound)

    # Zero out invalid points
    for v in 1:H, u in 1:W
        if invalid[v, u, 1] || invalid[v, u, 2] || invalid[v, u, 3]
            normalized[v, u, :] .= 0.0
        end
    end

    return normalized
end

"""
    preprocess_image(rgb, depth, mask, K, pose;
                    crop_ratio=1.2, target_size=(160, 160),
                    mesh_diameter=nothing, mesh_radius=nothing, debug=false)

Complete preprocessing pipeline for FoundationPose.
"""
function preprocess_image(rgb, depth, mask, K, pose;
                         crop_ratio=1.2,
                         target_size=(160, 160),
                         mesh_diameter=nothing,
                         mesh_radius=nothing,
                         debug=false)

    # Step 1: Preprocess depth
    depth_eroded = erode_depth(depth; radius=2, depth_diff_threshold=0.001,
                              ratio_threshold=0.8, zfar=100.0)
    depth_filtered = bilateral_filter_depth(depth_eroded; radius=2, sigma_d=2.0,
                                          sigma_r=100000.0, zfar=100.0)

    # Step 2: Mask-based cropping
    T_mask, crop_info_mask = compute_crop_transform_mask(mask;
                                                        crop_ratio=crop_ratio,
                                                        target_size=target_size)

    rgb_crop_mask = warp_perspective(rgb, T_mask, target_size; mode=:linear)
    depth_crop_mask = warp_perspective(depth_filtered, T_mask, target_size; mode=:nearest)
    mask_crop = warp_perspective(Float64.(mask), T_mask, target_size; mode=:nearest) .> 0.5

    # Update K for mask crop
    K_homo = [K [0; 0; 1]]
    K_crop_mask = crop_info_mask.T_forward * K_homo
    K_crop_mask = K_crop_mask[1:3, 1:3]

    # Convert to XYZ
    xyz_crop_mask = depth_to_xyz(depth_crop_mask, K_crop_mask; debug=debug && true)

    # Normalize XYZ if we have object info
    if mesh_radius !== nothing
        pose_center = pose[1:3, 4]
        if debug
            println("Normalizing with pose center: ", pose_center)
            println("Mesh radius: ", mesh_radius)
        end
        xyz_norm_mask = normalize_xyz(xyz_crop_mask, pose_center, mesh_radius)
    else
        xyz_norm_mask = xyz_crop_mask
    end

    # Step 3: Pose-based cropping
    if mesh_diameter !== nothing
        T_pose, crop_info_pose = compute_pose_based_crop(pose, K, mesh_diameter;
                                                         crop_ratio=1.5,
                                                         target_size=target_size,
                                                         debug=debug)

        rgb_crop_pose = warp_perspective(rgb, T_pose, target_size; mode=:linear)
        depth_crop_pose = warp_perspective(depth_filtered, T_pose, target_size; mode=:nearest)

        # Update K for pose crop
        K_crop_pose = crop_info_pose.T_forward * K_homo
        K_crop_pose = K_crop_pose[1:3, 1:3]

        # Convert to XYZ
        xyz_crop_pose = depth_to_xyz(depth_crop_pose, K_crop_pose; debug=false)

        if mesh_radius !== nothing
            xyz_norm_pose = normalize_xyz(xyz_crop_pose, pose_center, mesh_radius)
        else
            xyz_norm_pose = xyz_crop_pose
        end
    else
        rgb_crop_pose = nothing
        depth_crop_pose = nothing
        xyz_crop_pose = nothing
        xyz_norm_pose = nothing
        crop_info_pose = nothing
        K_crop_pose = nothing
    end

    return (
        # Mask-based crop
        rgb_mask = rgb_crop_mask,
        depth_mask = depth_crop_mask,
        xyz_mask = xyz_norm_mask,
        mask = mask_crop,
        K_mask = K_crop_mask,
        crop_info_mask = crop_info_mask,

        # Pose-based crop
        rgb_pose = rgb_crop_pose,
        depth_pose = depth_crop_pose,
        xyz_pose = xyz_norm_pose,
        K_pose = K_crop_pose,
        crop_info_pose = crop_info_pose,

        # Filtered depth
        depth_filtered = depth_filtered
    )
end

"""
    draw_crop_box(img, crop_box; color=[1.0, 0.0, 0.0], thickness=3)

Draw crop box on image.
"""
function draw_crop_box(img, crop_box; color=[1.0, 0.0, 0.0], thickness=3)
    img_with_box = copy(img)
    top, bottom, left, right = crop_box

    H, W = size(img)[1:2]

    # Ensure bounds
    top = max(1, floor(Int, top))
    bottom = min(H, ceil(Int, bottom))
    left = max(1, floor(Int, left))
    right = min(W, ceil(Int, right))

    # Draw box
    for t in 0:thickness-1
        # Horizontal lines
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

        # Vertical lines
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

"""
    visualize_preprocessing_stages(rgb, depth, mask, processed)

Create comprehensive visualization of all preprocessing stages.
"""
function visualize_preprocessing_stages(rgb, depth, mask, processed)
    # Create figure with subplots
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
    rgb_with_mask_box = draw_crop_box(rgb, processed.crop_info_mask.crop_box;
                                     color=[1.0, 0.0, 0.0])
    plot!(fig[4], colorview(RGB, permutedims(rgb_with_mask_box, (3, 1, 2))),
          title="RGB with Mask Crop Box", axis=false)

    plot!(fig[5], colorview(RGB, permutedims(processed.rgb_mask, (3, 1, 2))),
          title="Mask Cropped RGB", axis=false)

    valid_depth_mask = processed.depth_mask .> 0
    if any(valid_depth_mask)
        depth_range = extrema(processed.depth_mask[valid_depth_mask])
        heatmap!(fig[6], processed.depth_mask,
                 title="Mask Cropped Depth\n$(round(depth_range[1], digits=3))-$(round(depth_range[2], digits=3))m",
                 aspect_ratio=:equal, yflip=true, colorbar=true, clim=(0, depth_range[2]))
    else
        heatmap!(fig[6], processed.depth_mask,
                 title="Mask Cropped Depth",
                 aspect_ratio=:equal, yflip=true, colorbar=true)
    end

    # Row 3: XYZ channels from mask crop
    for (i, (channel, name)) in enumerate(zip(1:3, ["X", "Y", "Z"]))
        data = processed.xyz_mask[:, :, channel]
        valid = processed.xyz_mask[:, :, 3] .> 0.001

        if any(valid)
            valid_data = data[valid]
            if !isempty(valid_data)
                vrange = extrema(valid_data)
                # Show actual values in title
                title_str = "Mask $name\n$(round(vrange[1], digits=2)) to $(round(vrange[2], digits=2))"

                # Use diverging colormap for X and Y to show sign
                if channel <= 2
                    # For X: should be negative on left, positive on right
                    # For Y: should be negative on top, positive on bottom
                    heatmap!(fig[6 + i], data,
                             title=title_str,
                             aspect_ratio=:equal, yflip=true, colorbar=true,
                             clim=vrange, color=:RdBu)
                else
                    heatmap!(fig[6 + i], data,
                             title=title_str,
                             aspect_ratio=:equal, yflip=true, colorbar=true,
                             clim=vrange, color=:RdBu)
                end
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

    # Save first figure
    savefig(fig, "figures/preprocessing_mask_crop.png")

    # Create second figure for pose-based crop if available
    if processed.rgb_pose !== nothing
        fig2 = plot(layout=(2, 3), size=(1400, 800), titlefontsize=8, dpi=100)

        # Row 1: Pose-based crop
        rgb_with_pose_box = draw_crop_box(rgb, processed.crop_info_pose.crop_box;
                                         color=[0.0, 1.0, 0.0])
        plot!(fig2[1], colorview(RGB, permutedims(rgb_with_pose_box, (3, 1, 2))),
              title="RGB with Pose Crop Box", axis=false)

        plot!(fig2[2], colorview(RGB, permutedims(processed.rgb_pose, (3, 1, 2))),
              title="Pose Cropped RGB", axis=false)

        valid_depth_pose = processed.depth_pose .> 0
        if any(valid_depth_pose)
            depth_range = extrema(processed.depth_pose[valid_depth_pose])
            heatmap!(fig2[3], processed.depth_pose,
                     title="Pose Cropped Depth\n$(round(depth_range[1], digits=3))-$(round(depth_range[2], digits=3))m",
                     aspect_ratio=:equal, yflip=true, colorbar=true, clim=(0, depth_range[2]))
        else
            heatmap!(fig2[3], processed.depth_pose,
                     title="Pose Cropped Depth",
                     aspect_ratio=:equal, yflip=true, colorbar=true)
        end

        # Row 2: XYZ channels from pose crop
        for (i, (channel, name)) in enumerate(zip(1:3, ["X", "Y", "Z"]))
            data = processed.xyz_pose[:, :, channel]
            valid = processed.xyz_pose[:, :, 3] .> 0.001

            if any(valid)
                valid_data = data[valid]
                if !isempty(valid_data)
                    vrange = extrema(valid_data)
                    title_str = "Pose $name\n$(round(vrange[1], digits=2)) to $(round(vrange[2], digits=2))"

                    if channel <= 2
                        heatmap!(fig2[3 + i], data,
                                 title=title_str,
                                 aspect_ratio=:equal, yflip=true, colorbar=true,
                                 clim=vrange, color=:RdBu)
                    else
                        heatmap!(fig2[3 + i], data,
                                 title=title_str,
                                 aspect_ratio=:equal, yflip=true, colorbar=true,
                                 clim=vrange, color=:RdBu)
                    end
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

end # module ImageProcessor
