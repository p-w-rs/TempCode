# ImageProcessor.jl
module ImageProcessor

export process_mask_crop, process_pose_crop
export visualize_preprocessing_stages
export guess_object_center

using Images, ImageTransformations, ImageFiltering
using CoordinateTransformations, Interpolations
using Statistics, LinearAlgebra
using Plots

# Core depth preprocessing functions
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

# Core transformation functions
function compute_crop_transform_mask(mask; crop_ratio=1.2, target_size=(160, 160))
    mask_indices = findall(mask)
    if isempty(mask_indices)
        throw(ArgumentError("Empty mask provided"))
    end

    vs = [idx[1] for idx in mask_indices]
    us = [idx[2] for idx in mask_indices]

    umin, umax = extrema(us)
    vmin, vmax = extrema(vs)

    uc = (umin + umax) / 2.0
    vc = (vmin + vmax) / 2.0

    uc_0 = uc - 1
    vc_0 = vc - 1

    height = vmax - vmin
    width = umax - umin

    crop_size = max(height, width) * crop_ratio
    half_size = crop_size / 2.0

    left = uc_0 - half_size
    top = vc_0 - half_size

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
        bbox = (vmin, vmax, umin, umax),
        crop_box = (top+1, top+crop_size+1, left+1, left+crop_size+1),
        crop_size = crop_size,
        T_inv = T_inv,
        T_forward = T_forward
    )

    return T_inv, crop_info
end

function compute_pose_based_crop(pose, K, mesh_diameter; crop_ratio=1.5, target_size=(160, 160))
    center_3d = pose[1:3, 4]
    radius = mesh_diameter * crop_ratio / 2.0

    # Project center to image
    pt_2d_homo = K * center_3d
    uc = pt_2d_homo[1] / pt_2d_homo[3] + 1  # Convert to 1-indexed
    vc = pt_2d_homo[2] / pt_2d_homo[3] + 1

    # Project bounding sphere to get radius in pixels
    points_3d = [
        center_3d,
        center_3d + [radius, 0, 0],
        center_3d + [-radius, 0, 0],
        center_3d + [0, radius, 0],
        center_3d + [0, -radius, 0],
    ]

    us = Float64[]
    vs = Float64[]
    for pt_3d in points_3d
        pt_2d_homo = K * pt_3d
        u = pt_2d_homo[1] / pt_2d_homo[3] + 1
        v = pt_2d_homo[2] / pt_2d_homo[3] + 1
        push!(us, u)
        push!(vs, v)
    end

    radius_px = 0.0
    for i in 2:length(us)
        dist = sqrt((us[i] - uc)^2 + (vs[i] - vc)^2)
        radius_px = max(radius_px, dist)
    end

    uc_0 = uc - 1
    vc_0 = vc - 1

    left = uc_0 - radius_px
    top = vc_0 - radius_px
    crop_size = 2 * radius_px

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
        crop_box = (top+1, top+crop_size+1, left+1, left+crop_size+1),
        crop_size = crop_size,
        T_inv = T_inv,
        T_forward = T_forward
    )

    return T_inv, crop_info
end

function warp_perspective(image, T_inv, target_size; mode=:linear)
    H_out, W_out = target_size

    if ndims(image) == 3
        output = zeros(eltype(image), H_out, W_out, size(image, 3))
    else
        output = zeros(eltype(image), H_out, W_out)
    end

    for v_out in 1:H_out, u_out in 1:W_out
        pt_out = [u_out - 1; v_out - 1; 1.0]
        pt_in = T_inv * pt_out
        u_in = pt_in[1] / pt_in[3] + 1
        v_in = pt_in[2] / pt_in[3] + 1

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

function depth_to_xyz(depth, K)
    H, W = size(depth)
    xyz = zeros(Float64, H, W, 3)

    fx = K[1,1]
    fy = K[2,2]
    cx = K[1,3]
    cy = K[2,3]

    for v in 1:H, u in 1:W
        z = depth[v, u]
        if z > 0.001
            x = (Float64(u - 1) - cx) * z / fx
            y = (Float64(v - 1) - cy) * z / fy
            xyz[v, u, 1] = x
            xyz[v, u, 2] = y
            xyz[v, u, 3] = z
        end
    end

    return xyz
end

function normalize_xyz(xyz, center, mesh_radius; bound=2.0)
    normalized = zeros(size(xyz))
    H, W = size(xyz, 1), size(xyz, 2)

    invalid = xyz[:, :, 3] .< 0.001

    for i in 1:3
        normalized[:, :, i] = (xyz[:, :, i] .- center[i]) ./ mesh_radius
    end

    invalid = invalid .| (abs.(normalized) .>= bound)

    for v in 1:H, u in 1:W
        if invalid[v, u, 1] || invalid[v, u, 2] || invalid[v, u, 3]
            normalized[v, u, :] .= 0.0
        end
    end

    return normalized
end

# Estimate object center from mask and depth (for when we don't have ground truth pose)
function guess_object_center(depth, mask, K)
    indices = findall(mask)
    if isempty(indices)
        return zeros(3)
    end

    # Extract row and column indices
    vs = [idx[1] for idx in indices]
    us = [idx[2] for idx in indices]

    uc = (minimum(us) + maximum(us)) / 2.0
    vc = (minimum(vs) + maximum(vs)) / 2.0

    valid = mask .& (depth .>= 0.001)
    if !any(valid)
        return zeros(3)
    end

    zc = median(depth[valid])

    # Convert from pixel to camera coordinates
    # Using 0-indexed pixel coordinates for camera model
    center = inv(K) * [uc - 1; vc - 1; 1.0] * zc

    return center
end

# Main processing functions
function process_mask_crop(rgb, depth, mask, K;
                          crop_ratio=1.2,
                          target_size=(160, 160),
                          mesh_radius=nothing,
                          estimate_center=true,
                          object_center=nothing)

    # Preprocess depth
    depth_filtered = erode_depth(depth; radius=2)
    depth_filtered = bilateral_filter_depth(depth_filtered; radius=2)

    # Compute crop transform
    T_inv, crop_info = compute_crop_transform_mask(mask;
                                                   crop_ratio=crop_ratio,
                                                   target_size=target_size)

    # Apply warping
    rgb_crop = warp_perspective(rgb, T_inv, target_size; mode=:linear)
    depth_crop = warp_perspective(depth_filtered, T_inv, target_size; mode=:nearest)
    mask_crop = warp_perspective(Float64.(mask), T_inv, target_size; mode=:nearest) .> 0.5

    # Update K for crop
    K_homo = [K [0; 0; 1]]
    K_crop = crop_info.T_forward * K_homo
    K_crop = K_crop[1:3, 1:3]

    # Convert to XYZ
    xyz_crop = depth_to_xyz(depth_crop, K_crop)

    # Normalize XYZ if we have object info
    if mesh_radius !== nothing
        if estimate_center && object_center === nothing
            # Estimate center from original mask and depth
            object_center = guess_object_center(depth, mask, K)
        elseif object_center === nothing
            object_center = zeros(3)
        end
        xyz_norm = normalize_xyz(xyz_crop, object_center, mesh_radius)
    else
        xyz_norm = xyz_crop
    end

    return (
        rgb = rgb_crop,
        depth = depth_crop,
        xyz = xyz_norm,
        mask = mask_crop,
        K = K_crop,
        crop_info = crop_info,
        object_center = object_center
    )
end

function process_pose_crop(rgb, depth, pose, K, mesh_diameter;
                          crop_ratio=1.5,
                          target_size=(160, 160),
                          mesh_radius=nothing)

    # Preprocess depth
    depth_filtered = erode_depth(depth; radius=2)
    depth_filtered = bilateral_filter_depth(depth_filtered; radius=2)

    # Compute crop transform
    T_inv, crop_info = compute_pose_based_crop(pose, K, mesh_diameter;
                                              crop_ratio=crop_ratio,
                                              target_size=target_size)

    # Apply warping
    rgb_crop = warp_perspective(rgb, T_inv, target_size; mode=:linear)
    depth_crop = warp_perspective(depth_filtered, T_inv, target_size; mode=:nearest)

    # Update K for crop
    K_homo = [K [0; 0; 1]]
    K_crop = crop_info.T_forward * K_homo
    K_crop = K_crop[1:3, 1:3]

    # Convert to XYZ
    xyz_crop = depth_to_xyz(depth_crop, K_crop)

    # Normalize XYZ if we have object info
    if mesh_radius !== nothing
        pose_center = pose[1:3, 4]
        xyz_norm = normalize_xyz(xyz_crop, pose_center, mesh_radius)
    else
        xyz_norm = xyz_crop
    end

    return (
        rgb = rgb_crop,
        depth = depth_crop,
        xyz = xyz_norm,
        K = K_crop,
        crop_info = crop_info
    )
end

# Visualization functions (optional, for debugging)
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
    rgb_with_mask_box = draw_crop_box(rgb, mask_result.crop_info.crop_box;
                                     color=[1.0, 0.0, 0.0])
    plot!(fig[4], colorview(RGB, permutedims(rgb_with_mask_box, (3, 1, 2))),
          title="RGB with Mask Crop Box", axis=false)

    plot!(fig[5], colorview(RGB, permutedims(mask_result.rgb, (3, 1, 2))),
          title="Mask Cropped RGB", axis=false)

    valid_depth_mask = mask_result.depth .> 0
    if any(valid_depth_mask)
        depth_range = extrema(mask_result.depth[valid_depth_mask])
        heatmap!(fig[6], mask_result.depth,
                 title="Mask Cropped Depth\n$(round(depth_range[1], digits=3))-$(round(depth_range[2], digits=3))m",
                 aspect_ratio=:equal, yflip=true, colorbar=true, clim=(0, depth_range[2]))
    else
        heatmap!(fig[6], mask_result.depth,
                 title="Mask Cropped Depth",
                 aspect_ratio=:equal, yflip=true, colorbar=true)
    end

    # Row 3: XYZ channels from mask crop
    for (i, (channel, name)) in enumerate(zip(1:3, ["X", "Y", "Z"]))
        data = mask_result.xyz[:, :, channel]
        valid = mask_result.xyz[:, :, 3] .> 0.001

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

        rgb_with_pose_box = draw_crop_box(rgb, pose_result.crop_info.crop_box;
                                         color=[0.0, 1.0, 0.0])
        plot!(fig2[1], colorview(RGB, permutedims(rgb_with_pose_box, (3, 1, 2))),
              title="RGB with Pose Crop Box", axis=false)

        plot!(fig2[2], colorview(RGB, permutedims(pose_result.rgb, (3, 1, 2))),
              title="Pose Cropped RGB", axis=false)

        valid_depth_pose = pose_result.depth .> 0
        if any(valid_depth_pose)
            depth_range = extrema(pose_result.depth[valid_depth_pose])
            heatmap!(fig2[3], pose_result.depth,
                     title="Pose Cropped Depth\n$(round(depth_range[1], digits=3))-$(round(depth_range[2], digits=3))m",
                     aspect_ratio=:equal, yflip=true, colorbar=true, clim=(0, depth_range[2]))
        else
            heatmap!(fig2[3], pose_result.depth,
                     title="Pose Cropped Depth",
                     aspect_ratio=:equal, yflip=true, colorbar=true)
        end

        for (i, (channel, name)) in enumerate(zip(1:3, ["X", "Y", "Z"]))
            data = pose_result.xyz[:, :, channel]
            valid = pose_result.xyz[:, :, 3] .> 0.001

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

end # module ImageProcessor
