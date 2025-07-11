# CPUImageProcessor.jl
# CPU implementation of batch image processing operations

using Statistics, LinearAlgebra

# Core depth preprocessing functions
function erode_depth(depth; radius=2, depth_diff_threshold=0.001, ratio_threshold=0.8, zfar=100.0)
    N, H, W = size(depth)
    output = copy(depth)

    for n in 1:N
        for h in 1:H, w in 1:W
            d_ori = depth[n, h, w]
            if d_ori < 0.001 || d_ori >= zfar
                output[n, h, w] = 0.0
                continue
            end

            bad_cnt = 0.0
            total = 0.0

            for u in max(1, w-radius):min(W, w+radius)
                for v in max(1, h-radius):min(H, h+radius)
                    cur_depth = depth[n, v, u]
                    total += 1.0
                    if cur_depth < 0.001 || cur_depth >= zfar || abs(cur_depth - d_ori) > depth_diff_threshold
                        bad_cnt += 1.0
                    end
                end
            end

            if bad_cnt / total > ratio_threshold
                output[n, h, w] = 0.0
            end
        end
    end

    return output
end

function bilateral_filter_depth(depth; radius=2, sigma_d=2.0, sigma_r=100000.0, zfar=100.0)
    N, H, W = size(depth)
    output = zeros(Float64, N, H, W)

    for n in 1:N
        for h in 1:H, w in 1:W
            if depth[n, h, w] < 0.001
                continue
            end

            mean_depth = 0.0
            num_valid = 0

            for u in max(1, w-radius):min(W, w+radius)
                for v in max(1, h-radius):min(H, h+radius)
                    cur_depth = depth[n, v, u]
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

            depthCenter = depth[n, h, w]
            sum_weight = 0.0
            sum = 0.0

            for u in max(1, w-radius):min(W, w+radius)
                for v in max(1, h-radius):min(H, h+radius)
                    cur_depth = depth[n, v, u]
                    if cur_depth >= 0.001 && cur_depth < zfar && abs(cur_depth - mean_depth) < 0.01
                        weight = exp(-Float64((u-w)^2 + (h-v)^2) / (2.0*sigma_d*sigma_d) -
                                    (depthCenter-cur_depth)^2/(2.0*sigma_r*sigma_r))
                        sum_weight += weight
                        sum += weight * cur_depth
                    end
                end
            end

            if sum_weight > 0 && num_valid > 0
                output[n, h, w] = sum / sum_weight
            end
        end
    end

    return output
end

# Batch transformation functions
function compute_crop_transform_mask(masks; crop_ratio=1.2, target_size=(160, 160))
    N = size(masks, 1)
    T_invs = zeros(Float64, N, 3, 3)
    crop_infos = []

    for n in 1:N
        mask = masks[n, :, :]
        mask_indices = findall(mask)

        if isempty(mask_indices)
            # Default transform for empty mask
            T_inv = Matrix{Float64}(I, 3, 3)
            T_invs[n, :, :] = T_inv
            push!(crop_infos, (
                center = (240.0, 320.0),
                bbox = (1, 480, 1, 640),
                crop_box = (1.0, 480.0, 1.0, 640.0),
                crop_size = 640.0,
                T_inv = T_inv,
                T_forward = T_inv
            ))
            continue
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

        T_invs[n, :, :] = T_inv

        push!(crop_infos, (
            center = (vc, uc),
            bbox = (vmin, vmax, umin, umax),
            crop_box = (top+1, top+crop_size+1, left+1, left+crop_size+1),
            crop_size = crop_size,
            T_inv = T_inv,
            T_forward = T_forward
        ))
    end

    return T_invs, crop_infos
end

function compute_pose_based_crop(poses, Ks, mesh_diameters; crop_ratio=1.5, target_size=(160, 160))
    N = size(poses, 1)
    T_invs = zeros(Float64, N, 3, 3)
    crop_infos = []

    for n in 1:N
        pose = poses[n, :, :]
        K = Ks[n, :, :]
        mesh_diameter = mesh_diameters[n]

        center_3d = pose[1:3, 4]
        radius = mesh_diameter * crop_ratio / 2.0

        # Project center to image
        pt_2d_homo = K * center_3d
        uc = pt_2d_homo[1] / pt_2d_homo[3] + 1
        vc = pt_2d_homo[2] / pt_2d_homo[3] + 1

        # Project bounding sphere
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

        T_invs[n, :, :] = T_inv

        push!(crop_infos, (
            center = (vc, uc),
            crop_box = (top+1, top+crop_size+1, left+1, left+crop_size+1),
            crop_size = crop_size,
            T_inv = T_inv,
            T_forward = T_forward
        ))
    end

    return T_invs, crop_infos
end

function warp_perspective(images, T_invs, target_size; mode=:linear)
    N = size(images, 1)
    H_out, W_out = target_size

    if ndims(images) == 4  # RGB images
        C = size(images, 4)
        output = zeros(eltype(images), N, H_out, W_out, C)
    else  # Depth/mask
        output = zeros(eltype(images), N, H_out, W_out)
    end

    for n in 1:N
        T_inv = T_invs[n, :, :]

        for v_out in 1:H_out, u_out in 1:W_out
            pt_out = [u_out - 1; v_out - 1; 1.0]
            pt_in = T_inv * pt_out
            u_in = pt_in[1] / pt_in[3] + 1
            v_in = pt_in[2] / pt_in[3] + 1

            if mode == :nearest
                u_nearest = round(Int, u_in)
                v_nearest = round(Int, v_in)

                if 1 <= u_nearest <= size(images, 3) && 1 <= v_nearest <= size(images, 2)
                    if ndims(images) == 4
                        output[n, v_out, u_out, :] = images[n, v_nearest, u_nearest, :]
                    else
                        output[n, v_out, u_out] = images[n, v_nearest, u_nearest]
                    end
                end
            else  # bilinear
                u0 = floor(Int, u_in)
                v0 = floor(Int, v_in)
                u1 = u0 + 1
                v1 = v0 + 1

                if u0 >= 1 && u1 <= size(images, 3) && v0 >= 1 && v1 <= size(images, 2)
                    wu1 = u_in - u0
                    wv1 = v_in - v0
                    wu0 = 1 - wu1
                    wv0 = 1 - wv1

                    if ndims(images) == 4
                        for c in 1:C
                            val = wv0 * wu0 * images[n, v0, u0, c] +
                                  wv0 * wu1 * images[n, v0, u1, c] +
                                  wv1 * wu0 * images[n, v1, u0, c] +
                                  wv1 * wu1 * images[n, v1, u1, c]
                            output[n, v_out, u_out, c] = val
                        end
                    else
                        output[n, v_out, u_out] = wv0 * wu0 * images[n, v0, u0] +
                                                  wv0 * wu1 * images[n, v0, u1] +
                                                  wv1 * wu0 * images[n, v1, u0] +
                                                  wv1 * wu1 * images[n, v1, u1]
                    end
                end
            end
        end
    end

    return output
end

function depth_to_xyz(depths, Ks)
    N, H, W = size(depths)
    xyz = zeros(Float64, N, H, W, 3)

    for n in 1:N
        K = Ks[n, :, :]
        fx = K[1,1]
        fy = K[2,2]
        cx = K[1,3]
        cy = K[2,3]

        for v in 1:H, u in 1:W
            z = depths[n, v, u]
            if z > 0.001
                x = (Float64(u - 1) - cx) * z / fx
                y = (Float64(v - 1) - cy) * z / fy
                xyz[n, v, u, 1] = x
                xyz[n, v, u, 2] = y
                xyz[n, v, u, 3] = z
            end
        end
    end

    return xyz
end

function normalize_xyz(xyz, centers, mesh_radii; bound=2.0)
    N, H, W, _ = size(xyz)
    normalized = zeros(size(xyz))

    for n in 1:N
        center = centers[n, :]
        mesh_radius = mesh_radii[n]

        invalid = xyz[n, :, :, 3] .< 0.001

        for i in 1:3
            normalized[n, :, :, i] = (xyz[n, :, :, i] .- center[i]) ./ mesh_radius
        end

        invalid = invalid .| (abs.(normalized[n, :, :, :]) .>= bound)

        for v in 1:H, u in 1:W
            if invalid[v, u, 1] || invalid[v, u, 2] || invalid[v, u, 3]
                normalized[n, v, u, :] .= 0.0
            end
        end
    end

    return normalized
end

function guess_object_center(depths, masks, Ks)
    N = size(depths, 1)
    centers = zeros(Float64, N, 3)

    for n in 1:N
        depth = depths[n, :, :]
        mask = masks[n, :, :]
        K = Ks[n, :, :]

        indices = findall(mask)
        if isempty(indices)
            continue
        end

        vs = [idx[1] for idx in indices]
        us = [idx[2] for idx in indices]

        uc = (minimum(us) + maximum(us)) / 2.0
        vc = (minimum(vs) + maximum(vs)) / 2.0

        valid = mask .& (depth .>= 0.001)
        if !any(valid)
            continue
        end

        zc = median(depth[valid])

        # Convert from pixel to camera coordinates
        center = inv(K) * [uc - 1; vc - 1; 1.0] * zc
        centers[n, :] = center
    end

    return centers
end

# Main processing functions
function process_mask_crop(rgbs, depths, masks, Ks;
                          crop_ratio=1.2,
                          target_size=(160, 160),
                          mesh_radii=nothing,
                          estimate_centers=true,
                          object_centers=nothing)

    N = size(rgbs, 1)

    # Preprocess depth
    depths_filtered = erode_depth(depths; radius=2)
    depths_filtered = bilateral_filter_depth(depths_filtered; radius=2)

    # Compute crop transforms
    T_invs, crop_infos = compute_crop_transform_mask(masks;
                                                    crop_ratio=crop_ratio,
                                                    target_size=target_size)

    # Apply warping
    rgbs_crop = warp_perspective(rgbs, T_invs, target_size; mode=:linear)
    depths_crop = warp_perspective(depths_filtered, T_invs, target_size; mode=:nearest)
    masks_crop = warp_perspective(Float64.(masks), T_invs, target_size; mode=:nearest) .> 0.5

    # Update K for crops
    Ks_crop = zeros(Float64, N, 3, 3)
    for n in 1:N
        K_homo = [Ks[n, :, :] [0; 0; 1]]
        K_crop = crop_infos[n].T_forward * K_homo
        Ks_crop[n, :, :] = K_crop[1:3, 1:3]
    end

    # Convert to XYZ
    xyz_crop = depth_to_xyz(depths_crop, Ks_crop)

    # Normalize XYZ if we have object info
    if mesh_radii !== nothing
        if estimate_centers && object_centers === nothing
            object_centers = guess_object_center(depths, masks, Ks)
        elseif object_centers === nothing
            object_centers = zeros(Float64, N, 3)
        end
        xyz_norm = normalize_xyz(xyz_crop, object_centers, mesh_radii)
    else
        xyz_norm = xyz_crop
    end

    return (
        rgb = rgbs_crop,
        depth = depths_crop,
        xyz = xyz_norm,
        mask = masks_crop,
        K = Ks_crop,
        crop_infos = crop_infos,
        object_centers = object_centers
    )
end

function process_pose_crop(rgbs, depths, poses, Ks, mesh_diameters;
                          crop_ratio=1.5,
                          target_size=(160, 160),
                          mesh_radii=nothing)

    N = size(rgbs, 1)

    # Preprocess depth
    depths_filtered = erode_depth(depths; radius=2)
    depths_filtered = bilateral_filter_depth(depths_filtered; radius=2)

    # Compute crop transforms
    T_invs, crop_infos = compute_pose_based_crop(poses, Ks, mesh_diameters;
                                                 crop_ratio=crop_ratio,
                                                 target_size=target_size)

    # Apply warping
    rgbs_crop = warp_perspective(rgbs, T_invs, target_size; mode=:linear)
    depths_crop = warp_perspective(depths_filtered, T_invs, target_size; mode=:nearest)

    # Update K for crops
    Ks_crop = zeros(Float64, N, 3, 3)
    for n in 1:N
        K_homo = [Ks[n, :, :] [0; 0; 1]]
        K_crop = crop_infos[n].T_forward * K_homo
        Ks_crop[n, :, :] = K_crop[1:3, 1:3]
    end

    # Convert to XYZ
    xyz_crop = depth_to_xyz(depths_crop, Ks_crop)

    # Normalize XYZ if we have object info
    if mesh_radii !== nothing
        pose_centers = poses[:, 1:3, 4]
        xyz_norm = normalize_xyz(xyz_crop, pose_centers, mesh_radii)
    else
        xyz_norm = xyz_crop
    end

    return (
        rgb = rgbs_crop,
        depth = depths_crop,
        xyz = xyz_norm,
        K = Ks_crop,
        crop_infos = crop_infos
    )
end
