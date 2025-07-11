# CUDAImageProcessor.jl - Optimized Version
# CUDA implementation with performance optimizations

using CUDA
using Statistics, LinearAlgebra

# Use Float32 for GPU operations (much faster than Float64)
const GPUFloat = Float32

# Pre-allocated buffers for common sizes
const BUFFER_CACHE = Dict{Tuple{Int,Int,Int}, CuArray}()

function get_buffer(T::Type, dims...)
    key = (dims...,)
    if !haskey(BUFFER_CACHE, key)
        BUFFER_CACHE[key] = CUDA.zeros(T, dims...)
    end
    return BUFFER_CACHE[key]
end

# Optimized erosion kernel using shared memory
function erode_depth_kernel_optimized!(output, depth, radius, depth_diff_threshold, ratio_threshold, zfar, N, H, W)
    # Shared memory for tile processing
    tile_size = 16
    shared_size = tile_size + 2 * radius
    shared = @cuDynamicSharedMem(GPUFloat, (shared_size, shared_size))

    # Thread and block indices
    tx = threadIdx().x
    ty = threadIdx().y
    bx = blockIdx().x
    by = blockIdx().y
    n = blockIdx().z

    # Global position
    gx = (bx - 1) * tile_size + tx
    gy = (by - 1) * tile_size + ty

    if n > N
        return nothing
    end

    # Load data into shared memory (with padding for radius)
    for dy in 0:1, dx in 0:1
        local_y = ty + dy * blockDim().y
        local_x = tx + dx * blockDim().x

        if local_y <= shared_size && local_x <= shared_size
            global_y = (by - 1) * tile_size + local_y - radius
            global_x = (bx - 1) * tile_size + local_x - radius

            if global_y >= 1 && global_y <= H && global_x >= 1 && global_x <= W
                shared[local_y, local_x] = depth[n, global_y, global_x]
            else
                shared[local_y, local_x] = 0.0f0
            end
        end
    end

    sync_threads()

    # Process only if within bounds
    if gx <= W && gy <= H
        local_y = ty + radius
        local_x = tx + radius

        d_ori = shared[local_y, local_x]

        if d_ori < 0.001f0 || d_ori >= zfar
            output[n, gy, gx] = 0.0f0
            return nothing
        end

        bad_cnt = 0.0f0
        total = 0.0f0

        for dy in -radius:radius, dx in -radius:radius
            cur_depth = shared[local_y + dy, local_x + dx]
            total += 1.0f0
            if cur_depth < 0.001f0 || cur_depth >= zfar || abs(cur_depth - d_ori) > depth_diff_threshold
                bad_cnt += 1.0f0
            end
        end

        if bad_cnt / total > ratio_threshold
            output[n, gy, gx] = 0.0f0
        else
            output[n, gy, gx] = d_ori
        end
    end

    return nothing
end

# Optimized bilateral filter using shared memory
function bilateral_filter_kernel_optimized!(output, depth, radius, sigma_d_inv, sigma_r_inv, zfar, N, H, W)
    tile_size = 16
    shared_size = tile_size + 2 * radius
    shared = @cuDynamicSharedMem(GPUFloat, (shared_size, shared_size))

    tx = threadIdx().x
    ty = threadIdx().y
    bx = blockIdx().x
    by = blockIdx().y
    n = blockIdx().z

    gx = (bx - 1) * tile_size + tx
    gy = (by - 1) * tile_size + ty

    if n > N
        return nothing
    end

    # Load tile with padding
    for dy in 0:1, dx in 0:1
        local_y = ty + dy * blockDim().y
        local_x = tx + dx * blockDim().x

        if local_y <= shared_size && local_x <= shared_size
            global_y = (by - 1) * tile_size + local_y - radius
            global_x = (bx - 1) * tile_size + local_x - radius

            if global_y >= 1 && global_y <= H && global_x >= 1 && global_x <= W
                shared[local_y, local_x] = depth[n, global_y, global_x]
            else
                shared[local_y, local_x] = 0.0f0
            end
        end
    end

    sync_threads()

    if gx <= W && gy <= H
        local_y = ty + radius
        local_x = tx + radius

        center_depth = shared[local_y, local_x]

        if center_depth < 0.001f0
            output[n, gy, gx] = 0.0f0
            return nothing
        end

        # First pass: compute mean
        mean_depth = 0.0f0
        num_valid = 0

        for dy in -radius:radius, dx in -radius:radius
            cur_depth = shared[local_y + dy, local_x + dx]
            if cur_depth >= 0.001f0 && cur_depth < zfar
                num_valid += 1
                mean_depth += cur_depth
            end
        end

        if num_valid == 0
            output[n, gy, gx] = 0.0f0
            return nothing
        end

        mean_depth /= GPUFloat(num_valid)

        # Second pass: bilateral filter
        sum_weight = 0.0f0
        sum = 0.0f0

        for dy in -radius:radius, dx in -radius:radius
            cur_depth = shared[local_y + dy, local_x + dx]
            if cur_depth >= 0.001f0 && cur_depth < zfar && abs(cur_depth - mean_depth) < 0.01f0
                spatial_dist_sq = GPUFloat(dx * dx + dy * dy)
                depth_dist_sq = (center_depth - cur_depth)^2

                weight = exp(-spatial_dist_sq * sigma_d_inv - depth_dist_sq * sigma_r_inv)
                sum_weight += weight
                sum += weight * cur_depth
            end
        end

        if sum_weight > 0.0f0
            output[n, gy, gx] = sum / sum_weight
        else
            output[n, gy, gx] = 0.0f0
        end
    end

    return nothing
end

function erode_depth(depth; radius=2, depth_diff_threshold=0.001, ratio_threshold=0.8, zfar=100.0)
    depth_gpu = CuArray{GPUFloat}(depth)
    N, H, W = size(depth_gpu)
    output = similar(depth_gpu)

    # Use 2D thread blocks for better cache locality
    threads = (16, 16)
    blocks = (
        cld(W, threads[1]),
        cld(H, threads[2]),
        N
    )

    # Calculate shared memory size
    tile_size = 16
    shared_size = tile_size + 2 * radius
    shmem_size = sizeof(GPUFloat) * shared_size * shared_size

    @cuda threads=threads blocks=blocks shmem=shmem_size erode_depth_kernel_optimized!(
        output, depth_gpu, radius, GPUFloat(depth_diff_threshold),
        GPUFloat(ratio_threshold), GPUFloat(zfar), N, H, W
    )

    return Array(output)
end

function bilateral_filter_depth(depth; radius=2, sigma_d=2.0, sigma_r=100000.0, zfar=100.0)
    depth_gpu = CuArray{GPUFloat}(depth)
    N, H, W = size(depth_gpu)
    output = CUDA.zeros(GPUFloat, N, H, W)

    # Precompute inverse sigmas
    sigma_d_inv = GPUFloat(0.5 / (sigma_d * sigma_d))
    sigma_r_inv = GPUFloat(0.5 / (sigma_r * sigma_r))

    threads = (16, 16)
    blocks = (
        cld(W, threads[1]),
        cld(H, threads[2]),
        N
    )

    tile_size = 16
    shared_size = tile_size + 2 * radius
    shmem_size = sizeof(GPUFloat) * shared_size * shared_size

    @cuda threads=threads blocks=blocks shmem=shmem_size bilateral_filter_kernel_optimized!(
        output, depth_gpu, radius, sigma_d_inv, sigma_r_inv, GPUFloat(zfar), N, H, W
    )

    return Array(output)
end

# Fused warp and interpolate kernel
function warp_perspective_kernel_fused!(output, images, T_invs, mode_nearest, N, H_in, W_in, H_out, W_out, C)
    gx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = blockIdx().z

    if gx > W_out || gy > H_out || n > N
        return nothing
    end

    # Load transformation matrix into registers
    t11 = T_invs[n, 1, 1]
    t12 = T_invs[n, 1, 2]
    t13 = T_invs[n, 1, 3]
    t21 = T_invs[n, 2, 1]
    t22 = T_invs[n, 2, 2]
    t23 = T_invs[n, 2, 3]

    # Transform coordinates
    u_out_0 = GPUFloat(gx - 1)
    v_out_0 = GPUFloat(gy - 1)

    u_in = t11 * u_out_0 + t12 * v_out_0 + t13 + 1.0f0
    v_in = t21 * u_out_0 + t22 * v_out_0 + t23 + 1.0f0

    if mode_nearest
        u_nearest = round(Int32, u_in)
        v_nearest = round(Int32, v_in)

        if 1 <= u_nearest <= W_in && 1 <= v_nearest <= H_in
            for c in 1:C
                output[n, gy, gx, c] = images[n, v_nearest, u_nearest, c]
            end
        else
            for c in 1:C
                output[n, gy, gx, c] = 0.0f0
            end
        end
    else
        u0 = floor(Int32, u_in)
        v0 = floor(Int32, v_in)

        if u0 >= 1 && u0 < W_in && v0 >= 1 && v0 < H_in
            u1 = u0 + 1
            v1 = v0 + 1

            wu1 = u_in - GPUFloat(u0)
            wv1 = v_in - GPUFloat(v0)
            wu0 = 1.0f0 - wu1
            wv0 = 1.0f0 - wv1

            for c in 1:C
                val = wv0 * wu0 * images[n, v0, u0, c] +
                      wv0 * wu1 * images[n, v0, u1, c] +
                      wv1 * wu0 * images[n, v1, u0, c] +
                      wv1 * wu1 * images[n, v1, u1, c]
                output[n, gy, gx, c] = val
            end
        else
            for c in 1:C
                output[n, gy, gx, c] = 0.0f0
            end
        end
    end

    return nothing
end

function warp_perspective(images, T_invs, target_size; mode=:linear)
    N = size(images, 1)
    H_out, W_out = target_size

    # Convert to GPU arrays with Float32
    images_gpu = CuArray{GPUFloat}(images)
    T_invs_gpu = CuArray{GPUFloat}(T_invs)

    if ndims(images) == 4  # RGB images
        H_in, W_in, C = size(images)[2:4]
        output_gpu = CUDA.zeros(GPUFloat, N, H_out, W_out, C)
    else  # Depth/mask
        H_in, W_in = size(images)[2:3]
        C = 1
        images_gpu = reshape(images_gpu, N, H_in, W_in, 1)
        output_gpu = CUDA.zeros(GPUFloat, N, H_out, W_out, 1)
    end

    threads = (32, 32)  # Warp size aligned
    blocks = (
        cld(W_out, threads[1]),
        cld(H_out, threads[2]),
        N
    )

    mode_nearest = mode == :nearest

    @cuda threads=threads blocks=blocks warp_perspective_kernel_fused!(
        output_gpu, images_gpu, T_invs_gpu, mode_nearest,
        N, H_in, W_in, H_out, W_out, C
    )

    output = Array(output_gpu)

    if ndims(images) == 3
        output = dropdims(output, dims=4)
    end

    return output
end

# Fused depth to XYZ and normalization kernel
function depth_to_xyz_normalized_kernel!(xyz, depths, Ks, centers, mesh_radii, bound, N, H, W)
    gx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = blockIdx().z

    if gx > W || gy > H || n > N
        return nothing
    end

    z = depths[n, gy, gx]

    if z > 0.001f0
        fx = Ks[n, 1, 1]
        fy = Ks[n, 2, 2]
        cx = Ks[n, 1, 3]
        cy = Ks[n, 2, 3]

        # Convert to camera coordinates
        x = (GPUFloat(gx - 1) - cx) * z / fx
        y = (GPUFloat(gy - 1) - cy) * z / fy

        # Normalize if centers provided
        if centers != nothing
            center_x = centers[n, 1]
            center_y = centers[n, 2]
            center_z = centers[n, 3]
            mesh_radius = mesh_radii[n]

            norm_x = (x - center_x) / mesh_radius
            norm_y = (y - center_y) / mesh_radius
            norm_z = (z - center_z) / mesh_radius

            if abs(norm_x) < bound && abs(norm_y) < bound && abs(norm_z) < bound
                xyz[n, gy, gx, 1] = norm_x
                xyz[n, gy, gx, 2] = norm_y
                xyz[n, gy, gx, 3] = norm_z
            else
                xyz[n, gy, gx, 1] = 0.0f0
                xyz[n, gy, gx, 2] = 0.0f0
                xyz[n, gy, gx, 3] = 0.0f0
            end
        else
            xyz[n, gy, gx, 1] = x
            xyz[n, gy, gx, 2] = y
            xyz[n, gy, gx, 3] = z
        end
    else
        xyz[n, gy, gx, 1] = 0.0f0
        xyz[n, gy, gx, 2] = 0.0f0
        xyz[n, gy, gx, 3] = 0.0f0
    end

    return nothing
end

function depth_to_xyz(depths, Ks)
    depths_gpu = CuArray{GPUFloat}(depths)
    Ks_gpu = CuArray{GPUFloat}(Ks)
    N, H, W = size(depths)
    xyz_gpu = CUDA.zeros(GPUFloat, N, H, W, 3)

    threads = (32, 32)
    blocks = (
        cld(W, threads[1]),
        cld(H, threads[2]),
        N
    )

    @cuda threads=threads blocks=blocks depth_to_xyz_normalized_kernel!(
        xyz_gpu, depths_gpu, Ks_gpu, nothing, nothing, GPUFloat(Inf), N, H, W
    )

    return Array(xyz_gpu)
end

function normalize_xyz(xyz, centers, mesh_radii; bound=2.0)
    # Fused version - just convert and normalize in one pass
    xyz_gpu = CuArray{GPUFloat}(xyz)
    centers_gpu = CuArray{GPUFloat}(centers)
    mesh_radii_gpu = CuArray{GPUFloat}(mesh_radii)

    N, H, W, _ = size(xyz)
    normalized_gpu = CUDA.zeros(GPUFloat, size(xyz))

    # Copy input to output first
    copyto!(normalized_gpu, xyz_gpu)

    # Then apply normalization in-place
    threads = (32, 32)
    blocks = (
        cld(W, threads[1]),
        cld(H, threads[2]),
        N
    )

    depths_dummy = view(xyz_gpu, :, :, :, 3)  # Use Z channel as "depth"

    @cuda threads=threads blocks=blocks depth_to_xyz_normalized_kernel!(
        normalized_gpu, depths_dummy, centers_gpu, centers_gpu, mesh_radii_gpu,
        GPUFloat(bound), N, H, W
    )

    return Array(normalized_gpu)
end

# Keep the rest of the functions the same but use GPUFloat instead of Float32/Float64
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

        center = inv(K) * [uc - 1; vc - 1; 1.0] * zc
        centers[n, :] = center
    end

    return centers
end

# Keep compute_crop_transform_mask and compute_pose_based_crop the same as before
# (include the full implementations from the previous version)

function compute_crop_transform_mask(masks; crop_ratio=1.2, target_size=(160, 160))
    N = size(masks, 1)
    T_invs = zeros(Float64, N, 3, 3)
    crop_infos = []

    for n in 1:N
        mask = masks[n, :, :]
        mask_indices = findall(mask)

        if isempty(mask_indices)
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

        pt_2d_homo = K * center_3d
        uc = pt_2d_homo[1] / pt_2d_homo[3] + 1
        vc = pt_2d_homo[2] / pt_2d_homo[3] + 1

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

# Main processing functions - optimized with fused operations
function process_mask_crop(rgbs, depths, masks, Ks;
                          crop_ratio=1.2,
                          target_size=(160, 160),
                          mesh_radii=nothing,
                          estimate_centers=true,
                          object_centers=nothing)

    N = size(rgbs, 1)

    # Preprocess depth on GPU
    depths_filtered = erode_depth(depths; radius=2)
    depths_filtered = bilateral_filter_depth(depths_filtered; radius=2)

    # Compute crop transforms (CPU operation)
    T_invs, crop_infos = compute_crop_transform_mask(masks;
                                                    crop_ratio=crop_ratio,
                                                    target_size=target_size)

    # Apply warping on GPU
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

    # Convert to XYZ on GPU (with optional normalization)
    if mesh_radii !== nothing
        if estimate_centers && object_centers === nothing
            object_centers = guess_object_center(depths, masks, Ks)
        elseif object_centers === nothing
            object_centers = zeros(Float64, N, 3)
        end

        # Fused depth to XYZ and normalization
        depths_gpu = CuArray{GPUFloat}(depths_crop)
        Ks_gpu = CuArray{GPUFloat}(Ks_crop)
        centers_gpu = CuArray{GPUFloat}(object_centers)
        radii_gpu = CuArray{GPUFloat}(mesh_radii)

        N, H, W = size(depths_crop)
        xyz_norm_gpu = CUDA.zeros(GPUFloat, N, H, W, 3)

        threads = (32, 32)
        blocks = (cld(W, threads[1]), cld(H, threads[2]), N)

        @cuda threads=threads blocks=blocks depth_to_xyz_normalized_kernel!(
            xyz_norm_gpu, depths_gpu, Ks_gpu, centers_gpu, radii_gpu,
            GPUFloat(2.0), N, H, W
        )

        xyz_norm = Array(xyz_norm_gpu)
    else
        xyz_norm = depth_to_xyz(depths_crop, Ks_crop)
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

    # Preprocess depth on GPU
    depths_filtered = erode_depth(depths; radius=2)
    depths_filtered = bilateral_filter_depth(depths_filtered; radius=2)

    # Compute crop transforms (CPU operation)
    T_invs, crop_infos = compute_pose_based_crop(poses, Ks, mesh_diameters;
                                                 crop_ratio=crop_ratio,
                                                 target_size=target_size)

    # Apply warping on GPU
    rgbs_crop = warp_perspective(rgbs, T_invs, target_size; mode=:linear)
    depths_crop = warp_perspective(depths_filtered, T_invs, target_size; mode=:nearest)

    # Update K for crops
    Ks_crop = zeros(Float64, N, 3, 3)
    for n in 1:N
        K_homo = [Ks[n, :, :] [0; 0; 1]]
        K_crop = crop_infos[n].T_forward * K_homo
        Ks_crop[n, :, :] = K_crop[1:3, 1:3]
    end

    # Convert to XYZ on GPU (with optional normalization)
    if mesh_radii !== nothing
        pose_centers = poses[:, 1:3, 4]

        # Fused depth to XYZ and normalization
        depths_gpu = CuArray{GPUFloat}(depths_crop)
        Ks_gpu = CuArray{GPUFloat}(Ks_crop)
        centers_gpu = CuArray{GPUFloat}(pose_centers)
        radii_gpu = CuArray{GPUFloat}(mesh_radii)

        N, H, W = size(depths_crop)
        xyz_norm_gpu = CUDA.zeros(GPUFloat, N, H, W, 3)

        threads = (32, 32)
        blocks = (cld(W, threads[1]), cld(H, threads[2]), N)

        @cuda threads=threads blocks=blocks depth_to_xyz_normalized_kernel!(
            xyz_norm_gpu, depths_gpu, Ks_gpu, centers_gpu, radii_gpu,
            GPUFloat(2.0), N, H, W
        )

        xyz_norm = Array(xyz_norm_gpu)
    else
        xyz_norm = depth_to_xyz(depths_crop, Ks_crop)
    end

    return (
        rgb = rgbs_crop,
        depth = depths_crop,
        xyz = xyz_norm,
        K = Ks_crop,
        crop_infos = crop_infos
    )
end
