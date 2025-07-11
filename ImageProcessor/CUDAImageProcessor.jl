# CUDAImageProcessor.jl - Ultra-Optimized CUDA Implementation
# Maximum performance image processing for FoundationPose

using CUDA
using Statistics, LinearAlgebra

const GPUFloat = Float32
const WARP_SIZE = 32
const TILE_SIZE = 32
const BILATERAL_TILE = 16

# Buffer pool for memory reuse
mutable struct BufferPool
    buffers::Dict{Tuple{Vararg{Int}}, CuArray}
    lock::ReentrantLock
end

const GLOBAL_BUFFER_POOL = BufferPool(Dict(), ReentrantLock())

function get_buffer!(pool::BufferPool, T::Type, dims...)
    key = dims
    lock(pool.lock) do
        if !haskey(pool.buffers, key)
            pool.buffers[key] = CuArray{T}(undef, dims...)
        end
        return pool.buffers[key]
    end
end

# Erosion kernel with warp-aligned access patterns
function erode_depth_kernel_warp!(output, depth, radius, depth_diff_threshold, ratio_threshold, zfar, N, H, W)
    shared = @cuDynamicSharedMem(GPUFloat, (TILE_SIZE + 2 * radius, TILE_SIZE + 2 * radius))

    tid = threadIdx().x + (threadIdx().y - 1) * blockDim().x
    tx = threadIdx().x
    ty = threadIdx().y
    bx = blockIdx().x
    by = blockIdx().y
    n = blockIdx().z

    if n > N
        return nothing
    end

    gx = (bx - 1) * TILE_SIZE + tx
    gy = (by - 1) * TILE_SIZE + ty

    shared_size = TILE_SIZE + 2 * radius
    total_threads = blockDim().x * blockDim().y
    pixels_per_thread = cld(shared_size * shared_size, total_threads)

    for i in 0:(pixels_per_thread-1)
        pixel_id = tid + i * total_threads
        if pixel_id <= shared_size * shared_size
            local_y = cld(pixel_id, shared_size)
            local_x = pixel_id - (local_y - 1) * shared_size

            global_y = (by - 1) * TILE_SIZE + local_y - radius
            global_x = (bx - 1) * TILE_SIZE + local_x - radius

            val = (global_y >= 1 && global_y <= H && global_x >= 1 && global_x <= W) ?
                  depth[n, global_y, global_x] : 0.0f0
            shared[local_y, local_x] = val
        end
    end

    sync_threads()

    if gx <= W && gy <= H
        local_y = ty + radius
        local_x = tx + radius

        d_ori = shared[local_y, local_x]

        if d_ori < 0.001f0 || d_ori >= zfar
            output[n, gy, gx] = 0.0f0
            return nothing
        end

        bad_cnt = 0
        total = (2 * radius + 1) * (2 * radius + 1)

        @inbounds for dy in -radius:radius, dx in -radius:radius
            cur_depth = shared[local_y + dy, local_x + dx]
            bad_cnt += (cur_depth < 0.001f0 || cur_depth >= zfar ||
                       abs(cur_depth - d_ori) > depth_diff_threshold) ? 1 : 0
        end

        output[n, gy, gx] = (bad_cnt > ratio_threshold * total) ? 0.0f0 : d_ori
    end

    return nothing
end

# Bilateral filter with precomputed spatial weights
function bilateral_filter_kernel_fast!(output, depth, radius, sigma_d_lut, sigma_r_inv, zfar, N, H, W)
    shared = @cuDynamicSharedMem(GPUFloat, (BILATERAL_TILE + 2 * radius, BILATERAL_TILE + 2 * radius))

    tx = threadIdx().x
    ty = threadIdx().y
    bx = blockIdx().x
    by = blockIdx().y
    n = blockIdx().z

    gx = (bx - 1) * BILATERAL_TILE + tx
    gy = (by - 1) * BILATERAL_TILE + ty

    if n > N
        return nothing
    end

    shared_size = BILATERAL_TILE + 2 * radius
    @inbounds for dy in 0:cld(shared_size, blockDim().y)-1, dx in 0:cld(shared_size, blockDim().x)-1
        local_y = ty + dy * blockDim().y
        local_x = tx + dx * blockDim().x

        if local_y <= shared_size && local_x <= shared_size
            global_y = (by - 1) * BILATERAL_TILE + local_y - radius
            global_x = (bx - 1) * BILATERAL_TILE + local_x - radius

            val = (global_y >= 1 && global_y <= H && global_x >= 1 && global_x <= W) ?
                  depth[n, global_y, global_x] : 0.0f0
            shared[local_y, local_x] = val
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

        sum_weight = 0.0f0
        sum = 0.0f0

        @inbounds for dy in -radius:radius, dx in -radius:radius
            cur_depth = shared[local_y + dy, local_x + dx]
            if cur_depth >= 0.001f0 && cur_depth < zfar
                spatial_weight = sigma_d_lut[abs(dy) + 1, abs(dx) + 1]
                depth_diff = center_depth - cur_depth
                weight = spatial_weight * exp(-depth_diff * depth_diff * sigma_r_inv)
                sum_weight += weight
                sum += weight * cur_depth
            end
        end

        output[n, gy, gx] = sum_weight > 0.0f0 ? sum / sum_weight : 0.0f0
    end

    return nothing
end

# Warp kernel for images/depth - standard bilinear/nearest interpolation
function warp_perspective_kernel!(output, images, T_invs, mode_nearest, N, H_in, W_in, H_out, W_out, C)
    gx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = blockIdx().z

    if gx > W_out || gy > H_out || n > N
        return nothing
    end

    u_out = GPUFloat(gx - 1)
    v_out = GPUFloat(gy - 1)

    @inbounds begin
        t11 = T_invs[n, 1, 1]
        t12 = T_invs[n, 1, 2]
        t13 = T_invs[n, 1, 3]
        t21 = T_invs[n, 2, 1]
        t22 = T_invs[n, 2, 2]
        t23 = T_invs[n, 2, 3]
    end

    u_in = CUDA.fma(t11, u_out, CUDA.fma(t12, v_out, t13 + 1.0f0))
    v_in = CUDA.fma(t21, u_out, CUDA.fma(t22, v_out, t23 + 1.0f0))

    if mode_nearest
        u_nearest = round(Int32, u_in)
        v_nearest = round(Int32, v_in)

        if 1 <= u_nearest <= W_in && 1 <= v_nearest <= H_in
            @inbounds for c in 1:C
                output[n, gy, gx, c] = images[n, v_nearest, u_nearest, c]
            end
        else
            @inbounds for c in 1:C
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

            w00 = wv0 * wu0
            w01 = wv0 * wu1
            w10 = wv1 * wu0
            w11 = wv1 * wu1

            @inbounds for c in 1:C
                val = CUDA.fma(w00, images[n, v0, u0, c],
                      CUDA.fma(w01, images[n, v0, u1, c],
                      CUDA.fma(w10, images[n, v1, u0, c],
                               w11 * images[n, v1, u1, c])))
                output[n, gy, gx, c] = val
            end
        else
            @inbounds for c in 1:C
                output[n, gy, gx, c] = 0.0f0
            end
        end
    end

    return nothing
end

# Specialized kernel for warping XYZ maps with normalization
function warp_xyz_normalize_kernel!(output, xyz_maps, T_invs, centers, mesh_radii, bound, N, H_in, W_in, H_out, W_out)
    gx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = blockIdx().z

    if gx > W_out || gy > H_out || n > N
        return nothing
    end

    u_out = GPUFloat(gx - 1)
    v_out = GPUFloat(gy - 1)

    @inbounds begin
        t11 = T_invs[n, 1, 1]
        t12 = T_invs[n, 1, 2]
        t13 = T_invs[n, 1, 3]
        t21 = T_invs[n, 2, 1]
        t22 = T_invs[n, 2, 2]
        t23 = T_invs[n, 2, 3]
    end

    u_in = CUDA.fma(t11, u_out, CUDA.fma(t12, v_out, t13 + 1.0f0))
    v_in = CUDA.fma(t21, u_out, CUDA.fma(t22, v_out, t23 + 1.0f0))

    # Always use nearest neighbor for XYZ to avoid interpolation artifacts
    u_nearest = round(Int32, u_in)
    v_nearest = round(Int32, v_in)

    if 1 <= u_nearest <= W_in && 1 <= v_nearest <= H_in
        @inbounds begin
            x = xyz_maps[n, v_nearest, u_nearest, 1]
            y = xyz_maps[n, v_nearest, u_nearest, 2]
            z = xyz_maps[n, v_nearest, u_nearest, 3]
        end

        if z > 0.001f0  # Valid depth
            if centers !== nothing && mesh_radii !== nothing
                # Normalize XYZ
                @inbounds begin
                    radius_inv = 1.0f0 / mesh_radii[n]
                    norm_x = (x - centers[n, 1]) * radius_inv
                    norm_y = (y - centers[n, 2]) * radius_inv
                    norm_z = (z - centers[n, 3]) * radius_inv
                end

                if abs(norm_x) < bound && abs(norm_y) < bound && abs(norm_z) < bound
                    @inbounds begin
                        output[n, gy, gx, 1] = norm_x
                        output[n, gy, gx, 2] = norm_y
                        output[n, gy, gx, 3] = norm_z
                    end
                else
                    @inbounds begin
                        output[n, gy, gx, 1] = 0.0f0
                        output[n, gy, gx, 2] = 0.0f0
                        output[n, gy, gx, 3] = 0.0f0
                    end
                end
            else
                # No normalization
                @inbounds begin
                    output[n, gy, gx, 1] = x
                    output[n, gy, gx, 2] = y
                    output[n, gy, gx, 3] = z
                end
            end
        else
            @inbounds begin
                output[n, gy, gx, 1] = 0.0f0
                output[n, gy, gx, 2] = 0.0f0
                output[n, gy, gx, 3] = 0.0f0
            end
        end
    else
        @inbounds begin
            output[n, gy, gx, 1] = 0.0f0
            output[n, gy, gx, 2] = 0.0f0
            output[n, gy, gx, 3] = 0.0f0
        end
    end

    return nothing
end

# Depth to XYZ kernel using precomputed K inverse
function depth_to_xyz_kernel!(xyz, depths, K_inv_gpu, N, H, W)
    gx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    gy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n = blockIdx().z

    if gx > W || gy > H || n > N
        return nothing
    end

    @inbounds z = depths[n, gy, gx]

    if z > 0.001f0
        u = GPUFloat(gx - 1)
        v = GPUFloat(gy - 1)

        @inbounds begin
            x = CUDA.fma(K_inv_gpu[n, 1, 1], u, CUDA.fma(K_inv_gpu[n, 1, 2], v, K_inv_gpu[n, 1, 3])) * z
            y = CUDA.fma(K_inv_gpu[n, 2, 1], u, CUDA.fma(K_inv_gpu[n, 2, 2], v, K_inv_gpu[n, 2, 3])) * z

            xyz[n, gy, gx, 1] = x
            xyz[n, gy, gx, 2] = y
            xyz[n, gy, gx, 3] = z
        end
    else
        @inbounds begin
            xyz[n, gy, gx, 1] = 0.0f0
            xyz[n, gy, gx, 2] = 0.0f0
            xyz[n, gy, gx, 3] = 0.0f0
        end
    end

    return nothing
end

# Precompute spatial weights for bilateral filter
function precompute_spatial_weights(radius, sigma_d)
    weights = zeros(Float32, radius + 1, radius + 1)
    sigma_d_inv = 0.5f0 / (sigma_d * sigma_d)
    for dy in 0:radius, dx in 0:radius
        weights[dy + 1, dx + 1] = exp(-Float32(dx^2 + dy^2) * sigma_d_inv)
    end
    return CuArray(weights)
end

# Public interface functions
function erode_depth(depth::Array{Float32}; radius=2, depth_diff_threshold=0.001f0,
                     ratio_threshold=0.8f0, zfar=100.0f0)
    depth_gpu = CuArray(depth)
    N, H, W = size(depth_gpu)

    output = get_buffer!(GLOBAL_BUFFER_POOL, Float32, N, H, W)

    threads = (TILE_SIZE, TILE_SIZE)
    blocks = (cld(W, TILE_SIZE), cld(H, TILE_SIZE), N)

    shared_size = sizeof(Float32) * (TILE_SIZE + 2 * radius)^2

    @cuda threads=threads blocks=blocks shmem=shared_size erode_depth_kernel_warp!(
        output, depth_gpu, radius, depth_diff_threshold,
        ratio_threshold, zfar, N, H, W
    )

    return Array(output)
end

function bilateral_filter_depth(depth::Array{Float32}; radius=2, sigma_d=2.0f0,
                               sigma_r=100000.0f0, zfar=100.0f0)
    depth_gpu = CuArray(depth)
    N, H, W = size(depth_gpu)

    output = get_buffer!(GLOBAL_BUFFER_POOL, Float32, N, H, W)

    sigma_d_lut = precompute_spatial_weights(radius, sigma_d)
    sigma_r_inv = 0.5f0 / (sigma_r * sigma_r)

    threads = (BILATERAL_TILE, BILATERAL_TILE)
    blocks = (cld(W, BILATERAL_TILE), cld(H, BILATERAL_TILE), N)

    shared_size = sizeof(Float32) * (BILATERAL_TILE + 2 * radius)^2

    @cuda threads=threads blocks=blocks shmem=shared_size bilateral_filter_kernel_fast!(
        output, depth_gpu, radius, sigma_d_lut, sigma_r_inv, zfar, N, H, W
    )

    return Array(output)
end

function warp_perspective(images::Array{Float32}, T_invs, target_size; mode=:linear)
    N = size(images, 1)
    H_out, W_out = target_size

    images_gpu = CuArray(images)
    T_invs_gpu = CuArray{Float32}(T_invs)

    if ndims(images) == 4
        H_in, W_in, C = size(images)[2:4]
        output_gpu = get_buffer!(GLOBAL_BUFFER_POOL, Float32, N, H_out, W_out, C)
    else
        H_in, W_in = size(images)[2:3]
        C = 1
        images_gpu = reshape(images_gpu, N, H_in, W_in, 1)
        output_gpu = get_buffer!(GLOBAL_BUFFER_POOL, Float32, N, H_out, W_out, 1)
    end

    threads = (WARP_SIZE, WARP_SIZE)
    blocks = (cld(W_out, WARP_SIZE), cld(H_out, WARP_SIZE), N)

    mode_nearest = mode == :nearest

    @cuda threads=threads blocks=blocks warp_perspective_kernel!(
        output_gpu, images_gpu, T_invs_gpu, mode_nearest,
        N, H_in, W_in, H_out, W_out, C
    )

    output = Array(output_gpu)

    if ndims(images) == 3
        output = dropdims(output, dims=4)
    end

    return output
end

function warp_xyz_maps(xyz_maps::Array{Float32,4}, T_invs, target_size;
                      centers=nothing, mesh_radii=nothing, bound=2.0f0)
    N, H_in, W_in, _ = size(xyz_maps)
    H_out, W_out = target_size

    xyz_gpu = CuArray(xyz_maps)
    T_invs_gpu = CuArray{Float32}(T_invs)
    output_gpu = get_buffer!(GLOBAL_BUFFER_POOL, Float32, N, H_out, W_out, 3)

    centers_gpu = centers !== nothing ? CuArray{Float32}(centers) : nothing
    radii_gpu = mesh_radii !== nothing ? CuArray{Float32}(mesh_radii) : nothing

    threads = (WARP_SIZE, WARP_SIZE)
    blocks = (cld(W_out, WARP_SIZE), cld(H_out, WARP_SIZE), N)

    @cuda threads=threads blocks=blocks warp_xyz_normalize_kernel!(
        output_gpu, xyz_gpu, T_invs_gpu, centers_gpu, radii_gpu,
        bound, N, H_in, W_in, H_out, W_out
    )

    return Array(output_gpu)
end

function depth_to_xyz(depths::Array{Float32}, Ks)
    depths_gpu = CuArray(depths)
    N, H, W = size(depths)
    xyz_gpu = get_buffer!(GLOBAL_BUFFER_POOL, Float32, N, H, W, 3)

    K_inv = zeros(Float32, N, 3, 3)
    for n in 1:N
        K_inv[n, :, :] = inv(Ks[n, :, :])
    end
    K_inv_gpu = CuArray(K_inv)

    threads = (WARP_SIZE, WARP_SIZE)
    blocks = (cld(W, WARP_SIZE), cld(H, WARP_SIZE), N)

    @cuda threads=threads blocks=blocks depth_to_xyz_kernel!(
        xyz_gpu, depths_gpu, K_inv_gpu, N, H, W
    )

    return Array(xyz_gpu)
end

# CPU utility functions for crop computation
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

# Main processing functions matching Foundation Pose approach
function process_mask_crop(rgbs::Array{Float32}, depths::Array{Float32},
                          masks::Union{Array{Float32}, BitArray}, Ks;
                          crop_ratio=1.2f0, target_size=(160, 160),
                          mesh_radii=nothing, estimate_centers=true,
                          object_centers=nothing)

    N = size(rgbs, 1)

    stream1 = CuStream()
    stream2 = CuStream()

    # Start depth preprocessing on GPU
    CUDA.stream!(stream1) do
        global depths_filtered = erode_depth(depths; radius=2)
    end

    # Compute crop transforms on CPU using original mask format
    T_invs, crop_infos = compute_crop_transform_mask(masks;
                                                    crop_ratio=Float64(crop_ratio),
                                                    target_size=target_size)

    # Update K for crops
    Ks_crop = zeros(Float32, N, 3, 3)
    for n in 1:N
        K_homo = [Ks[n, :, :] [0; 0; 1]]
        K_crop = crop_infos[n].T_forward * K_homo
        Ks_crop[n, :, :] = K_crop[1:3, 1:3]
    end

    # Estimate centers if needed
    if mesh_radii !== nothing && estimate_centers && object_centers === nothing
        object_centers = Float32.(guess_object_center(depths, masks, Ks))
    elseif object_centers === nothing
        object_centers = zeros(Float32, N, 3)
    end

    # Wait for erosion and continue with bilateral filter
    synchronize(stream1)
    CUDA.stream!(stream1) do
        global depths_filtered = bilateral_filter_depth(depths_filtered; radius=2)
    end

    # Compute XYZ maps at original resolution (matching Foundation Pose)
    CUDA.stream!(stream2) do
        global xyz_maps_ori = depth_to_xyz(depths_filtered, Ks)
    end

    # Wait for both operations
    synchronize(stream1)
    synchronize(stream2)

    # Warp RGB and XYZ maps to crop size
    rgbs_crop = warp_perspective(rgbs, T_invs, target_size; mode=:linear)
    xyz_crop = warp_xyz_maps(xyz_maps_ori, T_invs, target_size;
                            centers=object_centers,
                            mesh_radii=mesh_radii,
                            bound=2.0f0)

    # Also get the cropped depth for completeness
    depths_crop = warp_perspective(depths_filtered, T_invs, target_size; mode=:nearest)

    return (
        rgb = rgbs_crop,
        depth = depths_crop,
        xyz = xyz_crop,
        mask = nothing,
        K = Ks_crop,
        crop_infos = crop_infos,
        object_centers = object_centers
    )
end

function process_pose_crop(rgbs::Array{Float32}, depths::Array{Float32},
                          poses, Ks, mesh_diameters;
                          crop_ratio=1.5f0, target_size=(160, 160),
                          mesh_radii=nothing)

    N = size(rgbs, 1)

    stream1 = CuStream()
    stream2 = CuStream()

    # Start depth preprocessing
    CUDA.stream!(stream1) do
        global depths_filtered = erode_depth(depths; radius=2)
    end

    # Compute crop transforms on CPU
    T_invs, crop_infos = compute_pose_based_crop(poses, Ks, mesh_diameters;
                                                 crop_ratio=Float64(crop_ratio),
                                                 target_size=target_size)

    # Update K for crops
    Ks_crop = zeros(Float32, N, 3, 3)
    for n in 1:N
        K_homo = [Ks[n, :, :] [0; 0; 1]]
        K_crop = crop_infos[n].T_forward * K_homo
        Ks_crop[n, :, :] = K_crop[1:3, 1:3]
    end

    # Extract pose centers
    pose_centers = Float32.(poses[:, 1:3, 4])

    # Wait for erosion and continue with bilateral
    synchronize(stream1)
    CUDA.stream!(stream1) do
        global depths_filtered = bilateral_filter_depth(depths_filtered; radius=2)
    end

    # Compute XYZ maps at original resolution
    CUDA.stream!(stream2) do
        global xyz_maps_ori = depth_to_xyz(depths_filtered, Ks)
    end

    # Wait for both operations
    synchronize(stream1)
    synchronize(stream2)

    # Warp RGB and XYZ maps
    rgbs_crop = warp_perspective(rgbs, T_invs, target_size; mode=:linear)

    if mesh_radii !== nothing
        xyz_crop = warp_xyz_maps(xyz_maps_ori, T_invs, target_size;
                                centers=pose_centers,
                                mesh_radii=Float32.(mesh_radii),
                                bound=2.0f0)
    else
        xyz_crop = warp_xyz_maps(xyz_maps_ori, T_invs, target_size)
    end

    # Also get the cropped depth
    depths_crop = warp_perspective(depths_filtered, T_invs, target_size; mode=:nearest)

    return (
        rgb = rgbs_crop,
        depth = depths_crop,
        xyz = xyz_crop,
        K = Ks_crop,
        crop_infos = crop_infos
    )
end
