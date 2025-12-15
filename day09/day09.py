import itertools
import time

import numpy as np
import taichi as ti

from tqdm import tqdm


def sign(v0, v1, v2):
    return (v0[0] - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (v0[1] - v2[1])


@ti.func
def sign_parallel(v0, v1, v2):
    return (v0[0] - v2[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (v0[1] - v2[1])


def point_in_triangle(pt, v0, v1, v2):
    d0 = sign(pt, v0, v1)
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v0)
    
    has_neg = d0 < 0 or d1 < 0 or d2 < 0
    has_pos = d0 > 0 or d1 > 0 or d2 > 0
    
    return not (has_neg and has_pos)


@ti.func
def point_in_triangle_parallel(pt, v0, v1, v2):
    d0 = sign_parallel(pt, v0, v1)
    d1 = sign_parallel(pt, v1, v2)
    d2 = sign_parallel(pt, v2, v0)
    
    has_neg = d0 < 0 or d1 < 0 or d2 < 0
    has_pos = d0 > 0 or d1 > 0 or d2 > 0
    
    return not (has_neg and has_pos)


def signed_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def ear_clipping(vertices: np.ndarray):
    # Ensure counter-clockwise winding
    if signed_area(vertices) < 0:
        vertices = vertices[::-1]
    
    indices = list(range(len(vertices)))
    triangles = []
    
    while len(indices) > 3:
        ear_found = False
        
        for i in range(len(indices)):
            # Get three consecutive vertices
            i_prev = (i - 1) % len(indices)
            i_next = (i + 1) % len(indices)
            
            idx_prev = indices[i_prev]
            idx_curr = indices[i]
            idx_next = indices[i_next]
            
            v_prev = vertices[idx_prev]
            v_curr = vertices[idx_curr]
            v_next = vertices[idx_next]
            
            # Check if this vertex is convex (forms a left turn)
            edge1 = v_curr - v_prev
            edge2 = v_next - v_curr
            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            
            if cross <= 0:  # Reflex or collinear vertex, skip
                continue
            
            # Check if any other vertex is inside this triangle
            is_ear = True
            for j, idx_test in enumerate(indices):
                if j in (i_prev, i, i_next):
                    continue
                
                if point_in_triangle(vertices[idx_test], v_prev, v_curr, v_next):
                    is_ear = False
                    break
            
            if is_ear:
                # Found a valid ear, clip it
                triangles.append((idx_prev, idx_curr, idx_next))
                indices.pop(i)
                ear_found = True
                break
        
        if not ear_found:
            # Fallback: this shouldn't happen with valid polygons
            # but if it does, just take the first convex vertex
            print("Warning: No ear found, forcing triangle removal")
            if len(indices) >= 3:
                triangles.append((indices[0], indices[1], indices[2]))
                indices.pop(1)
            else:
                break
    
    if len(indices) == 3:
        triangles.append(tuple(indices))
    
    return np.array(triangles)

@ti.func
def point_in_polygon(
    pt: ti.types.vector(2, ti.f32),
    poly_verts: ti.template(),
    poly_tris: ti.template(),
):
    hit = False
    for t in range(poly_tris.shape[0]):
        if not hit:  # Early exit optimization
            i0 = poly_tris[t, 0]
            i1 = poly_tris[t, 1]
            i2 = poly_tris[t, 2]

            v0 = ti.Vector([poly_verts[i0, 0], poly_verts[i0, 1]])
            v1 = ti.Vector([poly_verts[i1, 0], poly_verts[i1, 1]])
            v2 = ti.Vector([poly_verts[i2, 0], poly_verts[i2, 1]])

            if point_in_triangle_parallel(pt, v0, v1, v2):
                hit = True

    return hit


def edges_of_rect(r0, r1):
    xmin, xmax = sorted((r0[0], r1[0]))
    ymin, ymax = sorted((r0[1], r1[1]))
    
    edges = set()
    for x in range(xmin, xmax + 1):
        edges.add((x, ymin))
        edges.add((x, ymax))
    
    for y in range(ymin, ymax + 1):
        edges.add((xmin, y))
        edges.add((xmax, y))
    
    return np.array(list(edges))


@ti.kernel
def valid_rect(
    edges: ti.types.ndarray(dtype=ti.f32, ndim=2),
    poly_verts: ti.types.ndarray(dtype=ti.f32, ndim=2),
    poly_tris: ti.types.ndarray(dtype=ti.i32, ndim=2),
) -> ti.i32:

    result = 1  # true

    for i in range(edges.shape[0]):
        edge = ti.Vector([edges[i, 0], edges[i, 1]])
        if not point_in_polygon(edge, poly_verts, poly_tris):
            result = 0

    return result


def rect_area(r0, r1):
    return (abs(r1[0] - r0[0]) + 1) * (abs(r1[1] - r0[1]) + 1)


@ti.kernel
def rect_areas(
        results: ti.types.ndarray(dtype=ti.i64),
        pairs: ti.types.ndarray(dtype=ti.i32, ndim=2),
        poly_verts: ti.types.ndarray(dtype=ti.i32, ndim=2),
        poly_tris: ti.types.ndarray(dtype=ti.i32, ndim=2)
):
    for i in range(pairs.shape[0]):
        r0_x = poly_verts[pairs[i, 0], 0]
        r0_y = poly_verts[pairs[i, 0], 1]
        r1_x = poly_verts[pairs[i, 1], 0]
        r1_y = poly_verts[pairs[i, 1], 1]
        
        xmin = ti.min(r0_x, r1_x)
        xmax = ti.max(r0_x, r1_x)
        
        ymin = ti.min(r0_y, r1_y)
        ymax = ti.max(r0_y, r1_y)
        
        is_valid = True
        
        # Check top and bottom edges (avoiding duplicate corner checks)
        for x in range(xmin, xmax + 1):
            if is_valid:
                p0 = ti.Vector([ti.f32(x), ti.f32(ymin)])
                if not point_in_polygon(p0, poly_verts, poly_tris):
                    is_valid = False
            
            if is_valid and ymin != ymax:  # Avoid checking same point twice
                p1 = ti.Vector([ti.f32(x), ti.f32(ymax)])
                if not point_in_polygon(p1, poly_verts, poly_tris):
                    is_valid = False
        
        # Check left and right edges (skip corners already checked)
        if is_valid:
            for y in range(ymin, ymax + 1):  # Skip corners
                if is_valid:
                    p0 = ti.Vector([ti.f32(xmin), ti.f32(y)])
                    if not point_in_polygon(p0, poly_verts, poly_tris):
                        is_valid = False
                
                if is_valid and xmin != xmax:  # Avoid checking same point twice
                    p1 = ti.Vector([ti.f32(xmax), ti.f32(y)])
                    if not point_in_polygon(p1, poly_verts, poly_tris):
                        is_valid = False
        
        results[i] = ti.select(
            is_valid,
            ti.int64((ti.abs(ti.cast(r1_x, ti.int64) - ti.cast(r0_x, ti.int64)) + 1) * (ti.abs(ti.cast(r1_y, ti.int64) - ti.cast(r0_y, ti.int64)) + 1)),
            0
        )
        
    
def main():
    with open("input.txt") as f:
        coords_txt = f.read().splitlines()
    
    coords = []
    for coord_txt in coords_txt:
        coords.append(list(map(int, coord_txt.split(","))))
    
    coords = np.array(coords)
    
    print("Starting ear clipping...")
    ear_clipping_start = time.perf_counter()
    triangles = ear_clipping(coords)
    ear_clipping_end = time.perf_counter()
    print(f"Ear clipping completed in {ear_clipping_end - ear_clipping_start:.2f}s")
    
    kernel_setup_start = time.perf_counter()
    
    coords_ti = ti.ndarray(dtype=ti.i32, shape=coords.shape)
    coords_ti.from_numpy(coords)
    
    triangles_ti = ti.ndarray(dtype=ti.int32, shape=triangles.shape)
    triangles_ti.from_numpy(triangles)
    
    pair_iter = np.array(list(itertools.combinations(range(len(coords)), 2)), dtype=np.int32)
    
    kernel_setup_end = time.perf_counter()
    print(f"Kernel setup time: {kernel_setup_end - kernel_setup_start:.2f}s")
    
    # Process in chunks
    total_pairs = len(pair_iter)
    chunk_size = max(1, total_pairs // 1)
    num_chunks = (total_pairs + chunk_size - 1) // chunk_size  # Ceiling division
    
    print(f"Processing {total_pairs} pairs in {num_chunks} chunks of ~{chunk_size} pairs each")
    
    all_results = np.zeros(total_pairs, dtype=np.int64)
    
    kernel_start = time.perf_counter()
    
    for chunk_idx in tqdm(range(num_chunks)):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_pairs)
        
        chunk_pairs = pair_iter[start_idx:end_idx]
        
        pair_ti = ti.ndarray(shape=chunk_pairs.shape, dtype=ti.int32)
        pair_ti.from_numpy(chunk_pairs)
        
        results = ti.ndarray(dtype=ti.int64, shape=len(chunk_pairs))
        
        rect_areas(results, pair_ti, coords_ti, triangles_ti)
        ti.sync()
        
        all_results[start_idx:end_idx] = results.to_numpy()

    kernel_end = time.perf_counter()
    
    print(f"Kernel runtime: {kernel_end - kernel_start:.2f}s")
    
    part_2 = all_results.max()
    print(f"Part 2: {part_2}")


if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    main()
