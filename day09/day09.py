import itertools
import time

import numpy as np
import taichi as ti


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


def ear_clipping(vertices: np.ndarray):
    indices = [i for i in range(len(vertices))]
    triangles = []
    
    while len(indices) > 3:
        verts_left = vertices[indices]
        
        prev_verts = np.roll(verts_left, 1, axis=0)
        next_verts = np.roll(verts_left, -1, axis=0)
        to_prev = verts_left - prev_verts
        to_next = verts_left - next_verts
        
        convex = (to_prev[:, 0] * to_next[:, 1] - to_prev[:, 1] * to_next[:, 0])
        
        i0 = np.argmax(convex > 0)
        i1 = (i0 - 1) % len(verts_left)
        i2 = (i0 + 1) % len(verts_left)
        
        v0 = verts_left[i0]
        v1 = verts_left[i1]
        v2 = verts_left[i2]
        
        for j, idx in enumerate(indices):
            if j in (i0, i1, i2):
                continue
            
            if point_in_triangle(vertices[idx], v0, v1, v2):
                break
        
        else:  # no points in triangle
            triangles.append((
                indices[i0],
                indices[i1],
                indices[i2]
            ))
            del indices[i0]
    
    triangles.append(tuple(indices))  # there are 3 indices left
    
    return np.array(triangles)


@ti.func
def point_in_polygon(
    pt: ti.types.vector(2, ti.f32),
    poly_verts: ti.template(),
    poly_tris: ti.template(),
):
    hit = False
    for t in range(poly_tris.shape[0]):
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
        results: ti.types.ndarray(dtype=ti.i32),
        pairs: ti.types.ndarray(dtype=ti.i32, ndim=2),
        poly_verts: ti.types.ndarray(dtype=ti.i32, ndim=2),
        poly_tris: ti.types.ndarray(dtype=ti.i32, ndim=2)
):
    for i in range(pairs.shape[0]):
        r0 = ti.Vector([poly_verts[pairs[i, 0], 0], poly_verts[pairs[i, 0], 1]])
        r1 = ti.Vector([poly_verts[pairs[i, 1], 0], poly_verts[pairs[i, 1], 1]])
        
        xmin = ti.min(r0[0], r1[0])
        xmax = ti.max(r0[0], r1[0])
        
        ymin = ti.min(r0[1], r1[1])
        ymax = ti.max(r0[1], r1[1])
        
        is_valid = True
        for x in range(xmin, xmax + 1):
            p0 = ti.Vector([x, ymin])
            p1 = ti.Vector([x, ymax])
            
            if not point_in_polygon(p0, poly_verts, poly_tris):
                is_valid = False
                break
            
            if not point_in_polygon(p1, poly_verts, poly_tris):
                is_valid = False
                break
        
        if is_valid:
            for y in range(ymin, ymax + 1):
                p0 = ti.Vector([xmin, y])
                p1 = ti.Vector([xmax, y])
                
                if not point_in_polygon(p0, poly_verts, poly_tris):
                    is_valid = False
                    break
                
                if not point_in_polygon(p1, poly_verts, poly_tris):
                    is_valid = False
                    break
        
        results[i] = ti.select(
            is_valid,
            (abs(r1[0] - r0[0]) + 1) * (abs(r1[1] - r0[1]) + 1),
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
    print(f"Ear clipping completed in {ear_clipping_end - ear_clipping_start:.2f} s")
    
    kernel_setup_start = time.perf_counter()
    
    coords_ti = ti.ndarray(dtype=ti.i32, shape=coords.shape)
    coords_ti.from_numpy(coords)
    
    triangles_ti = ti.ndarray(dtype=ti.int32, shape=triangles.shape)
    triangles_ti.from_numpy(triangles)
    
    pair_iter = np.array(list(itertools.combinations(range(len(coords)), 2)), dtype=np.int32)
    pair_iter = pair_iter[0:1000]
    pair_ti = ti.ndarray(shape=pair_iter.shape, dtype=ti.int32)
    pair_ti.from_numpy(pair_iter)
    
    results = ti.ndarray(dtype=ti.int32, shape=len(pair_iter))
    
    kernel_setup_end = time.perf_counter()
    print(f"Kernel setup time: {kernel_setup_end - kernel_setup_start:.2f}")
    
    kernel_start = time.perf_counter()
    rect_areas(results, pair_ti, coords_ti, triangles_ti)
    ti.sync()
    kernel_end = time.perf_counter()
    
    print(f"Kernel runtime: {kernel_end - kernel_start:.2f}")
    
    part_2 = results.to_numpy().max()
    print(f"Part 2: {part_2}")


if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    main()
