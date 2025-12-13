import itertools
from tqdm import tqdm

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

    
def main():
    with open("input.txt") as f:
        coords_txt = f.read().splitlines()
    
    coords = []
    for coord_txt in coords_txt:
        coords.append(list(map(int, coord_txt.split(","))))
    
    coords = np.array(coords)
    triangles = ear_clipping(coords)
    
    coords_ti = ti.ndarray(dtype=ti.f32, shape=coords.shape)
    coords_ti.from_numpy(coords)
    
    triangles_ti = ti.ndarray(dtype=ti.int32, shape=triangles.shape)
    triangles_ti.from_numpy(triangles)
    
    
    num_pairs = len(coords) * (len(coords) - 1) // 2
    
    max_area = 0
    pair_iter = itertools.combinations(range(len(coords)), 2)
    
    for i0, i1 in tqdm(pair_iter, total=num_pairs):
        r0 = coords[i0]
        r1 = coords[i1]
        
        edges = edges_of_rect(r0, r1)
        edges_ti = ti.ndarray(dtype=ti.f32, shape=edges.shape)
        edges_ti.from_numpy(edges)
        
        if not valid_rect(edges_ti, coords_ti, triangles_ti):
            continue
        
        area = rect_area(r0, r1)
        if area > max_area:
            max_area = area
    
    print(f"Part 2: {max_area}")


if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    main()
