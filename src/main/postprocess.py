import numpy as np
import src.visualization.visualization as viz

def get_tour_length(tour, dists):
    """
    Args:
        tour (list): sequence of nodes in the tour. The start and end nodes
        should be added. Example: [0, 5, 3, 4, 2, 1, 0].
        dists (matrix): 2D-matrix or nested list. The symmetric distance matrix.

    Returns:
        length (int): Distance of tour.
    """
    length = 0
    for i in range(1, len(tour)):
        a = tour[i]
        b = tour[i-1]
        length += dists[a][b]
    return length


def get_ratio(segment):
    """
    Args:
        segment (np.array): a mask extracted from the pixel map. The segment
        should be binary.

    Returns:
        ratio (float): number between 0 and 1, representing the percentage of
        pixels in the segment that have a value of 1.

    Raises:
        Exception if a value in the segment is not between 0 and 1. Not a
        very strict check but stick to this for now.
    """
    total_pixels = np.prod(segment.shape)
    if total_pixels == 0:
        return 1
    if np.max(segment) > 1 or np.min(segment) < 0:
        raise Exception("Segment has to be binary.")
    predicted_pixels = np.sum(segment)
    return predicted_pixels / total_pixels


def is_valid_path(a, b):
    """
    Args:
        a (np.array): a part of the "L" in the manhattan distance.
        b (np.array): a part of the "L" in the manhattan distance.
        If the manhattan distance is not an "L", just a vertical line or
        horizontal line, then one of these can be empty.

    Returns:
        boolean: True only if both segments have a value of greater than the
        threshold.
    """
    ra = get_ratio(a)
    rb = get_ratio(b)
    threshold = 0.70
    return ra > threshold and rb > threshold


def is_not_crossing_city(a, b):
    """
    Args:
        a (np.array): a part of the "L" in the manhattan distance.
        b (np.array): a part of the "L" in the manhattan distance.
        If the manhattan distance is not an "L", just a vertical line or
        horizontal line, then one of these can be empty.

    Returns:
        boolean: True only if in the "L" or the "--" or the "|" only has two
        cities.
    """
    a = np.sum(a)
    b = np.sum(b)
    c = a + b
    # This case is for the standard L
    if c == 2:
        return True
    else:
        return False


def extract_segments(pixel_map, coords):
    """
    Args:
        pixel_map (np.array): the binary pixel map.
        coords (tuple): 4 values of y1 y2 x1 x2, such that y1 < y2 and x1 < x2.
        This allows you to immediately find the segments.

    Returns:
        tuple: the order of right, btm, left, top, the various segments
        extracted given two points.
    """
    # Verified that the combination of -1 and +2 correctly identifies the
    # region.
    l, r = 1, 2
    ans = pixel_map
    y1, y2, x1, x2 = coords
    right = ans[y1:y2+1, x2-l:x2+r]
    btm = ans[y2-l:y2+r, x1:x2+1]
    left = ans[y1:y2+1, x1-l:x1+r]
    top = ans[y1-l:y1+r, x1:x2+1]
    if x1 == x2:
        top = np.array([])
        btm = np.array([])
    if y1 == y2:
        left = np.array([])
        right = np.array([])
    return right, btm, left, top


def get_edges(a, b, path_predictions, cities):
    """
    This was very complex and I'll write up detailed docs in future.
    """
    a = tuple(a)
    b = tuple(b)
    ax, ay = a
    bx, by = b
    # Reorient the points such that A is the one at the bottom.
    if ay < by:
        c = a
        a = b
        b = c
    ax, ay = a
    bx, by = b
    y1 = min(ay, by)
    y2 = max(ay, by)
    x1 = min(ax, bx)
    x2 = max(ax, bx)
    # Note the opencv convention. If this is reversed, it's totally wrong.
    # Each L has two segments, and between any two points, there are two such
    # "L"s. s1a and s1b thus make up a single L.
    coords = (y1, y2, x1, x2)
    pr, pb, pl, pt = extract_segments(path_predictions, coords)
    cr, cb, cl, ct = extract_segments(cities, coords)
    edges = []
    # B             D
    #
    # C             A
    if ay > by and ax > bx:
        if is_valid_path(pl, pb) and is_not_crossing_city(cl, cb):
            c = (bx, ay)
            edges.append(((a, c), (c, b)))
        if is_valid_path(pr, pt) and is_not_crossing_city(cr, ct):
            d = (ax, by)
            edges.append(((a, d), (d, b)))
    # D             B
    #
    # A             C
    elif ay > by and ax < bx:
        if is_valid_path(pl, pt) and is_not_crossing_city(cl, ct):
            d = (ax, by)
            edges.append(((a, d), (d, b)))
        if is_valid_path(pr, pb) and is_not_crossing_city(cr, cb):
            c = (bx, ay)
            edges.append(((a, c), (c, b)))
    # B
    #
    # A
    elif ay > by and ax == bx:
        empty = np.array([])
        if is_valid_path(pl, empty) and is_not_crossing_city(cl, empty):
            # This tricky thing here is to create a nested tuple.
            edges.append(((a, b),))
    # A         B       or
    # B         A
    elif ay == by:
        empty = np.array([])
        if is_valid_path(pt, empty) and is_not_crossing_city(ct, empty):
            # This tricky thing here is to create a nested tuple.
            edges.append(((a, b),))
    else:
        raise Exception("Did not catch a case.")

    sorted_edges = []
    for L in edges:
        t = []
        for line in L:
            line = tuple(sorted(line))
            t.append(line)
        sorted_edges.append(tuple(t))
    return sorted_edges
    # l, r = 1, 2
    # ans[y1:y2+1, x2-l:x2+r] = 0
    # ans[y2-l:y2+r, x1:x2+1] = 0
    # ans[y1:y2+1, x1-l:x1+r] = 0
    # ans[y1-l:y1+r, x1:x2+1] = 0


def get_one_pixel_city_image(size, subset):
    img = np.zeros(size, np.uint8)
    img = viz.draw_dots(img, subset, 0, viz.RED, False)
    img[np.where((img == viz.RED).all(axis=2))] = [1, 0, 0]
    return img[:, :, 0]


def decode(nodes, ans, dists, start):
    path = [start]
    while len(nodes) != 1:
        idx = path[-1]
        try:
            fixed_node = nodes[idx]
        except KeyError:
            break
        del nodes[idx]
        print(idx)
        valid_nodes = []
        for key, node in nodes.items():
            l3 = get_edges(fixed_node, node, ans)
            dist = dists[idx][key]
            if l3:
                valid_nodes.append((key, dist))
        print(valid_nodes)
        best_dist = 1000000
        best_node = -1
        for tup in valid_nodes:
            node, dist = tup
            if dist < best_dist:
                best_dist = dist
                best_node = node
        path.append(best_node)
    path.append(start)
    return path


def get_edge_dictionary(subset, pred_img, city_img):
    dic = {}
    edges = []
    aaa = []
    for i in range(len(subset)):
        f = {}
        a = 0
        for j in range(len(subset)):
            if i == j:
                continue
            e = get_edges(subset[i], subset[j], pred_img, city_img)
            if len(e) != 0:
                a += 1
                f[j] = e
            edges += e
        dic[i] = f
        aaa.append(a)
    return dic