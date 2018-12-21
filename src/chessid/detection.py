"""
Board detection from https://github.com/daylen/chess-id
"""
from collections import defaultdict
from functools import partial
from typing import NamedTuple, List, Tuple

import cv2
import numpy as np
from PIL.Image import Image, fromarray as image_fromarray, BICUBIC
import scipy.spatial as spatial
import scipy.cluster as clstr


def auto_canny(image, sigma=0.33):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


Point = NamedTuple('Point', [('x', float), ('y', float)])
Line = NamedTuple('Line', [('rho', float), ('theta', float)])


def hor_vert_lines(lines: List[Line]) -> Tuple[List[Line], List[Line]]:
    """
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    """
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v


def intersections(h: List[Line], v: List[Line]) -> List[Point]:
    """
    Given lists of lines in (rho, theta) form, returns list of (x, y) intersection points.
    """
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b).tolist()
            points.append(point)
    return points


def cluster(points: List[Point], max_dist=50):
    """
    Given a list of points, returns a list of cluster centers.
    """
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = list(clusters.values())
    clusters = [(np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])) for arr in clusters]
    return clusters


def closest_point(points: List[Point], loc) -> List[Point]:
    """
    Returns the list of points, sorted by distance from loc.
    """
    dists = np.array(list(map(partial(spatial.distance.euclidean, loc), points)))
    return points[dists.argmin()]


def perspective_transform(board_corner_points, grid_points, output_square_width=100):
    # order has to be bottom left, top left, top right, bottom right to match output_points
    board_corner_points = np.asarray(board_corner_points, dtype=np.float32)
    grid_points = np.asarray(grid_points, dtype=np.float32)

    output_points = np.array([
        [0, 0], [0, 8 * output_square_width], [8 * output_square_width, 8 * output_square_width],
        [8 * output_square_width, 0]], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(board_corner_points, output_points)
    # opencv quirk https://stackoverflow.com/questions/27585355/python-open-cv-perspectivetransform
    return cv2.perspectiveTransform(grid_points[None, :, :], transform_matrix).squeeze()


def find_corners2(h_lines: List[Line], v_lines: List[Line], points: List[Point], topn=5) -> List[Point]:
    """
    Given a list of points, returns a list containing the four corner points.
    """

    def only_inside_points(output_points, output_square_size, epsilon=10):
        inside_mask = np.logical_and(
            output_points >= -epsilon,
            output_points <= (8 * output_square_size + epsilon)
        ).all(axis=1)
        return output_points[inside_mask, :]

    def ideal_output_points(output_square_size):
        grid = np.linspace(start=0, stop=output_square_size * 8, num=8 + 1)
        return [(x, y) for x in grid for y in grid]

    def closest_points_to_ideal_distances(points, output_square_size):
        ideal_points = ideal_output_points(output_square_size)
        real_to_ideal_distances = spatial.distance.cdist(points, ideal_points, 'euclidean')
        return real_to_ideal_distances

    def mixed_distance(candidate_corner_points, all_points, output_square_size=100):
        output_points = perspective_transform(candidate_corner_points, all_points, output_square_size)
        inside_points = only_inside_points(output_points, output_square_size)
        distances = closest_points_to_ideal_distances(inside_points, output_square_size)
        return np.mean(distances.min(axis=0))

    def sort_v_lines_left_to_right(v_lines: List[Line]) -> List[Line]:
        v_lines = np.asarray(v_lines)
        x_axis_line = Line(rho=0, theta=np.pi / 2)
        with_axis_intersections = intersections(v_lines, [x_axis_line])
        left_to_right_indexes = np.argsort(np.array(with_axis_intersections)[:, 0])
        return v_lines[left_to_right_indexes].tolist()

    def sort_h_lines_bottom_to_top(h_lines: List[Line]) -> List[Line]:
        h_lines = np.asarray(h_lines)
        order_by_rho = np.argsort(h_lines[:, 0])
        return h_lines[order_by_rho, :].tolist()

    def corner_intersections(bottom_line, top_line, left_line, right_line) -> List[Point]:
        bottom_left, bottom_right, top_left, top_right = intersections(
            [bottom_line, top_line], [left_line, right_line])
        # order to match expected order for perspective transform
        return [bottom_left, top_left, top_right, bottom_right]

    print('points:', len(points))

    selected_h_lines = sort_h_lines_bottom_to_top(cluster(h_lines, max_dist=20))
    selected_v_lines = sort_v_lines_left_to_right(cluster(v_lines, max_dist=20))
    print('h_lines:', len(h_lines), '->', len(selected_h_lines))
    print('v_lines:', len(v_lines), '->', len(selected_v_lines))

    from itertools import product
    borders_candidates = list(product(
        selected_h_lines[:topn],
        selected_h_lines[-topn:],
        selected_v_lines[:topn],
        selected_v_lines[-topn:]))

    print('borders_candidates', len(borders_candidates))

    best_criterium = np.infty
    best_corners = None

    for borders in borders_candidates:
        possible_corners = corner_intersections(*borders)
        c = mixed_distance(possible_corners, points)

        if c <= best_criterium:
            best_criterium = c
            best_corners = possible_corners

    print('best_criterium', best_criterium)
    return best_corners


def four_point_transform(img, points: List[Point], square_length=1816):
    # pts1 order has to match pts2! bottom left, top left, top right, bottom right
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))


def detect_lines(edges_image: np.array) -> Tuple[List[Line], List[Line]]:
    # Hough line detection
    lines = cv2.HoughLines(edges_image, 1, np.pi / 180, threshold=300)
    lines = np.reshape(lines, (-1, 2))
    h, v = hor_vert_lines(lines)

    print(edges_image.shape)
    side_image_lines = [Line(rho=1, theta=0), Line(rho=edges_image.shape[1], theta=0)]
    return h, v + side_image_lines


def draw_lines(img: np.array, lines: List[Line]) -> np.array:
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 4000 * (-b))
        y1 = int(y0 + 4000 * (a))
        x2 = int(x0 - 4000 * (-b))
        y2 = int(y0 - 4000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def draw_points(img: np.array, points: List[Point], color=(0, 0, 255)) -> np.array:
    for point in points:
        cv2.circle(img, center=(int(point[0]), int(point[1])), radius=25, color=color, thickness=-1)
    return img


def cv_to_pil(image_array: np.array, ratio=1, size=None) -> Image:
    if len(image_array.shape) > 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = image_fromarray(image_array)
    return image.resize(size or (ratio * np.array(image.size)).astype(int), resample=BICUBIC)


class Result(NamedTuple):
    debug: Image
    board: Image
    squares: List[Image]
    all_points: List[Point] = []
    corner_points: List[Point] = []
    h_lines: List[Line] = []
    v_lines: List[Line] = []


MAX_EDGE_PIXEL_RATIO = .03


def find_board(buffer) -> Result:
    img = cv2.imdecode(buffer, 1)
    print(img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))

    edges_image = auto_canny(gray)

    edge_pixels_ratio = np.count_nonzero(edges_image) / float(edges_image.shape[0] * edges_image.shape[1])
    print(edge_pixels_ratio)
    if edge_pixels_ratio > MAX_EDGE_PIXEL_RATIO:
        print('too many edges')
        return Result(cv_to_pil(edges_image, ratio=.5), None, None)

    h_lines, v_lines = detect_lines(edges_image)
    edges_image = draw_lines(cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR), h_lines + v_lines)

    if len(h_lines) < 9 or len(v_lines) < 9:
        print('too few lines')
        return Result(cv_to_pil(edges_image, ratio=.5), None, None)

    points = intersections(h_lines, v_lines)

    points = cluster(points, max_dist=50)
    edges_image = draw_points(edges_image, points)

    corners = find_corners2(h_lines, v_lines, points)

    #corners = find_corners(points, (img.shape[1], img.shape[0]))
    edges_image = draw_points(edges_image, corners, color=(255, 0, 0))

    new_img = four_point_transform(img, corners)

    return Result(
        corner_points=corners,
        all_points=points,
        h_lines=h_lines,
        v_lines=v_lines,
        debug=cv_to_pil(edges_image, ratio=.5),
        board=cv_to_pil(new_img, size=(80 * 8, 80 * 8)),
        squares=split_board(new_img))


def split_board(img) -> List[Image]:
    """
    Given a board image, returns an array of 64 smaller images.
    """
    # TODO use https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    arr = []
    sq_len = int(img.shape[0] / 8)
    for i in range(8):
        for j in range(8):
            image = img[i * sq_len: (i + 1) * sq_len, j * sq_len: (j + 1) * sq_len]
            arr.append(cv_to_pil(image))

    return arr


def shrink_blanks(fen):
    if '_' not in fen:
        return fen
    new_fen = ''
    blanks = 0
    for char in fen:
        if char == '_':
            blanks += 1
        else:
            if blanks != 0:
                new_fen += str(blanks)
                blanks = 0
            new_fen += char
    if blanks != 0:
        new_fen += str(blanks)
    return new_fen


def get_fen(arr):
    fen = ''
    for sq in arr:
        if sq == 'empty':
            fen += '_'
        elif sq[0] == 'b':
            fen += sq[1]
        else:
            fen += str(sq[1]).upper()
    fens = [fen[i:i + 8] for i in range(0, 64, 8)]
    fens = list(map(shrink_blanks, fens))
    fen = '/'.join(fens)
    return fen