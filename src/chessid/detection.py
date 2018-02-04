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


def intersections(h: List[Line], v: List[Line]):
    """
    Given lists of lines in (rho, theta) form, returns list of (x, y) intersection points.
    """
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)


def cluster(points, max_dist=50):
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


def closest_point(points, loc):
    """
    Returns the list of points, sorted by distance from loc.
    """
    dists = np.array(list(map(partial(spatial.distance.euclidean, loc), points)))
    return points[dists.argmin()]


def find_corners(points, img_dim):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    center_point = closest_point(points, (img_dim[0] / 2, img_dim[1] / 2))
    points.remove(center_point)
    center_adjacent_point = closest_point(points, center_point)
    points.append(center_point)
    grid_dist = spatial.distance.euclidean(np.array(center_point), np.array(center_adjacent_point))

    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
    board_corners = []
    tolerance = 0.25  # bigger = more tolerance
    for img_corner in img_corners:
        while True:
            cand_board_corner = closest_point(points, img_corner)
            points.remove(cand_board_corner)
            cand_board_corner_adjacent = closest_point(points, cand_board_corner)
            corner_grid_dist = spatial.distance.euclidean(np.array(cand_board_corner),
                                                          np.array(cand_board_corner_adjacent))
            if corner_grid_dist > (1 - tolerance) * grid_dist and corner_grid_dist < (1 + tolerance) * grid_dist:
                points.append(cand_board_corner)
                board_corners.append(cand_board_corner)
                break
    return board_corners


def four_point_transform(img, points, square_length=1816):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))


def detect_lines(edges_image: np.array) -> Tuple[List[Line], List[Line]]:
    # Hough line detection
    lines = cv2.HoughLines(edges_image, 1, np.pi / 180, 200)
    lines = np.reshape(lines, (-1, 2))
    h, v = hor_vert_lines(lines)
    return h, v


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


def draw_points(img: np.array, points: np.array) -> np.array:
    for point in points:
        cv2.circle(img, center=tuple(point), radius=25, color=(0, 0, 255), thickness=-1)
    return img


def cv_to_pil(image_array: np.array, ratio=1, size=None) -> Image:
    if len(image_array.shape) > 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = image_fromarray(image_array)
    return image.resize(size or (ratio * np.array(image.size)).astype(int), resample=BICUBIC)


DetectionResult = NamedTuple('DetectionResult', [
    ('debug', Image),
    ('board', Image),
    ('squares', List[Image]),
])

MAX_EDGE_PIXEL_RATIO = .03


def find_board(buffer) -> DetectionResult:
    img = cv2.imdecode(buffer, 1)
    print(img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))

    edges_image = auto_canny(gray)

    edge_pixels_ratio = np.count_nonzero(edges_image) / float(edges_image.shape[0] * edges_image.shape[1])
    print(edge_pixels_ratio)
    if edge_pixels_ratio > MAX_EDGE_PIXEL_RATIO:
        print('too many edges')
        return DetectionResult(cv_to_pil(edges_image, ratio=.5), None, None)

    h_lines, v_lines = detect_lines(edges_image)
    edges_image = draw_lines(cv2.cvtColor(edges_image, cv2.COLOR_GRAY2BGR), h_lines + v_lines)

    if len(h_lines) < 9 or len(v_lines) < 9:
        print('too few lines')
        return DetectionResult(cv_to_pil(edges_image, ratio=.5), None, None)

    points = intersections(h_lines, v_lines)

    points = cluster(points)
    edges_image = draw_points(edges_image, points)

    # Find corners
    img_shape = np.shape(img)
    points = find_corners(points, (img_shape[1], img_shape[0]))

    # Perspective transform
    new_img = four_point_transform(img, points)

    return DetectionResult(
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