import numpy as np
import cv2

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def draw_lines(img, edges, delta):
    i = 0
    for L in edges:
        for s in L:
            a, b = s
            img = cv2.line(img, a, b, WHITE, delta//2)
            # print(i, s)
            # if i == 49:
            #     img = cv2.line(img, a, b, GREEN, delta//2)
            # if i == 50:
            #     img = cv2.line(img, a, b, RED, delta//2)
            # cv2.imwrite("%d.png" %i, img)
            i += 1
    return img


def draw_city(img, coords, size, delta):
    for coord in coords:
        x, y = coord
        xa = (x, 0)
        xb = (x, size[0])
        ya = (0, y)
        yb = (size[1], y)
        img = cv2.line(img, xa, xb, WHITE, delta//2)
        img = cv2.line(img, ya, yb, WHITE, delta//2)
    return img


def draw_manhattan_solution(img, delivery_locations, tour, delta):
    coords = delivery_locations
    for i in range(1, len(tour)):
        ax, ay = coords[tour[i]]
        bx, by = coords[tour[i-1]]
        img = cv2.line(img, (ax, ay), (ax, by), GREEN, delta//2)
        img = cv2.line(img, (ax, by), (bx, by), GREEN, delta//2)
    return img


def draw_dots(img, coords, delta, color, add_text):
    for i, coord in enumerate(coords):
        r = delta // 2
        x, y = coord
        # OpenCV has BGR instead of RGB
        img[y-r: y+r+1, x-r:x+r+1] = color
        if add_text:
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1)
    return img


def draw_input_data(img, coords, dl, size, delta, color, add_text):
    img = draw_city(img, coords, size, delta)
    img = draw_dots(img, dl, delta, color, add_text)
    return img


def draw_output_data(img, delivery_locations, tour, delta, color, add_text):
    img = draw_manhattan_solution(img, delivery_locations, tour, delta)
    img = draw_dots(img, delivery_locations, delta, color, add_text)
    return img
