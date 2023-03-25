import os
from IPython.display import Image
import png

import cv2

import numpy
from math import *


# if input image is in range 0..1, please first multiply img by 255
# assume image is ndarray of shape [height, width, channels] where channels can be 1, 3 or 4
def imshow(img):
    import cv2
    import IPython
    _, ret = cv2.imencode('.png', img)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)

def swap(a,b):
    a, b = b, a

def swap_axes(a):
    swap(a[0], a[1])

def pixel_coverage_under_line(y0, y1):
    if y0 > 1 and y1 > 1:
        return 1

    if y0 < 0 and y1 < 0:
        return 0

    if y0 < y1:
        y0, y1 = y1, y0

    if y1 >= 0:
        return (y0 + y1) * 0.5

    x = y0 / (y0 - y1)

    return (x * y0) * 0.5

# output_function has to accept int tuple (x, y) and coverage
def rasterize_line_antialiased(start, end, thickness, output_function):
    # - flip segment to always step in x direction
    # - find coarse rasterization range, compute coverage at each pixel
    # - output a pixel
    delta = end - start
    if (abs(delta[0]) + abs(delta[1])) < 0.05:
        return

    flipped = abs(delta[0]) < abs(delta[1])

    if flipped:
        swap_axes(delta)
        swap_axes(start)
        swap_axes(end)

    if start[0] > end[0]:
        swap(start, end)
        delta = -delta

    start_xi = trunc(start[0])
    end_xi = ceil(end[0]) + 1

    cossin_alpha = delta / sqrt(delta[0]*delta[0] + delta[1]*delta[1])
    thickness_y = (thickness / cossin_alpha[0]) / 2

    for x in range(start_xi, end_xi):
        center_x = x + 0.5
        center_y = start[1] + (center_x - start[0]) * delta[1] / delta[0]
        bottom_y = trunc(center_y - thickness * 0.5 - 0.0001)
        top_y = ceil(center_y + thickness * 0.5 + 0.0001) + 1
        for y in range(bottom_y, top_y):
            x0 = center_x - 0.5
            x1 = center_x + 0.5
            y0 = start[1] + (x0 - start[0]) * delta[1] / delta[0] - y
            y1 = start[1] + (x1 - start[0]) * delta[1] / delta[0] - y
            coverage = pixel_coverage_under_line(y0 + thickness_y, y1 + thickness_y)
            coverage -= pixel_coverage_under_line(y0 - thickness_y, y1 - thickness_y)
            output_coords = (center_x, y + 0.5)
            if flipped:
                swap_axes(output_coords)
            output_function(output_coords, coverage)


def distsqr(p):
    return p[0]*p[0] + p[1]*p[1]

def hue2rgb(h):
    c = 1
    x = c * ( 1.0 - abs((h / 60.0) % 2.0 - 1.0) )
    m = 1 - c
    if h < 60:
        return numpy.array([c, x, 0]) * 255.0
    if h < 120:
        return numpy.array([x, c, 0]) * 255.0
    if h < 180:
        return numpy.array([0, c, x]) * 255.0
    if h < 240:
        return numpy.array([0, x, c]) * 255.0
    if h < 300:
        return numpy.array([x, 0, c]) * 255.0

    return numpy.array([c, 0, x]) * 255.0


def CreateImage(resolution, limit):
    src = numpy.zeros((resolution, resolution, 3))

    radius = resolution * 0.4

    center = numpy.array([resolution, resolution], dtype='f4')

    center *= 0.5

    brightness = 0.2

    def DrawRefractedLine(h, thickness, ior):
        line_start = center + (-resolution / 2, h * radius)

        sin_gammai = h
        cos_gammai = sqrt(1 - sin_gammai * sin_gammai)

        line_end = center + numpy.array([-cos_gammai, sin_gammai]) * radius

        line_brightness = sqrt(1 - h * h) * brightness

        def output_additive(p, coverage):
            pi = (trunc(p[1]), trunc(p[0]))
            if (pi[0] >= 0) and (pi[1] >= 0) and (pi[0] < resolution) and (pi[1] < resolution):
                src[pi] += numpy.array([0.0, 0.0, 1.0]) * coverage * line_brightness

        rasterize_line_antialiased(line_start + 0.5, line_end + 0.5, thickness, output_additive)

        line_start = line_end

        sin_gammat = sin_gammai / ior

        cos_gammat = sqrt(1 - sin_gammat * sin_gammat)
        tg_gammat = sin_gammat / cos_gammat

        omegat = numpy.array([cos_gammai, -sin_gammai])
        omegat += numpy.array([sin_gammai, cos_gammai]) * tg_gammat

        line_end = line_start + omegat * resolution

        r0 = (1 - ior) / (1 + ior)
        r0 *= r0

        fresnel_shlick = r0 + (1 - r0) * pow(1.0 - cos_gammai, 5.0)

        def output_additive(p, coverage):
            pi = (trunc(p[1]), trunc(p[0]))
            if (pi[0] >= 0) and (pi[1] >= 0) and (pi[0] < resolution) and (pi[1] < resolution):
                p2c = p - center
                distance_to_axis = abs(p2c[1])
                amplification_factor = max( 1, radius * abs(h) / ( distance_to_axis + 0.5 ) ) # we are calculating 3d ball caustics, have to account for that
                src[pi] += numpy.array([0.0, 0.0, 1.0]) * coverage * line_brightness * (1 - fresnel_shlick) * amplification_factor

        rasterize_line_antialiased(line_start + 0.5, line_end + 0.5, thickness, output_additive)

    for i in range(-int(radius * pi * 0.5), int(radius * pi * 0.5)):
        h = sin(i / radius)
        if (h > limit): break
        DrawRefractedLine(h, 1, 1.55)

    # linear to "perceptional"
    for x in range(0, resolution):
        for y in range(0, resolution):
            value = src[(x,y)][2]

            src[(x,y)] = hue2rgb(value / 60.0 * 360.0) * sqrt(value)


    cv2.circle(src, center.astype(int), int(radius), (255, 0, 0), 2)

    return src
def CreateLUT(resolution):
    height = int( resolution / 2 )

    src = numpy.zeros((height, resolution, 3))


    radius = resolution / 2

    center = numpy.array([resolution, 0], dtype='f4')

    center *= 0.5

    brightness = 0.2
    def DrawRefractedLine(h, thickness, ior):
        line_start = center + (-resolution / 2, h * radius)

        sin_gammai = h
        cos_gammai = sqrt(1 - sin_gammai * sin_gammai)

        line_end = center + numpy.array([-cos_gammai, sin_gammai]) * radius

        line_brightness = sqrt(1 - h * h) * brightness

        def output_additive(p, coverage):
            pi = (trunc(p[1]), trunc(p[0]))
            if (pi[0] >= 0) and (pi[1] >= 0) and (pi[0] < height) and (pi[1] < resolution):
                src[pi] += numpy.array([0.0, 0.0, 1.0]) * coverage * line_brightness

        rasterize_line_antialiased(line_start + 0.5, line_end + 0.5, thickness, output_additive)

        line_start = line_end

        sin_gammat = sin_gammai / ior

        cos_gammat = sqrt(1 - sin_gammat * sin_gammat)
        tg_gammat = sin_gammat / cos_gammat

        omegat = numpy.array([cos_gammai, -sin_gammai])
        omegat += numpy.array([sin_gammai, cos_gammai]) * tg_gammat

        line_end = line_start + omegat * resolution

        r0 = (1 - ior) / (1 + ior)
        r0 *= r0

        fresnel_shlick = r0 + (1 - r0) * pow(1.0 - cos_gammai, 5.0)

        def output_additive(p, coverage):
            pi = (trunc(p[1]), trunc(p[0]))
            if (pi[0] >= 0) and (pi[1] >= 0) and (pi[0] < height) and (pi[1] < resolution):
                p2c = p - center
                distance_to_axis = abs(p2c[1])
                amplification_factor = max(1, radius * abs(h) / (
                            distance_to_axis + 0.5))  # we are calculating 3d ball caustics, have to account for that
                src[pi] += numpy.array([0.0, 0.0, 1.0]) * coverage * line_brightness * (
                            1 - fresnel_shlick) * amplification_factor

        rasterize_line_antialiased(line_start + 0.5, line_end + 0.5, thickness, output_additive)

    for i in range(0, int(radius * pi * 0.5)):
        h = sin(i / radius)
        DrawRefractedLine(h, 1, 1.55)

    # linear to "perceptional"
    for x in range(0, height):
        for y in range(0, resolution):
            value = src[(x, y)][2]

            src[(x, y)] = hue2rgb(value / 60.0 * 360.0) * sqrt(value)

    return src

def lerp(a, b, k):
    return b * k + a * (1 - k)

def saturate(a):
    return min( 1, max( 0, a ) )

def invlerp(a, b, x):
    return (x - a) / (b - a)

def CreateSlice(lut, resolution, h, angle):
    sliceImg = numpy.zeros((resolution, resolution, 3))

    lutwidth = 1023
    lutheight = 511

    max_dist_from_center = sqrt(1 - h * h)

    for i in range(0, resolution):
        for j in range(0, resolution):
            xcoord = i / (resolution - 1) * 2 - 1
            ycoord = j / (resolution - 1) * 2 - 1

            distance_from_center = sqrt(xcoord * xcoord + ycoord * ycoord)

            lerp_coef = distance_from_center / max_dist_from_center

            if distance_from_center < 0.25:
                continue

            interpolated_h = h * lerp_coef + h * 0.7 * ( 1 - lerp_coef )
            vec = numpy.array([xcoord, ycoord, interpolated_h])
            veclen = numpy.linalg.norm(vec)
            if veclen > 1:
                sliceImg[(j, i)] = numpy.array([0.7, 0.7, 0.7]) * 255
                continue

            vec /= veclen + 0.0001

            lightdir = numpy.array([sin(angle), 0, cos(angle)])

            wrapped_cos = sqrt( saturate( invlerp( -0.5, 1, cos(angle) ) ) )

            n_dot_l = numpy.linalg.multi_dot([vec, lightdir])

            sinalpha = sqrt(1 - n_dot_l * n_dot_l)

            lutcoords = numpy.array([sinalpha * lutheight * veclen, ( -n_dot_l * 0.5 * veclen + 0.5 ) * lutwidth])

            lutvalue = lut[(int(lutcoords[0]), int(lutcoords[1]))]
            intensity = lutvalue[0] + lutvalue[1] + lutvalue[2] + 0.0001
            lutvalue = lutvalue / intensity
            intensity *= intensity * wrapped_cos

            intensity = sqrt(intensity)
            lutvalue *= intensity


            sliceImg[(j, i)] = lutvalue + (numpy.array([0.2, 0.2, 0.2]) * 255) * wrapped_cos

    return sliceImg

