import numpy as np
import pythreejs as pjs
import torch

def to_numpy(tensor):
    return tensor.permute(1, 2, 0).numpy()

def to_uint8(array):
    return (array * 255.0).astype(np.uint8)

def visualize_3d(xyz, rgb=None, size=0.03, height=480, width=480):
    points_buf = pjs.BufferAttribute(array=xyz)
    geometryAttrs = {'position': points_buf}

    if rgb is not None:
        colors_buf = pjs.BufferAttribute(array=rgb)
        geometryAttrs['color'] = colors_buf

    geometry = pjs.BufferGeometry(attributes=geometryAttrs)

    material = pjs.PointsMaterial(vertexColors='VertexColors', size=size)
    pointCloud = pjs.Points(geometry=geometry, material=material)

    pythreejs_camera = pjs.PerspectiveCamera(
        up=[1, 0, 1],
        children=[pjs.DirectionalLight(color='white', intensity=0.5)])

    pythreejs_camera.rotateX(np.pi/4)
    pythreejs_camera.position = (-15., 0., 30.)

    scene = pjs.Scene(children=[
                    pointCloud,
                    pythreejs_camera,
                    pjs.AmbientLight(color='#777777')])

    axes = pjs.AxesHelper(size=3)
    scene.add(axes)

    control = pjs.OrbitControls(controlling=pythreejs_camera)
    renderer = pjs.Renderer(camera=pythreejs_camera,
                        scene=scene,
                        width=width,
                        height=height,
                        preserveDrawingBuffer=True,
                        controls=[control])

    return renderer

def visualize_3d_list(xyz_list, rgb_list, size=0.03, height=480, width=480):
    material = pjs.PointsMaterial(vertexColors='VertexColors', size=size)
    pointClouds = []
    for xyz, rgb in zip(xyz_list, rgb_list):
        points_buf = pjs.BufferAttribute(array=xyz)
        geometryAttrs = {'position': points_buf}

        colors_buf = pjs.BufferAttribute(array=rgb)
        geometryAttrs['color'] = colors_buf

        geometry = pjs.BufferGeometry(attributes=geometryAttrs)
        pointClouds.append(pjs.Points(geometry=geometry, material=material))

    pythreejs_camera = pjs.PerspectiveCamera(
        up=[1, 0, 1],
        children=[pjs.DirectionalLight(color='white', intensity=0.5)])

    pythreejs_camera.rotateX(np.pi/4)
    pythreejs_camera.position = (-15., 0., 30.)

    axes = pjs.AxesHelper(size=3)

    scene = pjs.Scene(children=pointClouds + [
                    pythreejs_camera,
                    pjs.AmbientLight(color='#777777'),
                    axes])

    control = pjs.OrbitControls(controlling=pythreejs_camera)
    renderer = pjs.Renderer(camera=pythreejs_camera,
                        scene=scene,
                        width=width,
                        height=height,
                        preserveDrawingBuffer=True,
                        controls=[control])

    return renderer