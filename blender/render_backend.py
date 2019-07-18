import argparse
import os
import math
import bpy
import numpy as np
import sys
from transforms3d.euler import euler2mat
import itertools
import glob

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.dirname(UTILS_DIR)
ROOT_DIR = os.path.dirname(LIB_DIR)

sys.path.append(UTILS_DIR)
sys.path.append(LIB_DIR)
sys.path.append(ROOT_DIR)

from config import cfg
from blender.blender_utils import get_K_P_from_blender, get_3x4_P_matrix_from_blender
import pickle
import time


def parse_argument():
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--input', type=str, default='./doc/car/models/model_normalized.obj',
                        help='The cad model to be rendered')
    parser.add_argument('--output_dir', type=str, default='/tmp',
                        help='The directory of the output 2d image.')
    parser.add_argument('--bg_imgs', type=str, default='/tmp',
                        help='Names of background images stored in a .npy file.')
    parser.add_argument('--poses_path', type=str, default='/tmp',
                        help='6d poses(azimuth, euler, theta, x, y, z) stored in a .npy file.')
    parser.add_argument('--use_cycles', type=str, default='False',
                        help='Decide whether to use cycles render or not.')
    parser.add_argument('--azi', type=float, default=0.0,
                        help='Azimuth of camera.')
    parser.add_argument('--ele', type=float, default=0.0,
                        help='Elevation of camera.')
    parser.add_argument('--theta', type=float, default=0.0,
                        help='In-plane rotation angle of camera.')
    parser.add_argument('--height', type=float, default=0.0,
                        help='Location z of plane.')

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    return args


def place_camera(cam, dis, azi, ele):
    azi = azi + 90
    azi = math.radians(azi)
    ele = math.radians(ele)
    cam.location = (dis * math.cos(ele) * math.cos(azi), dis * math.cos(ele) * math.sin(azi), dis * math.sin(ele))


def obj_location(dist, azi, ele):
    ele = math.radians(ele)
    azi = math.radians(azi)
    x = dist * math.cos(azi) * math.cos(ele)
    y = dist * math.sin(azi) * math.cos(ele)
    z = dist * math.sin(ele)
    return x, y, z


def setup_light(scene):
    bpy.ops.object.select_by_type(type='LAMP')
    bpy.ops.object.delete(use_global=False)

    for i in range(2):
        azi = np.random.uniform(0, 360)
        ele = np.random.uniform(0, 40)
        dist = np.random.uniform(1, 2)
        x, y, z = obj_location(dist, azi, ele)
        lamp_name = 'Lamp{}'.format(i)
        lamp_data = bpy.data.lamps.new(name=lamp_name, type='POINT')
        lamp_data.energy = np.random.uniform(0.5, 2)
        lamp = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)
        lamp.location = (x, y, z)
        scene.objects.link(lamp)


def setup():
    bpy.ops.object.select_all(action='TOGGLE')
    camera = bpy.data.objects['Camera']
    bpy.data.cameras['Camera'].clip_end = 10000

    # configure rendered image's parameters
    bpy.context.scene.render.resolution_x = cfg.WIDTH
    bpy.context.scene.render.resolution_y = cfg.HEIGHT
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

    # modify the camera intrinsic matrix
    # bpy.data.cameras['Camera'].sensor_width = 39.132693723430386
    # bpy.context.scene.render.pixel_aspect_y = 1.6272340492401836

    cam_constraint = camera.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(camera)
    # cam_constraint.target = b_empty

    # composite node
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new(type="CompositorNodeRLayers")
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.base_path = ''
    depth_file_output.format.file_format = 'OPEN_EXR'
    depth_file_output.format.color_depth = '32'

    map_node = tree.nodes.new(type="CompositorNodeMapRange")
    map_node.inputs[1].default_value = cfg.MIN_DEPTH
    map_node.inputs[2].default_value = cfg.MAX_DEPTH
    map_node.inputs[3].default_value = 0
    map_node.inputs[4].default_value = 1
    links.new(rl.outputs['Depth'], map_node.inputs[0])
    links.new(map_node.outputs[0], depth_file_output.inputs[0])

    return camera, depth_file_output


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return q1, q2, q3, q4


def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist
    t = math.sqrt(cx * cx + cy * cy)
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx * cx + ty * cy, -1), 1)
    # roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return q1, q2, q3, q4


def camRotQuaternion(cx, cy, cz, theta):
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return q1, q2, q3, q4


def quaternionProduct(qx, qy):
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e
    return q1, q2, q3, q4


def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return x, y, z


def render(camera, outfile, pose):
    bpy.context.scene.render.filepath = outfile
    depth_file_output.file_slots[0].path = bpy.context.scene.render.filepath + '_depth.png'

    azimuth, elevation, theta = pose[:3]
    cx, cy, cz = obj_centened_camera_pos(cfg.cam_dist, azimuth, elevation)
    q1 = camPosToQuaternion(cx, cy, cz)
    q2 = camRotQuaternion(cx, cy, cz, theta)
    q = quaternionProduct(q2, q1)
    camera.location[0] = cx  # + np.random.uniform(-cfg.pose_noise,g_camPos_noise)
    camera.location[1] = cy  # + np.random.uniform(-g_camPos_noise,g_camPos_noise)
    camera.location[2] = cz  # + np.random.uniform(-g_camPos_noise,g_camPos_noise)
    camera.rotation_mode = 'QUATERNION'

    camera.rotation_quaternion[0] = q[0]
    camera.rotation_quaternion[1] = q[1]
    camera.rotation_quaternion[2] = q[2]
    camera.rotation_quaternion[3] = q[3]
    # camera.location = [0, 1, 0]
    # camera.rotation_euler = [np.pi / 2, 0, np.pi]

    setup_light(bpy.context.scene)
    rotation_matrix = get_K_P_from_blender(camera)['RT'][:, :3]
    camera.location = -np.dot(rotation_matrix.T, pose[3:])
    bpy.ops.render.render(write_still=True)


def add_shader_on_world():
    bpy.data.worlds['World'].use_nodes = True
    env_node = bpy.data.worlds['World'].node_tree.nodes.new(type='ShaderNodeTexEnvironment')
    back_node = bpy.data.worlds['World'].node_tree.nodes['Background']
    bpy.data.worlds['World'].node_tree.links.new(env_node.outputs['Color'], back_node.inputs['Color'])


def add_shader_on_ply_object(obj):
    bpy.ops.material.new()
    material = list(bpy.data.materials)[0]

    material.use_nodes = True
    material.node_tree.links.clear()

    mat_out = material.node_tree.nodes['Material Output']
    diffuse_node = material.node_tree.nodes['Diffuse BSDF']
    gloss_node = material.node_tree.nodes.new(type='ShaderNodeBsdfGlossy')
    attr_node = material.node_tree.nodes.new(type='ShaderNodeAttribute')

    material.node_tree.nodes.remove(diffuse_node)
    attr_node.attribute_name = 'Col'
    material.node_tree.links.new(attr_node.outputs['Color'], gloss_node.inputs['Color'])
    material.node_tree.links.new(gloss_node.outputs['BSDF'], mat_out.inputs['Surface'])

    obj.data.materials.append(material)

    return material


def add_shader_on_obj_object(obj):
    bpy.ops.material.new()
    material = list(bpy.data.materials)[0]

    material.use_nodes = True
    material.node_tree.links.clear()

    mat_out = material.node_tree.nodes['Material Output']
    diffuse_node = material.node_tree.nodes['Diffuse BSDF']
    image_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')

    material.node_tree.links.new(diffuse_node.outputs['BSDF'], mat_out.inputs['Surface'])
    material.node_tree.links.new(image_node.outputs['Color'], diffuse_node.inputs['Color'])
    img_path = '/home/pengsida/Datasets/YCB/models/002_master_chef_can/texture_map.png'
    img_name = os.path.basename(img_path)
    bpy.data.images.load(img_path)
    image_node.image = bpy.data.images[img_name]

    obj.data.materials.clear()
    obj.data.materials.append(material)

    return material


def add_shader_on_plane(plane):
    bpy.ops.material.new()
    material = list(bpy.data.materials)[1]

    material.use_nodes = True
    material.node_tree.links.clear()

    mat_out = material.node_tree.nodes['Material Output']
    diffuse_node = material.node_tree.nodes['Diffuse BSDF']
    image_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')

    material.node_tree.links.new(image_node.outputs['Color'], diffuse_node.inputs['Color'])
    material.node_tree.links.new(diffuse_node.outputs['BSDF'], mat_out.inputs['Surface'])

    img_path = '/home/pengsida/Pictures/board.png'
    img_name = os.path.basename(img_path)
    bpy.data.images.load(img_path)
    image_node.image = bpy.data.images[img_name]
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.unwrap()
    bpy.ops.object.mode_set(mode='OBJECT')

    plane.data.materials.append(material)


def set_material_node_parameters(material):
    nodes = material.node_tree.nodes
    if os.path.basename(args.input).endswith('.ply'):
        nodes['Glossy BSDF'].inputs['Roughness'].default_value = np.random.uniform(0.8, 1)
    else:
        nodes['Diffuse BSDF'].inputs['Roughness'].default_value = np.random.uniform(0, 1)


def batch_render_with_linemod(args, camera):
    os.system('mkdir -p {}'.format(args.output_dir))
    bpy.ops.import_mesh.ply(filepath=args.input)
    object = bpy.data.objects[os.path.basename(args.input).replace('.ply', '')]
    bpy.context.scene.render.image_settings.file_format = 'JPEG'

    # set up the cycles render configuration
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.sample_clamp_indirect = 1.0
    bpy.context.scene.cycles.blur_glossy = 3.0
    bpy.context.scene.cycles.samples = 100

    bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = 'GPU'

    for mesh in bpy.data.meshes:
        mesh.use_auto_smooth = True

    add_shader_on_world()

    material = add_shader_on_ply_object(object)
    # add a plane under the object
    # bpy.ops.mesh.primitive_plane_add()
    # plane = bpy.data.objects['Plane']
    # plane.location = [0, 0, args.height]
    # plane.scale = [0.28, 0.28, 0.28]
    # add_shader_on_plane(plane)

    bg_imgs = np.load(args.bg_imgs).astype(np.str)
    bg_imgs = np.random.choice(bg_imgs, size=cfg.NUM_SYN)
    poses = np.load(args.poses_path)
    begin_num_imgs = len(glob.glob(os.path.join(args.output_dir, '*.jpg')))
    for i in range(begin_num_imgs, cfg.NUM_SYN):
        # overlay an background image and place the object
        img_name = os.path.basename(bg_imgs[i])
        bpy.data.images.load(bg_imgs[i])
        bpy.data.worlds['World'].node_tree.nodes['Environment Texture'].image = bpy.data.images[img_name]
        pose = poses[i]
        # x, y = np.random.uniform(-0.15, 0.15, size=2)
        x, y = 0, 0
        object.location = [x, y, 0]
        set_material_node_parameters(material)
        render(camera, '{}/{}'.format(args.output_dir, i), pose)
        object_to_world_pose = np.array([[1, 0, 0, x],
                                         [0, 1, 0, y],
                                         [0, 0, 1, 0]])
        object_to_world_pose = np.append(object_to_world_pose, [[0, 0, 0, 1]], axis=0)
        KRT = get_K_P_from_blender(camera)
        world_to_camera_pose = np.append(KRT['RT'], [[0, 0, 0, 1]], axis=0)
        world_to_camera_pose = np.dot(world_to_camera_pose, object_to_world_pose)[:3]
        with open('{}/{}_RT.pkl'.format(args.output_dir, i), 'wb') as f:
            pickle.dump({'RT': world_to_camera_pose, 'K': KRT['K']}, f)
        bpy.data.images.remove(bpy.data.images[img_name])


def batch_render_ycb(args, camera):
    os.system('mkdir -p {}'.format(args.output_dir))
    bpy.ops.import_scene.obj(filepath=args.input)
    object = list(bpy.data.objects)[-1]
    bpy.context.scene.render.image_settings.file_format = 'JPEG'

    # set up the cycles render configuration
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.sample_clamp_indirect = 1.0
    bpy.context.scene.cycles.blur_glossy = 3.0
    bpy.context.scene.cycles.samples = 100

    bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = 'GPU'

    for mesh in bpy.data.meshes:
        mesh.use_auto_smooth = True

    add_shader_on_world()

    material = add_shader_on_obj_object(object)
    # add a plane under the object
    # bpy.ops.mesh.primitive_plane_add()
    # plane = bpy.data.objects['Plane']
    # plane.location = [0, 0, args.height]
    # plane.scale = [0.28, 0.28, 0.28]
    # add_shader_on_plane(plane)

    bg_imgs = np.load(args.bg_imgs).astype(np.str)
    bg_imgs = np.random.choice(bg_imgs, size=cfg.NUM_SYN)
    poses = np.load(args.poses_path)
    begin_num_imgs = len(glob.glob(os.path.join(args.output_dir, '*.jpg')))
    for i in range(begin_num_imgs, cfg.NUM_SYN):
        # overlay an background image and place the object
        img_name = os.path.basename(bg_imgs[i])
        bpy.data.images.load(bg_imgs[i])
        bpy.data.worlds['World'].node_tree.nodes['Environment Texture'].image = bpy.data.images[img_name]
        pose = poses[i]
        # x, y = np.random.uniform(-0.15, 0.15, size=2)
        x, y = 0, 0

        azi, ele, theta = (0, 0, 0)
        object.rotation_euler = (azi, ele, theta)
        azi, ele, theta = object.rotation_euler
        object.location = [x, y, 0]
        set_material_node_parameters(material)
        render(camera, '{}/{}'.format(args.output_dir, i), pose)

        rotation = euler2mat(azi, ele, theta)
        object_to_world_pose = np.concatenate([rotation, [[x], [y], [0]]], axis=-1)
        object_to_world_pose = np.append(object_to_world_pose, [[0, 0, 0, 1]], axis=0)
        KRT = get_K_P_from_blender(camera)
        world_to_camera_pose = np.append(KRT['RT'], [[0, 0, 0, 1]], axis=0)
        world_to_camera_pose = np.dot(world_to_camera_pose, object_to_world_pose)[:3]
        with open('{}/{}_RT.pkl'.format(args.output_dir, i), 'wb') as f:
            pickle.dump({'RT': world_to_camera_pose, 'K': KRT['K']}, f)
        bpy.data.images.remove(bpy.data.images[img_name])


if __name__ == '__main__':
    begin = time.time()
    args = parse_argument()
    camera, depth_file_output = setup()
    if os.path.basename(args.input).endswith('.ply'):
        batch_render_with_linemod(args, camera)
    else:
        batch_render_ycb(args, camera)
    print('cost {} s'.format(time.time() - begin))

