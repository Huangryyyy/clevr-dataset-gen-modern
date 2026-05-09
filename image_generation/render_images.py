# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
      sys.path.insert(0, script_dir)
    try:
      import utils
    except ImportError as e:
      print("\nERROR")
      print("Running render_images.py from Blender and cannot import utils.py.")
      print("Try running from the image_generation directory or add it to")
      print("Blender's Python path. For Blender 3.6 this is usually:")
      print("echo $PWD >> $BLENDER/3.6/python/lib/python3.10/site-packages/clevr.pth")
      sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--num_objects', default=None, type=int,
    help="If set, render exactly this many objects in every image. This " +
         "overrides --min_objects and --max_objects.")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")
parser.add_argument('--random_seed', '--seed', dest='random_seed',
    default=None, type=int,
    help="Optional random seed for reproducible image generation. When set, " +
         "each image uses seed + image_index so distributed runs are stable.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated Cycles rendering. " +
         "Use --gpu_backend to choose CUDA, OPTIX, HIP, ONEAPI, or METAL.")
parser.add_argument('--gpu_backend', default='CUDA',
    choices=['CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'],
    help="GPU backend to enable when --use_gpu 1 is set. CUDA and OPTIX are " +
         "the usual NVIDIA choices in Blender 3.6.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")


def set_if_exists(owner, attr, value):
  if hasattr(owner, attr):
    try:
      setattr(owner, attr, value)
    except TypeError:
      pass


def rotate_vector(rotation, vec):
  try:
    return rotation @ vec
  except TypeError:
    return rotation * vec


def add_plane(size):
  if bpy.app.version >= (2, 80, 0):
    bpy.ops.mesh.primitive_plane_add(size=size)
  else:
    bpy.ops.mesh.primitive_plane_add(radius=size / 2.0)


def set_render_tile_size(scene, tile_size):
  render_args = scene.render
  if hasattr(render_args, 'tile_x'):
    render_args.tile_x = tile_size
    render_args.tile_y = tile_size
  elif hasattr(scene, 'cycles') and hasattr(scene.cycles, 'tile_size'):
    scene.cycles.tile_size = tile_size


def get_preferences():
  if hasattr(bpy.context, 'preferences'):
    return bpy.context.preferences
  return bpy.context.user_preferences


def ensure_cycles_addon():
  prefs = get_preferences()
  if 'cycles' not in prefs.addons:
    if hasattr(bpy.ops, 'preferences'):
      bpy.ops.preferences.addon_enable(module='cycles')
    else:
      bpy.ops.wm.addon_enable(module='cycles')
  return prefs.addons['cycles'].preferences


def refresh_cycles_devices(cycles_prefs):
  if hasattr(cycles_prefs, 'refresh_devices'):
    cycles_prefs.refresh_devices()
  elif hasattr(cycles_prefs, 'get_devices'):
    cycles_prefs.get_devices()


def configure_cycles_devices(gpu_backend):
  cycles_prefs = ensure_cycles_addon()
  try:
    cycles_prefs.compute_device_type = gpu_backend
  except TypeError:
    msg = 'Cycles GPU backend "%s" is not available in this Blender build'
    raise RuntimeError(msg % gpu_backend)

  refresh_cycles_devices(cycles_prefs)
  enabled_devices = []
  devices = getattr(cycles_prefs, 'devices', [])
  for device in devices:
    use_device = getattr(device, 'type', '') == gpu_backend
    device.use = use_device
    if use_device:
      enabled_devices.append(device.name)

  if len(enabled_devices) == 0:
    print('WARNING: No %s Cycles devices were reported by Blender' %
          gpu_backend)
  else:
    print('Using Cycles %s device(s): %s' %
          (gpu_backend, ', '.join(enabled_devices)))


def get_object(name):
  return bpy.data.objects.get(name)


def validate_args(args):
  if args.num_objects is not None and args.num_objects < 0:
    raise ValueError('--num_objects must be nonnegative')
  if args.min_objects > args.max_objects:
    raise ValueError('--min_objects must be less than or equal to --max_objects')


def get_num_objects(args):
  if args.num_objects is not None:
    return args.num_objects
  return random.randint(args.min_objects, args.max_objects)


def main(args):
  validate_args(args)
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  
  all_scene_paths = []
  for i in range(args.num_images):
    output_index = i + args.start_idx
    scene_seed = None
    if args.random_seed is not None:
      scene_seed = args.random_seed + output_index
      random.seed(scene_seed)

    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    num_objects = get_num_objects(args)
    render_scene(args,
      num_objects=num_objects,
      output_index=output_index,
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
      random_seed=scene_seed,
    )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  info = {
    'date': args.date,
    'version': args.version,
    'split': args.split,
    'license': args.license,
  }
  if args.random_seed is not None:
    info['random_seed'] = args.random_seed
  output = {
    'info': info,
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)



def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
    random_seed=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.image_settings.file_format = 'PNG'
  render_args.image_settings.color_mode = 'RGBA'
  set_render_tile_size(bpy.context.scene, args.render_tile_size)
  if args.use_gpu == 1:
    configure_cycles_devices(args.gpu_backend)

  # Some CYCLES-specific stuff
  world = bpy.data.worlds.get('World')
  if world is not None and hasattr(world, 'cycles'):
    set_if_exists(world.cycles, 'sample_as_light', True)
  set_if_exists(bpy.context.scene.cycles, 'blur_glossy', 2.0)
  bpy.context.scene.cycles.samples = args.render_num_samples
  set_if_exists(bpy.context.scene.cycles, 'transparent_min_bounces',
                args.render_min_bounces)
  set_if_exists(bpy.context.scene.cycles, 'transparent_max_bounces',
                args.render_max_bounces)
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }
  if random_seed is not None:
    scene_struct['random_seed'] = random_seed

  # Put a plane on the ground so we can compute cardinal directions
  add_plane(size=10)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  camera_quat = camera.matrix_world.to_quaternion()
  cam_behind = rotate_vector(camera_quat, Vector((0, 0, -1)))
  cam_left = rotate_vector(camera_quat, Vector((-1, 0, 0)))
  cam_up = rotate_vector(camera_quat, Vector((0, 1, 0)))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(scene_struct, num_objects, args, camera):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  for i in range(num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)
      x = random.uniform(-3, 3)
      y = random.uniform(-3, 3)
      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
          dists_good = False
          break
        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < args.margin:
            print(margin, args.margin, direction_name)
            print('BROKEN MARGIN!')
            margins_good = False
            break
        if not margins_good:
          break

      if dists_good and margins_good:
        break

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  fd, path = tempfile.mkstemp(suffix='.png')
  os.close(fd)
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  bpy.data.images.remove(img)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  scene = bpy.context.scene
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = getattr(render_args, 'use_antialiasing', None)
  old_display_settings = {}
  old_shading_settings = {}
  old_view_settings = {}

  if hasattr(scene, 'display'):
    for attr in ['render_aa', 'viewport_aa']:
      if hasattr(scene.display, attr):
        old_display_settings[attr] = getattr(scene.display, attr)
    if hasattr(scene.display, 'shading'):
      for attr in ['light', 'color_type', 'show_shadows', 'show_cavity']:
        if hasattr(scene.display.shading, attr):
          old_shading_settings[attr] = getattr(scene.display.shading, attr)

  if hasattr(scene, 'view_settings'):
    for attr in ['view_transform', 'look', 'exposure', 'gamma']:
      if hasattr(scene.view_settings, attr):
        old_view_settings[attr] = getattr(scene.view_settings, attr)

  hidden_objects = []
  for name in ['Lamp_Key', 'Lamp_Fill', 'Lamp_Back', 'Ground']:
    obj = get_object(name)
    if obj is not None:
      hidden_objects.append((obj, obj.hide_render))

  object_colors = set()
  old_materials = []
  old_object_colors = []
  shadeless_materials = []
  use_workbench = bpy.app.version >= (2, 80, 0)

  try:
    # Override some render settings to have flat shading. Blender Internal was
    # removed in 2.80, so use Workbench's flat object-color pass in newer builds.
    render_args.filepath = path
    if use_workbench:
      render_args.engine = 'BLENDER_WORKBENCH'
      set_if_exists(scene.display, 'render_aa', 'OFF')
      set_if_exists(scene.display, 'viewport_aa', 'OFF')
      if hasattr(scene.display, 'shading'):
        set_if_exists(scene.display.shading, 'light', 'FLAT')
        set_if_exists(scene.display.shading, 'color_type', 'OBJECT')
        set_if_exists(scene.display.shading, 'show_shadows', False)
        set_if_exists(scene.display.shading, 'show_cavity', False)
      if hasattr(scene, 'view_settings'):
        set_if_exists(scene.view_settings, 'view_transform', 'Standard')
        set_if_exists(scene.view_settings, 'look', 'None')
        set_if_exists(scene.view_settings, 'exposure', 0)
        set_if_exists(scene.view_settings, 'gamma', 1)
    else:
      render_args.engine = 'BLENDER_RENDER'
      render_args.use_antialiasing = False

    for obj, _ in hidden_objects:
      obj.hide_render = True

    # Assign random flat colors to all objects.
    for i, obj in enumerate(blender_objects):
      old_materials.append(obj.data.materials[0] if len(obj.data.materials) else None)
      old_object_colors.append(tuple(obj.color) if hasattr(obj, 'color') else None)
      while True:
        r, g, b = [random.random() for _ in range(3)]
        if (r, g, b) not in object_colors:
          break
      object_colors.add((r, g, b))

      if use_workbench:
        obj.color = (r, g, b, 1.0)
      else:
        mat = bpy.data.materials.new('Material_%d' % i)
        utils.set_material_diffuse_color(mat, [r, g, b, 1.0])
        if hasattr(mat, 'use_shadeless'):
          mat.use_shadeless = True
        obj.data.materials[0] = mat
        shadeless_materials.append(mat)

    # Render the scene
    bpy.ops.render.render(write_still=True)

  finally:
    # Restore object materials and object colors.
    for mat, obj, color in zip(old_materials, blender_objects, old_object_colors):
      if mat is not None:
        if len(obj.data.materials) == 0:
          obj.data.materials.append(mat)
        else:
          obj.data.materials[0] = mat
      if color is not None:
        obj.color = color

    for mat in shadeless_materials:
      try:
        bpy.data.materials.remove(mat)
      except RuntimeError:
        pass

    for obj, old_hide_render in hidden_objects:
      obj.hide_render = old_hide_render

    # Set the render settings back to what they were.
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    if old_use_antialiasing is not None:
      render_args.use_antialiasing = old_use_antialiasing
    if hasattr(scene, 'display'):
      for attr, value in old_display_settings.items():
        setattr(scene.display, attr, value)
      if hasattr(scene.display, 'shading'):
        for attr, value in old_shading_settings.items():
          setattr(scene.display.shading, attr, value)
    if hasattr(scene, 'view_settings'):
      for attr, value in old_view_settings.items():
        setattr(scene.view_settings, attr, value)

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')
