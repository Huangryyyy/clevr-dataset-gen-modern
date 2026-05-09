# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import sys, random, os, math
import bpy, bpy_extras


"""
Some utility functions for interacting with Blender
"""


def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))


def blender_version_at_least(major, minor=0, patch=0):
  return bpy.app.version >= (major, minor, patch)


def set_active_object(obj):
  if blender_version_at_least(2, 80, 0):
    bpy.context.view_layer.objects.active = obj
  else:
    bpy.context.scene.objects.active = obj


def select_object(obj, state=True):
  if hasattr(obj, 'select_set'):
    obj.select_set(state)
  else:
    obj.select = state


# I wonder if there's a better way to do this?
def delete_object(obj):
  """ Delete a specified blender object """
  if obj is None:
    return
  bpy.ops.object.select_all(action='DESELECT')
  set_active_object(obj)
  select_object(obj, True)
  bpy.ops.object.delete()


def get_camera_coords(cam, pos):
  """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
  scale = scene.render.resolution_percentage / 100.0
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def set_layer(obj, layer_idx):
  """ Move an object to a particular layer """
  if blender_version_at_least(2, 80, 0):
    # Blender 2.80 replaced layers with collections. The only use in this
    # project is to hide helper objects during a render pass.
    obj.hide_render = (layer_idx != 0)
    obj.hide_set(layer_idx != 0)
    return

  # Set the target layer to True first because an object must always be on
  # at least one layer.
  obj.layers[layer_idx] = True
  for i in range(len(obj.layers)):
    obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, scale, loc, theta=0):
  """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
  # First figure out how many of this object are already in the scene so we can
  # give the new object a unique name
  count = 0
  for obj in bpy.data.objects:
    if obj.name.startswith(name):
      count += 1

  old_objects = set(bpy.data.objects.keys())
  filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
  bpy.ops.wm.append(filename=filename)
  new_objects = [obj for obj in bpy.data.objects if obj.name not in old_objects]
  assert len(new_objects) == 1, 'Expected one object in %s' % filename
  obj = new_objects[0]

  # Give it a new name to avoid conflicts
  new_name = '%s_%d' % (name, count)
  obj.name = new_name

  # Set the new object as active, then rotate, scale, and translate it
  x, y = loc
  bpy.ops.object.select_all(action='DESELECT')
  set_active_object(obj)
  select_object(obj, True)
  obj.rotation_euler[2] = math.radians(theta)
  obj.scale = (scale, scale, scale)
  obj.location = (x, y, scale)


def load_materials(material_dir):
  """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'): continue
    name = os.path.splitext(fn)[0]
    filepath = os.path.join(material_dir, fn, 'NodeTree', name)
    bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
  """
  Create a new material and assign it to the active object. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
  # Figure out how many materials are already in the scene
  mat_count = len(bpy.data.materials)

  # Create a new node material directly. This works in both Blender 2.7x
  # and the 2.80+ collection/view-layer API.
  mat = bpy.data.materials.new('Material_%d' % mat_count)
  mat.use_nodes = True
  if 'Color' in properties:
    set_material_diffuse_color(mat, properties['Color'])

  # Attach the new material to the active object
  # Make sure it doesn't already have materials
  obj = bpy.context.active_object
  assert len(obj.data.materials) == 0
  obj.data.materials.append(mat)

  # Find the output node of the new material
  output_node = None
  for n in mat.node_tree.nodes:
    if n.name == 'Material Output' or getattr(n, 'type', None) == 'OUTPUT_MATERIAL':
      output_node = n
      break
  assert output_node is not None, 'Could not find material output node'

  # Add a new GroupNode to the node tree of the active material,
  # and copy the node tree from the preloaded node group to the
  # new group node. This copying seems to happen by-value, so
  # we can create multiple materials of the same type without them
  # clobbering each other
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]

  # Find and set the "Color" input of the new group node
  for inp in group_node.inputs:
    if inp.name in properties:
      inp.default_value = properties[inp.name]

  # Wire the output of the new group node to the input of
  # the MaterialOutput node
  surface_input = output_node.inputs['Surface']
  for link in list(mat.node_tree.links):
    if link.to_node == output_node and link.to_socket == surface_input:
      mat.node_tree.links.remove(link)
  mat.node_tree.links.new(
      group_node.outputs['Shader'],
      surface_input,
  )


def set_material_diffuse_color(mat, rgba):
  try:
    mat.diffuse_color = rgba
  except (TypeError, ValueError):
    mat.diffuse_color = rgba[:3]
