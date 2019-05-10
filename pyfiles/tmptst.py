import numpy as np
import scipy.ndimage.filters as filters
import BVH, Animation
from Quaternions import Quaternions
from Pivots import Pivots

def process_file(filename, window=240, window_step=120, export_trajectory=False):

    anim, names, frametime = BVH.load(filename)

    """ Subsample to 60 fps """
    anim = anim[::2]

    """ Do FK """
    global_xforms = Animation.transforms_global(anim)  # intermediate
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]
    global_rotations = Quaternions.from_transforms(global_xforms)


    """ Remove Uneeded Joints """ #>># done post-hoc in PFNN
    used_joints = np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])

    positions = global_positions[:,used_joints]
    global_rotations = global_rotations[:,used_joints]
    # ________________________________________________________

    """ Put on Floor """
    positions[:,:,1] -= positions[:,:,1].min()

    """ Get Foot Contacts """
    if True:
        velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])

        fid_l, fid_r = np.array([3,4]), np.array([7,8])
        feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
        feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
        feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

        feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
        feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
        feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

    """ Get Root Velocity """
    velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()

    """ Remove Translation """
    positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
    positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Get Root Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    root_rotation = Quaternions.between(forward, target)[:,np.newaxis]   # rotation needed fwd->z?
    rvelocity = (root_rotation[1:] * -root_rotation[:-1]).to_pivots()

    """ Local Space """  # NEW: define position of joints relative to
    local_positions = positions.copy()
    local_positions[:,:,0] = local_positions[:,:,0] - local_positions[:,0:1,0]  # x rel to root x
    local_positions[:,:,2] = local_positions[:,:,2] - local_positions[:,0:1,2]  # z rel to root z

    local_positions = root_rotation[:-1] * local_positions[:-1]   # remove Y rotation from pos
    local_velocities = local_positions[1:] - local_positions[:-1]
    local_rotations = abs((root_rotation[:-1] * global_rotations[:-1])).log()

    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps
    global_velocities = root_rotation[:-1] * velocity              # remove Y rotation from vel


    assert global_velocities.shape[1] == 1, "output assumes global_velocities dim2 = 1."
    n = root_rvelocity.shape[0]
    omit_end = range(0, n-1)
    out = np.hstack((root_rvelocity[omit_end,:],
                global_velocities[omit_end,:,0],
                global_velocities[omit_end,:,2],
                feet_l[omit_end,:], feet_r[omit_end,:],
                local_positions[omit_end].reshape(n-1, -1),
                local_velocities.reshape(n-1, -1),
                local_rotations[omit_end].reshape(n-1, -1)))
    return out
