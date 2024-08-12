import numpy as np

import torch
import torch.nn.functional as F


def vectors2rotmat(vec_rest, vec_final):
    batch_size = vec_final.shape[0]
    device = vec_final.device
    dtype = vec_final.dtype

    vec_final_norm = torch.norm(vec_final, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=1, keepdim=True)

    axis = torch.cross(vec_rest, vec_final, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    cos = torch.sum(vec_rest * vec_final, dim=1, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    axis = axis / (axis_norm + 1e-8)

    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat_loc


def get_pelvis_orient(rel_joint_pos, rel_rest_pose, parent_list):
    batch_size = rel_joint_pos.shape[0]
    device = rel_joint_pos.device
    dtype = rel_rest_pose.dtype

    pelvis_child = [1, 2, 3]

    spine_final_loc = rel_joint_pos[:, 1]
    spine_rest_loc = rel_rest_pose[:, 1]
    spine_norm = torch.norm(spine_final_loc.clone(), dim=1, keepdim=True)
    spine_norm = spine_final_loc.clone() / (spine_norm + 1e-8)

    rot_mat_spine = vectors2rotmat(spine_rest_loc.clone(), spine_final_loc.clone())

    center_final_loc = (rel_joint_pos[:, 1].clone() + rel_joint_pos[:, 2].clone()) / 2
    center_rest_loc = (rel_rest_pose[:, 1].clone() + rel_rest_pose[:, 2].clone()) / 2

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc)

    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / \
          (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def batch_vectors2rotmat(vec_rest, vec_final):
    batch_num, batch_size = vec_final.shape[0], vec_final.shape[1]
    device = vec_final.device
    dtype = vec_final.dtype

    vec_final_norm = torch.norm(vec_final, dim=2, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=2, keepdim=True)

    axis = torch.cross(vec_rest, vec_final, dim=2)
    axis_norm = torch.norm(axis, dim=2, keepdim=True)

    cos = torch.sum(vec_rest * vec_final, dim=2, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    axis = axis / (axis_norm + 1e-8)

    rx, ry, rz = torch.split(axis, 1, dim=2)
    zeros = torch.zeros((batch_num, batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
        .view((batch_num, batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat_loc


def batch_get_pelvis_orient(rel_joint_pos, rel_rest_pose, parent_list):
    batch_num, batch_size = rel_joint_pos.shape[0], rel_joint_pos.shape[1]
    device = rel_joint_pos.device
    dtype = rel_rest_pose.dtype

    pelvis_child = [1, 2, 3]

    spine_final_loc = rel_joint_pos[:, :, 1]
    spine_rest_loc = rel_rest_pose[:, :, 1]
    spine_norm = torch.norm(spine_final_loc.clone(), dim=2, keepdim=True)
    spine_norm = spine_final_loc.clone() / (spine_norm + 1e-8)

    rot_mat_spine = batch_vectors2rotmat(spine_rest_loc.clone(), spine_final_loc.clone())

    center_final_loc = (rel_joint_pos[:, :, 1].clone() + rel_joint_pos[:, :, 2].clone()) / 2
    center_rest_loc = (rel_rest_pose[:, :, 1].clone() + rel_rest_pose[:, :, 2].clone()) / 2

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc)

    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=2, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=2, keepdim=True) * spine_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=2, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=2, keepdim=True)

    axis = torch.cross(center_rest_loc, center_final_loc, dim=2)
    axis_norm = torch.norm(axis, dim=2, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=2, keepdim=True) / \
          (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=2)
    zeros = torch.zeros((batch_num, batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
        .view((batch_num, batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def batch_get_pelvis_orient_svd(rel_joint_pos, rel_rest_pose, parent_list):
    pelvis_child = [1, 2, 3]
    rest_mat = []
    target_mat = []
    for child in pelvis_child:
        rest_mat.append(rel_rest_pose[:, :, child])
        target_mat.append(rel_joint_pos[:, :, child])

    rest_mat = torch.cat(rest_mat, dim=3)
    target_mat = torch.cat(target_mat, dim=3)
    S = torch.matmul(rest_mat, target_mat.transpose(2, 3))
    mask_zero = S.sum(dim=(2, 3))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    device = S_non_zero.device
    U, _, V = torch.svd(S_non_zero)
    U = U.to(device=device)
    V = V.to(device=device)

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=device)

    det_u_v = torch.det(torch.matmul(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0

    return rot_mat


def batch_inverse_kinematics_naive(args, joint_pos, phi, joint_offset, parent_list,
                                   svd=False, detach=False):
    batch_num, batch_size, joint_num = joint_pos.shape[0], joint_pos.shape[1], joint_pos.shape[2]
    device = joint_pos.device

    rel_rest_pose = joint_offset.unsqueeze(-1)

    if detach:
        rel_joint_pos = joint_pos.unsqueeze(-1).clone().detach()
    else:
        rel_joint_pos = joint_pos.unsqueeze(-1).clone()
    rel_joint_pos[:, :] = rel_joint_pos[:, :] - rel_joint_pos[:, :, parent_list]
    rel_joint_pos[:, :, 0] *= 0

    res_joint_pos = torch.zeros_like(joint_pos)
    rot_mat_local = torch.eye(3).to(device).view((1, 1, 1, 3, 3)).repeat([batch_num, batch_size, joint_num, 1, 1])
    rot_mat_chain = torch.eye(3).to(device).view((1, 1, 1, 3, 3)).repeat([batch_num, batch_size, joint_num, 1, 1])

    if svd:
        root_rot_mat = batch_get_pelvis_orient_svd(rel_joint_pos.clone(), rel_rest_pose.clone(), parent_list)
    else:
        root_rot_mat = batch_get_pelvis_orient(rel_joint_pos.clone(), rel_rest_pose.clone(), parent_list)

    rot_mat_local[:, :, 0] = root_rot_mat
    rot_mat_chain[:, :, 0] = root_rot_mat

    children = -np.ones((parent_list.shape[0]), dtype=int)
    for i in range(1, parent_list.shape[0]):
        children[parent_list[i]] = i

    if args.DATA.train_part == "body":
        idx_levs = [[0],
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                    [13, 14, 15],
                    [18, 19, 20],
                    [21, 22, 23],
                    [24, 25, 26, 16, 17]]
    else:
        raise NotImplementedError

    for i in range(1, len(idx_levs)):
        indices = idx_levs[i]
        if i == len(idx_levs) - 1:
            pass
        else:
            len_indices = len(indices)

            child_final_loc = torch.matmul(rot_mat_chain[:, :, parent_list[indices]].transpose(3, 4),
                                           rel_joint_pos[:, :, children[indices]])
            child_rest_loc = rel_rest_pose[:, :, children[indices]]

            child_final_norm = torch.norm(child_final_loc, dim=3, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=3, keepdim=True)

            axis = torch.cross(child_rest_loc, child_final_loc, dim=3)
            axis_norm = torch.norm(axis, dim=3, keepdim=True)

            cos = torch.sum(child_rest_loc * child_final_loc, dim=3, keepdim=True) / \
                  (child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            axis = axis / (axis_norm + 1e-8)

            rx, ry, rz = torch.split(axis, 1, dim=3)
            zeros = torch.zeros((batch_num, batch_size, len_indices, 1, 1),
                                dtype=rx.dtype, device=rx.device)

            K = torch.cat([zeros, -rz, ry,
                           rz, zeros, -rx,
                           -ry, rx, zeros], dim=3).view((batch_num, batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=rx.dtype, device=rx.device).reshape((1, 1, 1, 3, 3))

            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            spin_axis = child_rest_loc / (child_rest_norm + 1e-8)

            rx, ry, rz = torch.split(spin_axis, 1, dim=3)
            zeros = torch.zeros((batch_num, batch_size, len_indices, 1, 1),
                                dtype=rx.dtype, device=rx.device)

            K = torch.cat([zeros, -rz, ry,
                           rz, zeros, -rx,
                           -ry, rx, zeros], dim=3).view((batch_num, batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=rx.dtype, device=rx.device).reshape((1, 1, 1, 3, 3))

            cos, sin = torch.split(phi[:, :, indices], 1, dim=3)
            cos = cos.unsqueeze(-1)
            sin = sin.unsqueeze(-1)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain[:, :, indices] = torch.matmul(rot_mat_chain[:, :, parent_list[indices]], rot_mat)
            rot_mat_local[:, :, indices] = rot_mat
            # break

    return rot_mat_local, res_joint_pos


def inverse_kinematics_naive(args, joint_pos, phi, joint_offset, parent_list,
                             svd=False, detach=False):
    batch_size, joint_num = joint_pos.shape[0], joint_pos.shape[1]
    device = joint_pos.device

    rel_rest_pose = joint_offset.unsqueeze(-1)

    if detach:
        rel_joint_pos = joint_pos.unsqueeze(-1).clone().detach()
    else:
        rel_joint_pos = joint_pos.unsqueeze(-1).clone()
    rel_joint_pos[:] = rel_joint_pos[:] - rel_joint_pos[:, parent_list]
    rel_joint_pos[:, 0] *= 0

    res_joint_pos = torch.zeros_like(joint_pos)
    rot_mat_local = torch.eye(3).to(device).view((1, 1, 3, 3)).repeat([batch_size, joint_num, 1, 1])
    rot_mat_chain = torch.eye(3).to(device).view((1, 1, 3, 3)).repeat([batch_size, joint_num, 1, 1])

    if svd:
        root_rot_mat = get_pelvis_orient_svd(rel_joint_pos.clone(), rel_rest_pose.clone(), parent_list)
    else:
        root_rot_mat = get_pelvis_orient(rel_joint_pos.clone(), rel_rest_pose.clone(), parent_list)

    rot_mat_local[:, 0] = root_rot_mat
    rot_mat_chain[:, 0] = root_rot_mat

    children = -np.ones((parent_list.shape[0]), dtype=int)
    for i in range(1, parent_list.shape[0]):
        children[parent_list[i]] = i

    if args.DATA.train_part == "body":
        idx_levs = [[0],
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                    [13, 14, 15],
                    [18, 19, 20],
                    [21, 22, 23],
                    [24, 25, 26, 16, 17]]
    else:
        raise NotImplementedError

    for i in range(1, len(idx_levs)):
        indices = idx_levs[i]
        if i == len(idx_levs) - 1:
            pass
        else:
            len_indices = len(indices)

            child_final_loc = torch.matmul(rot_mat_chain[:, parent_list[indices]].transpose(2, 3),
                                           rel_joint_pos[:, children[indices]])
            child_rest_loc = rel_rest_pose[:, children[indices]]

            child_final_norm = torch.norm(child_final_loc, dim=2, keepdim=True)
            child_rest_norm = torch.norm(child_rest_loc, dim=2, keepdim=True)

            axis = torch.cross(child_rest_loc, child_final_loc, dim=2)
            axis_norm = torch.norm(axis, dim=2, keepdim=True)

            cos = torch.sum(child_rest_loc * child_final_loc, dim=2, keepdim=True) / \
                  (child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            axis = axis / (axis_norm + 1e-8)

            rx, ry, rz = torch.split(axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1),
                                dtype=rx.dtype, device=rx.device)

            K = torch.cat([zeros, -rz, ry,
                           rz, zeros, -rx,
                           -ry, rx, zeros], dim=2).view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=rx.dtype, device=rx.device).reshape((1, 1, 3, 3))

            rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

            spin_axis = child_rest_loc / (child_rest_norm + 1e-8)

            rx, ry, rz = torch.split(spin_axis, 1, dim=2)
            zeros = torch.zeros((batch_size, len_indices, 1, 1),
                                dtype=rx.dtype, device=rx.device)

            K = torch.cat([zeros, -rz, ry,
                           rz, zeros, -rx,
                           -ry, rx, zeros], dim=2).view((batch_size, len_indices, 3, 3))
            ident = torch.eye(3, dtype=rx.dtype, device=rx.device).reshape((1, 1, 3, 3))

            cos, sin = torch.split(phi[:, indices], 1, dim=2)
            cos = cos.unsqueeze(-1)
            sin = sin.unsqueeze(-1)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.matmul(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain[:, indices] = torch.matmul(rot_mat_chain[:, parent_list[indices]], rot_mat)
            rot_mat_local[:, indices] = rot_mat
            # break

    return rot_mat_local, res_joint_pos


def inverse_kinematics(args, joint_pos, phi, joint_offset, parent_list, svd=False, detach=False):
    batch_num, joint_num = joint_pos.shape[0], joint_pos.shape[1]
    dtype = joint_pos.dtype
    device = joint_pos.device

    rel_rest_pose = joint_offset.unsqueeze(-1).clone()

    if detach:
        rel_joint_pos = joint_pos.unsqueeze(-1).clone().detach()
    else:
        rel_joint_pos = joint_pos.unsqueeze(-1).clone()
    rel_joint_pos[:, 1:] = rel_joint_pos[:, 1:] - rel_joint_pos[:, parent_list[1:]].clone()
    rel_joint_pos[:, 0] *= 0

    rotate_rest_pose = torch.zeros_like(rel_rest_pose)
    rotate_rest_pose[:, 0] = joint_pos[:, 0].unsqueeze(2)

    rel_pose_skeleton = joint_pos.unsqueeze(-1).clone()
    rel_pose_skeleton[:, 1:] = rel_pose_skeleton[:, 1:] - rel_pose_skeleton[:, parent_list[1:]].clone()
    rel_pose_skeleton[:, 0] = rel_rest_pose[:, 0]

    final_pose_skeleton = joint_pos.unsqueeze(-1).clone()
    # final_pose_skeleton = final_pose_skeleton - final_pose_skeleton[:, 0:1] + rel_rest_pose[:, 0:1]

    # res_joint_pos = torch.zeros_like(rel_joint_pos)
    # res_joint_pos[:, :, 0] = rel_joint_pos[:, :, 0]

    if svd:
        root_rot_mat = get_pelvis_orient_svd(rel_joint_pos.clone(), rel_rest_pose.clone(), parent_list)
    else:
        root_rot_mat = get_pelvis_orient(rel_joint_pos.clone(), rel_rest_pose.clone(), parent_list)

    rot_mat_local = [root_rot_mat]
    rot_mat_chain = [root_rot_mat]

    children = -np.ones((parent_list.shape[0]), dtype=int)
    for i in range(1, parent_list.shape[0]):
        children[parent_list[i]] = i

    for i in range(1, len(parent_list)):
        if children[i] == -1:
            rot_mat_local.append(torch.eye(3).to(device).view((1, 3, 3)).repeat([batch_num, 1, 1]))
            rot_mat_chain.append(torch.eye(3).to(device).view((1, 3, 3)).repeat([batch_num, 1, 1]))
        else:
            rotate_rest_pose[:, i] = rotate_rest_pose[:, parent_list[i]] + \
                                     torch.matmul(rot_mat_chain[parent_list[i]],
                                                  rel_rest_pose[:, i])

            child_final_loc = final_pose_skeleton[:, children[i]] - \
                              rotate_rest_pose[:, i]

            if detach:
                orig_vec = rel_pose_skeleton[:, children[i]]
                template_vec = rel_rest_pose[:, children[i]]
                norm_t = torch.norm(template_vec, dim=1, keepdim=True)
                orig_vec = orig_vec * norm_t / torch.norm(orig_vec, dim=1, keepdim=True)

                diff = torch.norm(child_final_loc - orig_vec, dim=1, keepdim=True)
                big_diff_idx = torch.where(diff > 15 / 1000)[0]

                child_final_loc[big_diff_idx] = orig_vec[big_diff_idx]

            child_final_loc = torch.matmul(rot_mat_chain[parent_list[i]].transpose(1, 2),
                                           child_final_loc)
            child_rest_loc = rel_rest_pose[:, children[i]]

            child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
            child_final_norm = torch.norm(child_final_loc, dim=1, keepdim=True)

            # (B, 3, 1)
            axis = torch.cross(child_rest_loc, child_final_loc, dim=1)
            axis_norm = torch.norm(axis, dim=1, keepdim=True)

            # (B, 1, 1)
            cos = torch.sum(child_rest_loc * child_final_loc, dim=1, keepdim=True) / (
                    child_rest_norm * child_final_norm + 1e-8)
            sin = axis_norm / (child_rest_norm * child_final_norm + 1e-8)

            # (B, 3, 1)
            axis = axis / (axis_norm + 1e-8)

            # Convert location revolve to rot_mat by rodrigues
            # (B, 1, 1)
            rx, ry, rz = torch.split(axis, 1, dim=1)
            zeros = torch.zeros((batch_num, 1, 1), dtype=dtype, device=device)

            K = torch.cat([zeros, -rz, ry,
                           rz, zeros, -rx,
                           -ry, rx, zeros], dim=1) \
                .view((batch_num, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

            spin_axis = child_rest_loc / child_rest_norm
            # (B, 1, 1)
            rx, ry, rz = torch.split(spin_axis, 1, dim=1)
            zeros = torch.zeros((batch_num, 1, 1), dtype=dtype, device=device)
            K = torch.cat([zeros, -rz, ry,
                           rz, zeros, -rx,
                           -ry, rx, zeros], dim=1) \
                .view((batch_num, 3, 3))
            ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
            # (B, 1, 1)
            cos, sin = torch.split(phi[:, i], 1, dim=1)
            cos = torch.unsqueeze(cos, dim=2)
            sin = torch.unsqueeze(sin, dim=2)
            rot_mat_spin = ident + sin * K + (1 - cos) * torch.bmm(K, K)
            rot_mat = torch.matmul(rot_mat_loc, rot_mat_spin)

            rot_mat_chain.append(torch.matmul(rot_mat_chain[parent_list[i]],
                                              rot_mat))
            rot_mat_local.append(rot_mat)

    return torch.stack(rot_mat_local, dim=1), rotate_rest_pose
