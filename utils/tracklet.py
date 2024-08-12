import numpy as np
from sklearn.metrics import pairwise_distances


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = np.linalg.norm(a, axis=1)[:, None], np.linalg.norm(b, axis=1)[:, None]
    a_norm = a / np.maximum(a_n, eps * np.ones_like(a_n))
    b_norm = b / np.maximum(b_n, eps * np.ones_like(b_n))
    sim_mt = np.matmul(a_norm, b_norm.transpose())
    return sim_mt


def link_frame(args, marker_pos_frame_1, marker_pos_frame_2, marker_feature_frame_1, marker_feature_frame_2):
    pos_diff_threshold = args.pos_diff_threshold
    sim_diff_threshold = args.sim_diff_threshold
    dist_mat = pairwise_distances(marker_pos_frame_1, marker_pos_frame_2)
    sim_mat = 1.0 - sim_matrix(marker_feature_frame_1[-1], marker_feature_frame_2[-1])
    score_mat = dist_mat + sim_mat * args.sim_mat_lambda

    res = []
    while True:
        min_dist_idx = np.unravel_index(np.argmin(score_mat), score_mat.shape)
        if dist_mat[min_dist_idx] > pos_diff_threshold or sim_mat[min_dist_idx] > sim_diff_threshold:
            break
        res.append(min_dist_idx)
        score_mat[min_dist_idx[0], :] = 1e10
        score_mat[:, min_dist_idx[1]] = 1e10
        dist_mat[min_dist_idx[0], :] = 1e10
        dist_mat[:, min_dist_idx[1]] = 1e10
        sim_mat[min_dist_idx[0], :] = 1e10
        sim_mat[:, min_dist_idx[1]] = 1e10
    return res


def generate_tracklet(frame_link_arr, anim_length, marker_num):
    res = []

    linked_list_arr = np.ones((anim_length, marker_num, 2), dtype=int) * -1
    visited = np.zeros((anim_length, marker_num), dtype=bool)

    for link in frame_link_arr:
        link_pos, link_next = link[0], link[1]
        linked_list_arr[link_pos[0], link_pos[1]] = link_next

    for frame_idx in range(anim_length):
        for marker_idx in range(marker_num):
            if not visited[frame_idx, marker_idx]:
                cur = [frame_idx, marker_idx]
                linked_list = []

                while not np.any(linked_list_arr[cur[0], cur[1]] == (-1, -1)):
                    visited[cur[0], cur[1]] = True
                    linked_list.append(cur)
                    next_pos = linked_list_arr[cur[0], cur[1]].copy()
                    linked_list_arr[cur[0], cur[1]] = (-1, -1)
                    cur = next_pos
                visited[cur[0], cur[1]] = True
                linked_list.append(cur)
                res.append(linked_list)

    return res


def tracklet_labeling(args, label_confidence, marker_pos, point_feature):
    anim_length, marker_num = marker_pos.shape[0], marker_pos.shape[1]
    frame_link_arr = []
    for i in range(anim_length - 1):
        frame_links = link_frame(args, marker_pos[i], marker_pos[i + 1], point_feature[i], point_feature[i + 1])
        frame_links = [((i, l[0]), (i + 1, l[1])) for l in frame_links]
        frame_link_arr.extend(frame_links)

    tracklets = generate_tracklet(frame_link_arr, anim_length, marker_num)
    res_label = np.ones((anim_length, marker_num), dtype=int) * -1

    for tracklet in tracklets:
        confidence = np.stack([label_confidence[idx[0], idx[1]] for idx in tracklet])

        if args.score_p == 0:
            label_id = np.argmax(np.bincount(np.argmax(confidence, axis=1)))
        else:
            score = np.linalg.norm(confidence, axis=0, ord=args.score_p)
            label_id = np.argmax(score)

        for pos in tracklet:
            res_label[pos[0], pos[1]] = label_id
    return res_label
