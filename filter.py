import argparse
import cv2
import time
import pickle
import numpy as np
import math

basket_width = 70
basket_height = 50

def init_video():
    # Video reader
    video_in = cv2.VideoCapture(args.video)
    width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    video_out = cv2.VideoWriter(video_out_filename, cv2.VideoWriter_fourcc(*'mp4v'), args.out_fps, (width, height))
    return video_in, video_out, width, height


def get_orig_image(previous_frame_id, frame_id):
    for i in range(previous_frame_id + 1, frame_id):
        ret, image = cam_in.read()
        #cam_out.write(image)

    return cam_in.read()


def draw_bbox(input_image, bbox, color=None):
    (x1, y1), (x2, y2) = bbox[:4]
    cv2.rectangle(input_image, (x1, y1), (x2, y2), color, 3)
    return input_image


def fit_missing_point(left, right, num_missing, position):
    l = np.array(left)
    r = np.array(right)
    missing = l + position * (r - l) / (num_missing + 1)
    return int(missing[0]), int(missing[1])


# Fills baskets in frames that don't have a basket
# Returns True if basket is on the left, else false if it is on the right
def smooth_baskets(baskets):
    frame_ids = sorted(baskets.keys())
    missing_frames = {}
    previous_frame_id = frame_ids[0]
    left_count = 0  # How many times did we see the basket on left half of the frame
    for frame in frame_ids[1:]:
        if baskets[frame][0][0] < im_width / 2:
            left_count += 1
        step = int((frame - previous_frame_id) / frame_ratio)
        for i in range(step-1):
            missing_frames[previous_frame_id+frame_ratio*(i+1)] = \
                fit_missing_point(baskets[frame][0], baskets[previous_frame_id][0], step-1, i+1), \
                fit_missing_point(baskets[frame][1], baskets[previous_frame_id][1], step-1, i+1)
        previous_frame_id = frame

    baskets.update(missing_frames)
    return left_count > len(frame_ids) / 2


def get_court_xy_limits(basket):
    x = 0
    y = (max(0, basket[1] - args.basket_vspace), im_height)
    if left_basket:
        x = (max(0, basket[0] - args.basket_hspace), im_width)
    else:
        x = (0, min(im_width, basket[0] + basket_width + args.basket_hspace))
    return x, y


# Returns true if point is in bbox. It expects x0 < x1 and y0 < y1
def point_in_limits(point, x, y):
    return x[0] <= point[0] <= x[1] and y[0] <= point[1] <= y[1]


# Returns a tuple x, where:
# x[0] = y location of top joint
# x[1] = True if arms are raised
# x[2] = True if pose is sitting
def analyze_pose(pose):
    max_pose_point = im_height
    max_arm_angle = -90
    for limb_id, limb in pose.items():
        max_pose_point = min(max_pose_point, limb['from'][1], limb['to'][1])
        if limb_id in ("rlowerarm", "llowerarm", "rupperarm", "lupperarm"):
            max_arm_angle = max(max_arm_angle, limb['ang'])
    return max_pose_point, max_arm_angle > 0, False


def is_player(pose, x_court_limits, y_court_limits):
    top_joint_y, arm_is_up, is_sitting = analyze_pose(pose)

    if is_sitting:
        return False
    elif top_joint_y > im_height / 2:  # if pose in lower half of the screen, look for missing legs
        return "rupperleg" in pose or "rlowerleg" in pose or "lupperleg" in pose or "llowerleg" in pose
    else:  # if pose in upper half of screen, arms should be raised or top joint needs to be below basket
        return arm_is_up or point_in_limits((x_court_limits[0], top_joint_y), x_court_limits, y_court_limits)


def add_person_joints(person, joint_matrix):
    for limb in person.values():
        joint_matrix[limb['from'][::-1]] = 1
        joint_matrix[limb['to'][::-1]] = 1


def get_frame_joint_matrix(poses, basket):
    x_limits, y_limits = get_court_xy_limits(basket)
    player_joint_matrix = np.zeros((im_height, im_width))
    audience_joint_matrix = np.zeros((im_height, im_width))

    # Add all player joints and blacklist audience joints
    for person in poses['people']:
        if is_player(person, x_limits, y_limits):
            add_person_joints(person, player_joint_matrix)
        else:
            add_person_joints(person, audience_joint_matrix)

    # Add all joints that are inside court limits and not blacklisted
    for body_part_id, body_parts in poses['body_parts'].items():
        for body_part_instance in body_parts:
            joint = body_part_instance['x'], body_part_instance['y']
            if point_in_limits(joint, x_limits, y_limits) and audience_joint_matrix[joint[::-1]] == 0:
                player_joint_matrix[joint[::-1]] = 1

    return player_joint_matrix


def get_basket(frame_id):
    if frame_id in all_baskets:
        xy = all_baskets[frame_id][0]
        return xy[0], xy[1]-basket_height
    elif left_basket:
        return 0, 0
    else:
        return im_width-1, 0


def get_video_joint_matrices():
    joint_matrices = {}
    for frame_id in all_poses:
        basket = get_basket(frame_id)
        joint_matrix = get_frame_joint_matrix(all_poses[frame_id], basket)
        joint_matrices[frame_id] = joint_matrix
    return joint_matrices


def draw_frame_num(input_image, frame_id, original_frame_id):
    label = '{}{:d}/{:d}'.format("", frame_id, original_frame_id)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(input_image, (1750, 90), (1750 + t_size[0] + 3, 90 + t_size[1] + 4), [0, 0, 0], -1)
    cv2.putText(input_image, label, (1750, 90 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return input_image


def mean_filter(grid):
    weights = np.arange(filter_grid_size[1], 0, -1)
    h_sum = np.sum(grid, axis=1)
    v_sum = np.sum(h_sum)
    if v_sum == 0:
        return 0
    else:
        return int(np.dot(weights, h_sum) / v_sum)


def apply_filter(joint_matrices, filter):
    size_x = filter_grid_size[0]
    size_y = filter_grid_size[1]

    num_rows = int((im_height - size_y)/grid_resolution + 1)
    num_columns = int((im_width - size_x)/grid_resolution + 1)
    filter_matrices = np.zeros((len(all_baskets), num_rows, num_columns))

    frame_ids = sorted(all_baskets.keys())
    for i, frame_id in enumerate(frame_ids):
        joint_matrix = joint_matrices[frame_id]
        for row in range(num_rows):
            for column in range(num_columns):
                image_row = row * grid_resolution
                image_column = column * grid_resolution
                filter_matrices[i][row][column] = \
                    filter(joint_matrix[image_row: image_row+size_y, image_column: image_column+size_x])

    return filter_matrices


def shift_matrix(matrix, x, y, left):
    matrix[0:y] = 0
    matrix = np.roll(matrix, -y, axis=0)
    if left:
        matrix[:, 0:x] = 0
        matrix = np.roll(matrix, -x, axis=1)
    else:
        matrix[:, x+1:im_width] = 0
        matrix = np.roll(matrix, im_width - x - 1, axis=1)
    return matrix


# Shifts filter values in reference to the basket
def shift_filter(mean_matrices):
    frame_ids = sorted(all_baskets.keys())
    for i, frame_id in enumerate(frame_ids):
        x_limits, y_limits = get_court_xy_limits(get_basket(frame_id))
        x = x_limits[0] if left_basket else x_limits[1]
        y = y_limits[0]
        mean_matrices[i] = shift_matrix(mean_matrices[i], x, y, left_basket)


def process():
    jump_filters = {}  # frame -> jump_filter
    frame_count = len(all_poses)
    pose_frame_ids = sorted(all_poses.keys())
    previous_video_frame_id = -1  # Previous video frame processed
    out_frame_id = -1
    frame_id_offset = min(all_poses.keys())
    joint_matrices = get_video_joint_matrices()
    #shift_filter(joint_matrices)
    #mean_matrices = apply_filter(joint_matrices, mean_filter)

    #for frame_id in pose_frame_ids[lookback-1:]:
    for frame_id in pose_frame_ids:
        video_frame_id = int((frame_id - frame_id_offset) / frame_ratio)
        ret_val, orig_image = get_orig_image(previous_video_frame_id, video_frame_id)
        previous_video_frame_id = video_frame_id
        out_frame_id += 1

        joints = joint_matrices[frame_id]
        for index in np.argwhere(joints > 0):
            cv2.circle(orig_image, (index[1], index[0]), 4, [0, 255, 0], thickness=-1)

        x_limits, y_limits = get_court_xy_limits(get_basket(frame_id))
        x = x_limits[0] if left_basket else x_limits[1]
        y = y_limits[0]
        cv2.line(orig_image, (0, y), (x, y), [255, 0, 0])
        cv2.line(orig_image, (x, im_height), (x, y), [255, 0, 0])
        draw_frame_num(orig_image, out_frame_id, frame_id)
        M = np.float32([[1, 0, im_width-x], [0, 1, -y]])
        cv2.warpAffine(orig_image, M, (im_width, im_height), orig_image)
        cam_out.write(orig_image)

    return jump_filters


def print_time(label, start, end):
    print("{}: {:.2} secs".format(label, end-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basket_bbox', type=str, required=True, help='pkl file with basket bbox')
    parser.add_argument('--video', type=str, required=True, help='video clip')
    parser.add_argument('--out_fps', type=int, required=True, help='fps of output video')
    #parser.add_argument('--frame_id_offset', type=int, required=True, help='frame_id of first frame in video')
    parser.add_argument('--frame_ratio', type=int, default=3, help='frame ratio between video file & bbox pkl file')
    parser.add_argument('--pose_features', type=str, required=True, help='pose features pkl file')
    parser.add_argument('--frame_lookback', type=int, default=5, help='# frames to consider when calculating travel')
    parser.add_argument('--basket_hspace', type=int, default=140, help='poses considered upto this space beyond basket')
    parser.add_argument('--basket_vspace', type=int, default=50, help='poses considered upto this space above basket')
    parser.add_argument('--filter_grid_size', type=int, default=[240, 440], nargs=2, help='filter grid size: x y')
    parser.add_argument('--grid_resolution', type=int, default=20, help='move grid by')

    args = parser.parse_args()
    video_out_filename = args.video.rsplit(".", 1)[0].rsplit("_", 1)[0] + "_filter.mp4"
    filters_out_filename = args.video.rsplit(".", 1)[0].rsplit("_", 1)[0] + "_filter.pkl"
    lookback = args.frame_lookback
    frame_ratio = args.frame_ratio
    filter_grid_size = tuple(args.filter_grid_size)
    grid_resolution = args.grid_resolution

    # init input and output videos
    cam_in, cam_out, im_width, im_height = init_video()

    # Pickle files for pose and tracking
    all_baskets = pickle.load(open(args.basket_bbox, "rb"), fix_imports=True)
    left_basket = smooth_baskets(all_baskets)
    all_poses = pickle.load(open(args.pose_features, "rb"))

    # joint_matrix = np.zeros((5, 5))
    # joint_matrix[0] = [1, 2, 3, 4, 5]
    # joint_matrix[1] = [6, 7, 8, 9, 10]
    # joint_matrix[2] = [11, 12, 13, 14, 15]
    # joint_matrix[3] = [16, 17, 18, 19, 20]
    # joint_matrix[4] = [21, 22, 23, 24, 25]
    # #joint_matrices = {1: joint_matrix}
    # #im_height = 5
    # im_width = 5
    # shift_matrix(joint_matrix, 2, 2, False)


    jump_filters = process()

    pickle.dump(jump_filters, open(filters_out_filename, "wb"))


# BACKUP CODE - in case
#
# def add_person_joints(person, joints):
#     for limb in person.values():
#         joints.add(limb['from'])
#         joints.add(limb['to'])
#
#
# def get_player_joints(poses, basket):
#     x_limits, y_limits = get_court_xy_limits(basket)
#
#     # Add all player joints and blacklist audience joints
#     audience_joints = set()
#     player_joints = set()
#     for person in poses['people']:
#         if is_player(person, x_limits, y_limits):
#             add_person_joints(person, player_joints)
#         else:
#             add_person_joints(person, audience_joints)
#
#     # Add all joints that are inside court limits and not blacklisted
#     for body_part_id, body_parts in poses['body_parts'].items():
#         for body_part_instance in body_parts:
#             joint = body_part_instance['x'], body_part_instance['y']
#             if point_in_limits(joint, x_limits, y_limits) and joint not in audience_joints:
#                 player_joints.add(joint)
#
#     return player_joints


# def get_court_xy_limits(basket):
#     x = 0
#     y = (max(0, min(basket[0][1], basket[1][1]) - args.basket_vspace), im_height)
#     if left_basket:
#         x = (max(0, min(basket[0][0], basket[1][0]) - args.basket_hspace), im_width)
#     else:
#         x = (0, min(im_width, max(basket[0][0], basket[1][0]) + args.basket_hspace))
#     return x, y
