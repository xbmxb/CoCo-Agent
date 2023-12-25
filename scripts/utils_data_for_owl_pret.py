from torch.utils.data import Dataset
import torch
import pickle
from tqdm import tqdm
import action_type
import numpy as np
import jax.numpy as jnp
import random
import argparse, jsonlines, os, json
from action_matching import is_tap_action, _resize_annotation_bounding_boxes, _yx_in_bounding_boxes

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (577, 768),
    "vit-large": (145, 1024),
    "vit-global": (1, 768),
    "vit-merge": (578, 768),
}

def _check_tap_actions_match(
    tap_yx,
    annotation_positions,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
    """Determines if two tap actions are the same."""
    resized_annotation_positions = _resize_annotation_bounding_boxes(
        annotation_positions,
        annotation_width_augment_fraction,
        annotation_height_augment_fraction,
    )
    # Check if the ground truth tap action falls in an annotation's bounding box.
    tap_in_box = _yx_in_bounding_boxes(tap_yx, resized_annotation_positions)
    return tap_in_box

def load_data(args, split):
    clusters_root = '/data/maxb/mmcot2/dataset/owl/general_longtermv2gpt/'
    clusters_id = '/data/maxb/mmcot2/dataset/owl/general_longtermv2gpt/clusters.json'
    centers = '/data/maxb/mmcot2/dataset/owl/general_longtermv2gpt/reps_loc.json'
    cid_f = open(clusters_id, 'r')
    cid = json.load(cid_f)
    center_f = open(centers, 'r')
    cter = json.load(center_f)
    
    target_text = []
    source_text = []
    source_image = []
    anno_positions = []
    goals = []
    element_ids = []
    element_texts = []
    icon_strings = []
    if args.all_data:
        if split == "train":
            data = []
            for subdir in ["general", "google_apps", "install", "single", "web_shopping"]:
                print(f"loading {subdir}", len(data))
                with open(f"{args.data_path}/{args.data_root}_{split}.obj", "rb") as rp:
                    sub_data = pickle.load(rp)
                if subdir == "google_apps":
                    sub_data = random.sample(sub_data, int(len(sub_data) * args.all_data))
                data.extend(sub_data)
        else:
            # we use general subset for dev/test
            with open(f"{args.eval_subset}_{split}.obj", "rb") as rp:
                    data = pickle.load(rp)
    else:
        print(f"Loading {args.data_path}/{args.data_root}_{split}.obj")
        with open(f"{args.data_path}/{args.data_root}_{split}.obj", "rb") as rp:
            data = pickle.load(rp)
            if args.data_ratio:
                data = random.sample(data, int(len(data) * args.data_ratio))

    for qid, episode in enumerate(tqdm(data)):
        episode_id = episode["episode_id"]
        episode_data = episode["data"]

        for step_idx, step_data in enumerate(episode_data):

            question = step_data["goal"]
            question = f"Goal: {question}"

            image = step_data["image"]

            ui_positions = step_data["ui_positions"]
            ui_text = step_data["ui_text"]
            ui_type = step_data["ui_type"]
            if args.use_layout:
                icon_string = ""
                for ui_idx, ui_type_i in enumerate(ui_type):
                    ui_axis = ui_positions[ui_idx]
                    top, left, height, width = ui_axis
                    # The y-axis is inverted for AndroidEnv, so bottom = top + height.
                    # bottom, right = top + height, left + width
                    # ui_axis = [top, left, bottom, right]
                    # ui_axis = ["{:.4f}".format(axis) for axis in ui_axis]
                    # ui_axis = f"({ui_axis[0]}, {ui_axis[1]}, {ui_axis[2]}, {ui_axis[3]})"
                    # if ui_type_i == "TEXT":
                    #     icon_string += f'<p id={ui_idx} class="text" alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                    # elif "ICON" in ui_type_i:
                    #     icon_string += f'<img id={ui_idx} class={ui_type_i} alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                    # else:
                    #     print(icon_string)
                    #     assert "parsing ui failed!!!"
                    
                    golden_location = [top + height/2, left + width/2]
                    golden_location = ["{:.4f}".format(g) for g in golden_location]
                    golden_location = f"[{golden_location[0]}, {golden_location[1]}]"
                    if ui_type[ui_idx] == "TEXT":
                        icon_string += f"{ui_text[ui_idx]} location: {golden_location}\n"
                    elif "ICON" in ui_type[ui_idx]:
                        icon_string += f"{ui_type[ui_idx]} location: {golden_location}\n"
                    else:
                        print(icon_string)
                        assert "parsing ui failed!!!"
                # finish the layout icon_string
            source_image.append(image)
            icon_strings.append(icon_string)
        if args.debug_num:
            if int(qid) > args.debug_num:
                break
    cid_f.close()
    center_f.close()
    return source_image, icon_strings

_SWIPE_DISTANCE_THRESHOLD = 0.04
def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = jnp.linalg.norm(
        jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD

def _check_drag_actions_match(
    drag_touch_yx,
    drag_lift_yx,
):
    """Determines if two drag actions are the same."""
    # Store drag deltas (the change in the y and x coordinates from touch to
    # lift), magnitudes, and the index of the main axis, which is the axis with
    # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
    # ending at (0.3, 0.5) has a main axis index of 1).
    drag_1_deltas = drag_lift_yx - drag_touch_yx
    drag_1_magnitudes = jnp.abs(drag_1_deltas)
    drag_1_main_axis = np.argmax(drag_1_magnitudes)

    # y axis
    if drag_1_main_axis == 0:
        if drag_1_deltas[0] < 0:
            scroll = "up"
        else:
            scroll = "down"
    elif drag_1_main_axis == 1:
        if drag_1_deltas[1] < 0:
            scroll = "left"
        else:
            scroll = "right"
            
    return scroll


def load_for_owl(inputfile, split, foreval=False, margs = None):
    # args = parse_args()
    class theargs: 
        debug_num = None
        data_ratio = None
        # use_history 12, 20, 40
        use_history = None
        use_img_history = False
        img_dim = 512
        use_layout = True
        data_root = 'general_parsed_episode_owl'
        data_path = '/data/maxb/mmcot2/dataset/owl'
        eval_subset = '/data/maxb/mmcot2/dataset/owl/general_parsed_episode_owl/'
        all_data = None
        transform_axis = True
        # use memory: center_type, center_full, highlevel, retrieve, or None
        # memory = 'highlevel'
        memory = None
    args = theargs()
    if margs:
        args.debug_num = margs.debug_num if margs.debug_num else args.debug_num
        args.data_ratio = margs.data_ratio if  margs.data_ratio else args.data_ratio
        args.use_history = margs.use_history if margs.use_history else args.use_history
        args.use_img_history = margs.use_img_history if margs.use_img_history else args.use_img_history
        args.img_dim = margs.img_dim if margs.img_dim else args.img_dim
        args.use_layout = margs.use_layout if margs.use_layout else args.use_layout
        args.data_root = margs.data_categ if margs.data_categ else args.data_categ
        args.eval_subset = margs.eval_subset if margs.eval_subset else args.eval_subset
        args.all_data = margs.all_data if margs.all_data else args.all_data
        args.transform_axis = margs.transform_axis if margs.transform_axis else args.transform_axis
        args.data_path = margs.data_path if margs.data_path else args.data_path

    print("args",args)
    source_image, icon_strings = load_data(args, split)
    # assert len(source_text) == len(source_image) == len(target_text) == len(anno_positions) == len(goals)
    
#     text_template_1 = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# Human: Given a mobile screen and a goal, provide the action based on the screen information. Screen: <image> Goal: <goal_info> 
# AI: <label_action>'''
    text_template_1 = '''Human: <image> <layout> Goal: <goal_info> Given a mobile screen image and a goal, provide the action based on the screen information.  
AI: <label_action>'''
    text_template_1 = '''Human: <image> <layout> Goal: <goal_info> Predict the next action.
AI: <label_action>'''
    text_template_pas = '''Human: <image> <Previous actions> Goal: <goal_info> Given a mobile screen image and a goal, provide the action based on the screen information.  
    AI: <label_action>'''
    text_template_pret_src = '''Human: <image> I need to <CLICK> <ele_name>AI: I need to <CLICK> <ele_name>, the location is <location>'''
    text_template_pret_tgt = '''I need to <CLICK> <ele_name>, the location is <location>'''
    text_template_pret_src = '''Human: <image> "<ele_name>" location is AI: <location>'''
    text_template_pret_tgt = '''<location>'''
    # text_template_pasimage = '''Human: Instruct: <goal_info>\n'''
    # pasimage = '\n'.join([ 'Human: Screen: <p_image>\nAI:<p_action>'] * args.use_history)
    # current = '''\nHuman: Screen: <p_image>\nAI: <label_action>'''
    # text_template_pasimage_full = text_template_pasimage + pasimage + current
    scroll_map = {
        "up": [[0.8, 0.5], [0.2, 0.5]],
        "down": [[0.2, 0.5], [0.8, 0.5]],
        "left": [[0.5, 0.8], [0.5, 0.2]],
        "right": [[0.5, 0.2], [0.5, 0.8]]
    }
    data = []
    
    for i in range(len(icon_strings)):
        if args.debug_num and i > args.debug_num:
            break
        icon_string_list = icon_strings[i].strip().split('\n')
        for j in range(len(icon_string_list)):
            if len(icon_string_list[j]):
                ele_name, loc = icon_string_list[j].split(' location: ')
                src = text_template_pret_src.replace('<ele_name>', ele_name)
                src = src.replace('<location>', loc)
                tgt = text_template_pret_tgt.replace('<ele_name>', ele_name)
                tgt = tgt.replace('<location>', loc)
                di = {
                    'image': os.path.join('/data/maxb/mmcot2', source_image[i]),
                    'target_text': tgt,
                    'text': src,
                    "task_type": "llava_sft",  
                }
                data.append(di)
    
    if not foreval:
        return data
    # elif args.debug_num:
    #     return anno_positions[:args.debug_num]
    # else:
    #     return anno_positions
    else:
        return 