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
    prev_actions_all = []
    target_text_cot = []
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
        # if qid == 32:
        #     print(qid)
        episode_data = episode["data"]
        if args.use_history:
            history_action = []
            if args.use_img_history:
                history_image = [torch.zeros(args.img_dim)] * args.use_history

        for step_idx, step_data in enumerate(episode_data):
            # if "4896994614189692781" in step_data["image"]:
            #     print('4896994614189692781')
            question = step_data["goal"]
            question = f"Goal: {question}"

            image = step_data["image"]

            ui_positions = step_data["ui_positions"]
            ui_text = step_data["ui_text"]
            ui_type = step_data["ui_type"]
            element_id = [-2]
            element_text = []
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
                question = f"{question}\nScreen: {icon_string}"
                # print(question)
            result_touch_yx = step_data["result_touch_yx"]
            result_lift_yx = step_data["result_lift_yx"]
            result_action = step_data["result_action"][0]
            result_text = step_data["result_action"][1]

            result_text = result_text.replace("\\", "").replace('"','').replace("'","")

            if args.transform_axis:
                scroll_map = {
                    "up": [[0.8000, 0.5000], [0.2000, 0.5000]],
                    "down": [[0.2000, 0.5000], [0.8000, 0.5000]],
                    "left": [[0.5000, 0.8000], [0.5000, 0.2000]],
                    "right": [[0.5000, 0.2000], [0.5000, 0.8000]]
                }
                action_touch_yx = jnp.asarray(result_touch_yx)
                action_lift_yx = jnp.asarray(result_lift_yx)
                if result_action == "DUAL_POINT":
                    if is_tap_action(action_touch_yx, action_lift_yx):
                        # find the element id of click action
                        taps_match = _check_tap_actions_match(
                            action_touch_yx,
                            ui_positions,
                            1.4,
                            1.4,
                        )
                        taps_match_idx = np.argwhere(taps_match == True)[:,0]
                        if len(taps_match_idx) <= 0:
                            # there is no match in the UI annotation
                            element_id = [-1]
                        else:
                            taps_match_idx = taps_match_idx.tolist()
                            element_id = taps_match_idx
                        # element_id -> element_text
                        
                        if element_id[0] >= 0:
                            for ele in element_id:
                                # print(ele)
                                if ui_type[ele] == "TEXT":
                                    element_text.append(ui_text[ele])
                                elif "ICON" in ui_type[ele]:
                                    element_text.append(ui_type[ele])
                        elif element_id == [-1]:
                            element_text.append('screen')
                        result_touch_yx = [round(axis, 4) for axis in result_touch_yx]
                        # if touching, the lift can be the same as touch
                        result_lift_yx = result_touch_yx
                    else:
                        drags_match = _check_drag_actions_match(
                            action_touch_yx, action_lift_yx
                        )
                        result_touch_yx, result_lift_yx = scroll_map[drags_match]

            target_action = f'"action_type": "{result_action}", "touch_point": "{result_touch_yx}", "lift_point": "{result_lift_yx}", "typed_text": "{result_text}"'
            target_action_cot = parsing_to_cot(eval("{" + target_action + "}"), element_text, element_id, icon_string, scroll_map)
            
            if args.use_history:
                prev_actions = "\n".join(history_action)
                # question = f"Previous Actions: {prev_actions}\n{question}"
                prev_actions_all.append([h for h in history_action])
                if args.use_img_history:
                    image = history_image + [image]
                    image = torch.stack(image)
            
            if args.memory:
                if args.memory == 'highlevel':
                    cid_flag = None
                    for i in range(len(cid.keys())):
                        if step_data["goal"].strip() == cid['cls'+str(i)]['goals'][0].strip():
                            cid_flag = i
                    memory = []
                    memo = []
                    with open(os.path.join(clusters_root, str(cid_flag)+'.txt'), 'r') as f:
                        flag = 0
                        for line in f:
                            if line.strip() == '=============':
                                flag = len(memo)
                            memo.append(line.strip())
                            if line.startswith('Action List') and flag > 0:
                                break
                    memory = 'For the goal of ' + step_data["goal"] + ', there are some experience:'
                    memory += '\n'.join(memo[flag+1:])
                elif args.memory.startswith('center'):
                    cid_flag = None
                    for i in range(len(cter.keys())):
                        if step_data["goal"].strip() == cter['cls'+str(i)]['goal'].strip():
                            cid_flag = i
                            rep = cter['cls'+str(i)]['center']
                    memory = 'For the goal of ' + step_data["goal"] + ', there is an action list example: '
                    if args.memory == 'center_type':
                        acttype_list = [i['result_action'][0] for i in rep] # 0 for type  1 for typed text
                        memory += ' '.join(acttype_list)
                    elif args.memory == 'center_full':
                        actfull_list = []
                        # parse action
                        for ri in rep:
                            result_action = ri['result_action']
                            result_text = result_action[1]
                            result_action = result_action[0]
                            result_text = result_text.replace("\\", "").replace('"','').replace("'","")
                            result_touch_yx = ri["result_touch_yx"]
                            result_lift_yx = ri["result_lift_yx"]
                            if args.transform_axis:
                                scroll_map = {
                                    "up": [[0.8000, 0.5000], [0.2000, 0.5000]],
                                    "down": [[0.2000, 0.5000], [0.8000, 0.5000]],
                                    "left": [[0.5000, 0.8000], [0.5000, 0.2000]],
                                    "right": [[0.5000, 0.2000], [0.5000, 0.8000]]
                                }
                                action_touch_yx = jnp.asarray(result_touch_yx)
                                action_lift_yx = jnp.asarray(result_lift_yx)
                                if result_action == "DUAL_POINT":
                                    if is_tap_action(action_touch_yx, action_lift_yx):
                                        result_touch_yx = [round(axis, 4) for axis in result_touch_yx]
                                        # if touching, the lift can be the same as touch
                                        result_lift_yx = result_touch_yx                        
                                    else:
                                        drags_match = _check_drag_actions_match(
                                            action_touch_yx, action_lift_yx
                                        )
                                        result_touch_yx, result_lift_yx = scroll_map[drags_match]
                            riact =  f'"action_type": "{result_action}", "touch_point": "{result_touch_yx}", "lift_point": "{result_lift_yx}", "typed_text": "{result_text}"'
                            actfull_list.append(riact)
                        memory += ' '.join(actfull_list)
                elif args.memory == 'retrieve':
                    pass
                    # jupiter done
                else:
                    raise NotImplementedError
                # add memory to question
                question = f"{memory}\n{question}"
                
            source_text.append(question)
            source_image.append(image)
            target_text.append(target_action)
            anno_positions.append(ui_positions)
            goals.append(step_data["goal"])
            element_ids.append(element_id) # -2: other action; -1:invalid(screen) 
            element_texts.append(element_text) # []: other action or invalid; [xxx] 
            icon_strings.append(icon_string)
            target_text_cot.append(target_action_cot)

            if args.use_history:
                history_action.append(target_action_cot)
                if args.use_img_history:
                    history_image.append(image[-1])
                    history_image.pop(0)
                if len(history_action) > args.use_history:
                    history_action.pop(0)
                        

        if args.debug_num:
            if int(qid) > args.debug_num:
                break
    cid_f.close()
    center_f.close()
    return source_text, source_image, target_text, anno_positions, goals, element_ids, element_texts, icon_strings, prev_actions_all, target_text_cot

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

def parsing_to_cot(pred, element_text, element_id, icon_string, scroll_map):
    act = pred['action_type']
    # print(act)
    if act == 'TYPE':
        text = pred['typed_text']
        target_cot = f'I need to <TYPE> a string here, \"typed_text\": \"{text}\"'
    elif act == 'DUAL_POINT':
        tp = pred['touch_point']
        lp = pred['lift_point']
        if is_tap_action(eval(tp), eval(lp)):
            element = element_text # TODO
            # print(element, element_id)
            if element == ['screen']:
                target_cot = f'I need to <TAP> on the screen, the location is \"tap_point\": \"{str(tp)}\"'
            elif len(element) > 0:
                # change the touch&lift point to the golden label
                one_ele = element[0]
                icon_str = icon_string.split('\n')[element_id[0]]
                # print(icon_str)
                ele_name, point = icon_str.split(' location: ')
                assert ele_name == one_ele
                target_cot = f'I need to <CLICK> {one_ele}, the location of {one_ele} on the screen is \"tap_point\": \"{point}\"'
                # target_cot = f'I need to <CLICK> {element[0]}, the location of {element[0]} on the screen is \"tap_point\": \"{str(tp)}\"'
        else: # scroll
            direct = None
            for di in scroll_map.keys():
                if scroll_map[di] == [eval(tp), eval(lp)]:
                    direct = di
            target_cot = f'I need to <SCROLL> {direct}, so \"touch_point\": \"{str(tp)}\", \"lift_point\": \"{str(lp)}\"'
            # print(target_cot)
    elif 'TASK' not in act:
        target_cot = f'I need to <{act}>'
    else:
        target_cot = f'For this goal, no more actions is needed, so <{act}>'    
    return target_cot

def load_for_owl(inputfile, split, foreval=False, margs = None):
    # args = parse_args()
    class theargs: 
        debug_num = None
        data_ratio = None
        # use_history 12, 20, 40
        use_history = 8
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
    source_text, source_image, target_text, anno_positions, goals, element_ids, element_texts, icon_strings, prev_actions_all, target_text_cot = load_data(args, split)
    assert len(source_text) == len(source_image) == len(target_text) == len(anno_positions) == len(goals)
    # look at the click action:  total click, invalid(screen), mulitple, single icon
    click_stat = [0,0,0,0] 
    for e in element_ids:
        if e != [-2]:
            click_stat[0] += 1
            if e == [-1]:
                click_stat[1] += 1
            elif len(e) > 1:
                click_stat[2] += 1
            elif len(e) == 1 and e[0] != -1:
                click_stat[3] += 1
    print(click_stat)
#     text_template_1 = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# Human: Given a mobile screen and a goal, provide the action based on the screen information. Screen: <image> Goal: <goal_info> 
# AI: <label_action>'''
    text_template_1 = '''Human: <image> <layout> Goal: <goal_info> Given a mobile screen image and a goal, provide the action based on the screen information.  
AI: <label_action>'''
    text_template_1 = '''Human: <image> <layout> Goal: <goal_info> Predict the next action.
AI: <label_action>'''
    text_template_pas = '''Human: <image> <Previous actions> Goal: <goal_info> Given a mobile screen image and a goal, provide the action based on the screen information.  
    AI: <label_action>'''
    text_template_pas = '''Human: <image> <layout>\nPrevious Actions:<Previous actions> Goal: <goal_info> Next action:
AI: <label_action>'''
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
        
    for i in range(len(source_text)):
        if args.debug_num and i > args.debug_num:
            break
        # process the target text i  to be in the form of CoT
        pred = eval("{" + target_text[i] + "}")
        
        target_cot = target_text_cot[i]
        
        
        di = {
            'image': os.path.join('/data/maxb/mmcot2', source_image[i]),
            'target_text': target_cot,
            'target_text_orig': target_text[i],
            'anno_pos': anno_positions[i],
            "task_type": "llava_sft",
            
        }
        if args.use_history is None:
            source_texti = text_template_1.replace('<goal_info>', goals[i])
            source_texti = source_texti.replace('<label_action>', target_cot)
            source_texti = source_texti.replace('<layout>', icon_strings[i])
            di['text'] = source_texti
        else:
            source_texti = text_template_pas.replace('<goal_info>', goals[i])
            source_texti = source_texti.replace('<label_action>', target_cot)
            source_texti = source_texti.replace('<layout>', icon_strings[i])
            # pas = source_text[i].split("\nGoal:")[0]
            pas = prev_actions_all[i]
            pas_ = ''
            for pa in pas:
                # pas_  += '<'+pa.split('<')[-1].split('>')[0]+'>'
                pas_  += pa+'\n'
            source_texti = source_texti.replace('<Previous actions>', pas_)
            di['text'] = source_texti
            # print(source_texti)
            # return
        data.append(di)
        # print(di)
    # with jsonlines.open('/data/maxb/mmcot2/dataset/owl/general_parsed_episode_owl_train.jsonl', 'w') as f:
    #     for line in data:
    #         f.write(line)
    if not foreval:
        return data
    elif args.debug_num:
        return anno_positions[:args.debug_num]
    else:
        return anno_positions