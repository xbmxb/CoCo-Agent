from torch.utils.data import Dataset
import torch
import pickle
from tqdm import tqdm
import action_type
import numpy as np
import jax.numpy as jnp
import random
import argparse, jsonlines, os, json

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (577, 768),
    "vit-large": (145, 1024),
    "vit-global": (1, 768),
    "vit-merge": (578, 768),
}

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='dataset/owl/general_parsed_episode_owl')
#     parser.add_argument('--output_dir', type=str, default='experiments')
#     parser.add_argument('--model', type=str, default='google/flan-t5-base')
#     parser.add_argument('--data_ratio', type=float, default=None)
#     parser.add_argument('--eval_name', type=str, default=None, help='the saved subset name used for evaluation')
#     parser.add_argument('--local_rank', type=int, default=-1)
#     parser.add_argument('--epoch', type=int, default=20)
#     parser.add_argument('--lr', type=float, default=5e-5)
#     parser.add_argument('--warmup_ratio', type=float, default=0.1)
#     parser.add_argument('--bs', type=int, default=2)
#     parser.add_argument('--debug_num', type=int, default=2)
#     parser.add_argument('--input_len', type=int, default=512)
#     parser.add_argument('--output_len', type=int, default=128)
#     parser.add_argument('--img_dim', type=int, default=512)
#     parser.add_argument('--eval_bs', type=int, default=16)
#     parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
#     parser.add_argument('--all_data', type=float, default=None, help='whether using all the data for training. Set the ratio for google apps to save computation')
#     parser.add_argument('--eval_subset', type=str, default=None, help='use which subset for evaluation/test when training with all data')
#     parser.add_argument('--use_history', type=int, default=None, help='only evaluate the model at the final epoch')
#     parser.add_argument('--use_img_history', action='store_true', help='only evaluate the model at the final epoch')
#     parser.add_argument('--use_layout', action='store_true', help='only evaluate the model at the final epoch')
#     parser.add_argument('--transform_axis', action='store_true', help='only for baseline to improve inference speed')
#     parser.add_argument('--use_generate', default=True, action='store_true', help='only for baseline to improve inference speed')
#     parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
#     parser.add_argument('--user_msg', type=str, default="debug", help='experiment type in the save_dir')
#     parser.add_argument('--img_type', type=str, default="clip", choices=['detr', 'clip', 'blip','vit','vit-large','vit-global','vit-merge'], help='type of image features')
#     parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
#     parser.add_argument('--seed', type=int, default=42, help='random seed')

#     args = parser.parse_args()
#     return args

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
    histories = []

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
        if args.use_history:
            history_action = []
            if args.use_img_history:
                # history_image = [torch.zeros(args.img_dim)] * args.use_history
                history_image = []

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
                    bottom, right = top + height, left + width
                    ui_axis = [top, left, bottom, right]
                    ui_axis = ["{:.4f}".format(axis) for axis in ui_axis]
                    ui_axis = f"({ui_axis[0]}, {ui_axis[1]}, {ui_axis[2]}, {ui_axis[3]})"
                    if ui_type_i == "TEXT":
                        icon_string += f'<p id={ui_idx} class="text" alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
                    elif "ICON" in ui_type_i:
                        icon_string += f'<img id={ui_idx} class={ui_type_i} alt="{ui_axis}">{ui_text[ui_idx]}</p>\n'
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
                        result_touch_yx = [round(axis, 4) for axis in result_touch_yx]
                        # if touching, the lift can be the same as touch
                        result_lift_yx = result_touch_yx
                    else:
                        drags_match = _check_drag_actions_match(
                            action_touch_yx, action_lift_yx
                        )
                        result_touch_yx, result_lift_yx = scroll_map[drags_match]

            target_action = f'"action_type": "{result_action}", "touch_point": "{result_touch_yx}", "lift_point": "{result_lift_yx}", "typed_text": "{result_text}"'
            
            if args.use_history:
                prev_actions = "\n".join(history_action)
                question = f"Previous Actions: {prev_actions}\n{question}"
                if args.use_img_history:
                    image = history_image + [image]
                    # image = torch.stack(image)
            
            history = []
            for i in range(len(history_action)):
                history.append( (history_image[i], history_action[i]) )
            
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
            histories.append(history)

            if args.use_history:
                history_action.append(target_action)
                if args.use_img_history:
                    history_image.append(image[-1])
                    
                if len(history_action) > args.use_history:
                    history_action.pop(0)
                    history_image.pop(0)
                        

        if args.debug_num:
            if int(qid) > args.debug_num:
                break
    cid_f.close()
    center_f.close()
    return source_text, source_image, target_text, anno_positions, goals, histories

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

# class ScienceQADatasetImg(Dataset):
#     """
#     Creating a custom dataset for reading the dataset and
#     loading it into the dataloader to pass it to the
#     neural network for finetuning the model

#     """

#     def __init__(
#         self, data, tokenizer, source_len, target_len
#     ):
#         """
#         Initializes a Dataset class

#         Args:
#             dataframe (pandas.DataFrame): Input dataframe
#             tokenizer (transformers.tokenizer): Transformers tokenizer
#             source_len (int): Max length of source text
#             target_len (int): Max length of target text
#             source_text (str): column name of source text
#             target_text (str): column name of target text
#         """
#         self.tokenizer = tokenizer
#         self.source_len = source_len
#         self.summ_len = target_len
#         self.source_text = data[0]
#         self.source_image = data[1]
#         self.target_text = data[2]
#         self.anno_positions = data[3]
            
#     def __len__(self):
#         """returns the length of dataframe"""
#         return len(self.target_text)

#     def __getitem__(self, index):
#         """return the input ids, attention masks and target ids"""

#         source_text = str(self.source_text[index])
#         source_image = self.source_image[index]
#         target_text = str(self.target_text[index])

        
#         target_dict = eval("{" + target_text + "}")
#         action = action_type.ActionType[target_dict["action_type"]].value

#         touch_point = eval(target_dict["touch_point"])
#         lift_point = eval(target_dict["lift_point"])

#         # cleaning data so as to ensure data is in string type
#         source_text = " ".join(source_text.split())
#         target_text = " ".join(target_text.split())

#         source = self.tokenizer.batch_encode_plus(
#             [source_text],
#             max_length=self.source_len,
#             pad_to_max_length=True,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         target = self.tokenizer.batch_encode_plus(
#             [target_text],
#             max_length=self.summ_len,
#             pad_to_max_length=True,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )

#         source_ids = source["input_ids"].squeeze()
#         source_mask = source["attention_mask"].squeeze()
#         target_ids = target["input_ids"].squeeze()
        
#         image_ids = torch.tensor(source_image).squeeze()
#         vis_attention_mask = torch.tensor([1]).squeeze()

#         act_ids = torch.tensor(action).squeeze()
#         touch_point = torch.tensor(touch_point).squeeze()
#         lift_point = torch.tensor(lift_point).squeeze()
        
        
#         return {
#             "input_ids": source_ids,
#             "attention_mask": source_mask,
#             "image_ids": image_ids,
#             "labels": target_ids,
#             "target_act": act_ids,
#             "target_touch": touch_point,
#             "target_lift": lift_point
#         }

def load_for_owl(inputfile, split, foreval=False, margs = None):
    # args = parse_args()
    class theargs: 
        debug_num = None
        data_ratio = None
        # use_history 12, 20, 40
        use_history = 4
        use_img_history = True
        img_dim = 512
        use_layout = False
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
    source_text, source_image, target_text, anno_positions, goals, histories = load_data(args, split)
    assert len(source_text) == len(source_image) == len(target_text) == len(anno_positions) == len(goals)
#     text_template_1 = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
# Human: Given a mobile screen and a goal, provide the action based on the screen information. Screen: <image> Goal: <goal_info> 
# AI: <label_action>'''
    text_template_1 = '''Human: <image> Goal: <goal_info> Given a mobile screen image and a goal, provide the action based on the screen information.  
AI: <label_action>'''
    text_template_pas = '''Human: <image> <Previous actions> Goal: <goal_info> Given a mobile screen image and a goal, provide the action based on the screen information.  
    AI: <label_action>'''
    text_template_pasimage = '''Human: Instruct:<goal_info>\n'''
    pasimage_list = [f'Human: Screen{i} is <image>\nAI: <p_action{i}>\n' for i in range(args.use_history)]
    # pasimage = '\n'.join(pasimage_list)
    current = '''Human: Screen<nextid> is <image>\nAI: <label_action>'''
    # text_template_pasimage_full = text_template_pasimage + pasimage + current
    
    data = []
        
    for i in range(len(source_text)):
        if args.debug_num and i > args.debug_num:
            break
        di = {
            # 'image': os.path.join('/data/maxb/mmcot2', source_image[i]),
            'image' : '',
            'text': '',
            'target_text': target_text[i],
            'anno_pos': anno_positions[i],
            "task_type": "llava_sft",
            
        }
        if args.use_history is None:
            source_texti = text_template_1.replace('<goal_info>', goals[i])
            source_texti = source_texti.replace('<label_action>', target_text[i])
            di['text'] = source_texti
        elif args.use_img_history is None:
            source_texti = text_template_pas.replace('<goal_info>', goals[i])
            source_texti = source_texti.replace('<label_action>', target_text[i])
            pas = source_text[i].split("\nGoal:")[0]
            source_texti = source_texti.replace('<Previous actions>', pas)
            di['text'] = source_texti
            # print(source_texti)
            # return
        else: #use action&img history
            source_texti = text_template_pasimage.replace('<goal_info>', goals[i])
            current_i = current.replace('<label_action>', target_text[i])
            current_i = current_i.replace('<nextid>', str(len(source_image[i])-1))
            for j in range(len(histories[i])):
                # source_texti = source_texti.replace(f'<p_image{i}>', histories[i][1])
                source_texti += pasimage_list[j].replace(f'<p_action{j}>', histories[i][j][1])
            source_texti += current_i
            
            if 'Human: Screen0' in source_texti:
                source_texti = source_texti.replace('\nHuman: Screen0', ' Screen0')
                
            di['text'] = source_texti
            assert len(source_image[i]) == len(histories[i]) + 1 # p_image + image
            di['image'] = [os.path.join('/data/maxb/mmcot2', source_image[i][j]) for j in range(len(source_image[i]))]
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