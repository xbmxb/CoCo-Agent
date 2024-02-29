import sys
sys.path.append('./google-research')
from android_in_the_wild import visualization_utils, action_type, action_matching

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import jax.numpy as jnp
import argparse
import pickle
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import tensorflow as tf
from PIL import Image
from transformers import AutoProcessor, Blip2Model
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Get the list of available physical devices
# physical_devices = tf.config.list_physical_devices('GPU')
# # Disable GPU support by setting the visible devices to only include the CPU
# tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# dataset_name = 'general'  #@param ["general", "google_apps", "install", "single", "web_shopping"]
# data_split = "general_texts_splits.json"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
# model.to(device)
# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

dataset_directories = {
    'general': 'gs://gresearch/android-in-the-wild/general/*',
    'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
    'install': 'gs://gresearch/android-in-the-wild/install/*',
    'single': 'gs://gresearch/android-in-the-wild/single/*',
    'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
}

def _decode_image(
    example,
    image_height,
    image_width,
    image_channels,
):
    """Decodes image from example and reshapes.

    Args:
        example: Example which contains encoded image.
        image_height: The height of the raw image.
        image_width: The width of the raw image.
        image_channels: The number of channels in the raw image.

    Returns:
        Decoded and reshaped image tensor.
    """
    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )

    height = tf.cast(image_height, tf.int32)
    width = tf.cast(image_width, tf.int32)
    n_channels = tf.cast(image_channels, tf.int32)

    return tf.reshape(image, (height, width, n_channels))

def parse_episode(
    episode,
    ep_id,
    get_images = False,
    get_annotations = False,
    get_actions = False,
    output_dir = '.'
):
    parsed_episode = []
    for i, ex in enumerate(episode):
        goal = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
        step_id = ex.features.feature['step_id'].int64_list.value[0]
        # episode_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        output_ep = {
            "goal": goal,
            "step_id": step_id
        }

        image_height = ex.features.feature['image/height'].int64_list.value[0]
        image_width = ex.features.feature['image/width'].int64_list.value[0]
        image_channels = ex.features.feature['image/channels'].int64_list.value[0]
        if get_images:
            # image = _decode_image(ex, image_height, image_width, image_channels)
            # image = image.numpy()
            # image = Image.fromarray(image).convert('RGB')

            # with torch.no_grad():
            #     inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
            #     image_features = model.get_image_features(**inputs).pooler_output[0]
            #     image_features = image_features.detach().cpu()
            # output_ep["image"] = image_features
            image = _decode_image(ex, image_height, image_width, image_channels)
            image = image.numpy()
            image = Image.fromarray(image).convert('RGB')
            # print(output_dir, ep_id, step_id)
            image_path = os.path.join(output_dir, str(ep_id) +'/'+str(step_id)+'.png')
            if not os.path.exists(os.path.join(output_dir, str(ep_id))):
                os.mkdir(os.path.join(output_dir, str(ep_id)))
            image.save(image_path)
            output_ep["image"] = image_path

        if get_annotations:
            flattened_positions = np.array(
            ex.features.feature['image/ui_annotations_positions'].float_list.value
            )
            ui_text = ex.features.feature['image/ui_annotations_text'].bytes_list.value
            ui_text = [value.decode('utf-8') for value in ui_text]
            ui_type = ex.features.feature['image/ui_annotations_ui_types'].bytes_list.value
            ui_type = [value.decode('utf-8') for value in ui_type]

            positions = np.reshape(flattened_positions, (-1, 4)) #(y, x, height, width)
            output_ep["ui_positions"] = positions
            output_ep["ui_text"] = ui_text
            output_ep["ui_type"] = ui_type
        
        if get_actions:
            touch_y, touch_x = ex.features.feature['results/yx_touch'].float_list.value
            lift_y, lift_x = ex.features.feature['results/yx_lift'].float_list.value
            ex_action_type = ex.features.feature['results/action_type'].int64_list.value[0]

            ex_action_type = action_type.ActionType(ex_action_type).name

            type_text = (ex.features.feature['results/type_action'].bytes_list.value[0].decode('utf-8'))
            
            output_ep["result_touch_yx"] = [touch_y, touch_x]
            output_ep["result_lift_yx"] = [lift_y, lift_x]
            output_ep["result_action"] = [ex_action_type, type_text]

        parsed_episode.append(output_ep)
    return parsed_episode

_SWIPE_DISTANCE_THRESHOLD = 0.04
def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = jnp.linalg.norm(
        jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD

def _check_tap_actions_match(
    tap_yx,
    annotation_positions,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
    """Determines if two tap actions are the same."""
    resized_annotation_positions = action_matching._resize_annotation_bounding_boxes(
        annotation_positions,
        annotation_width_augment_fraction,
        annotation_height_augment_fraction,
    )
    # Check if the ground truth tap action falls in an annotation's bounding box.
    tap_in_box = action_matching._yx_in_bounding_boxes(tap_yx, resized_annotation_positions)
    return tap_in_box

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

def fetch_episode(dataset_name, data_split, get_images, get_annotations, get_actions, output_dir):
    filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

    with open (data_split, "r") as rp:
        split_data = json.load(rp)
        train_data = split_data["train"]
        val_data = split_data["validation"]
        test_data = split_data["test"]
        print(f"train_data size: {len(train_data)}, val_data size: {len(val_data)}, test_data size: {len(test_data)}")
        print(train_data[0], val_data[0], test_data[0])

    all_parsed_episode = {
        "train": [],
        "val": [],
        "test": [],
    }
    total_screens = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    episode = []
    episode_id = None
    
    for d in tqdm(dataset):
        ex = tf.train.Example()
        ex.ParseFromString(d)
        ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        if '.' in ep_id:
            ep_id = ep_id.split(".")[0]
        # if (ep_id not in train_data) & (ep_id not in test_data):
        #     continue
        if episode_id is None:
            episode_id = ep_id
            episode.append(ex)
        elif ep_id == episode_id:
            episode.append(ex)
        else:
            # save data
            try:
                # here is a bug: ep_id is the new episode, this should be episode_id
                output = parse_episode(episode, ep_id, get_images=get_images, get_annotations=get_annotations, get_actions=get_actions, output_dir = output_dir)
            except Exception as exc:
                print(exc)
                #  bad data point; init a new episode
                episode_id = ep_id
                episode = [ex]

            if int(episode_id) in train_data:
                curr_split = "train"
            elif int(episode_id) in val_data:
                curr_split = "val"
            elif int(episode_id) in test_data:
                curr_split = "test"
            else:
                print("error episode")
            # print(all_parsed_episode)
            all_parsed_episode[curr_split].append({"episode_id":episode_id, "data":output})
            total_screens[curr_split] += len(episode)
            # init a new episode
            episode_id = ep_id
            episode = [ex]
    # last episode
    if len(episode) > 0:
        # save data
        output = parse_episode(episode, ep_id, get_images=get_images, get_annotations=get_annotations, get_actions=get_actions, output_dir = output_dir)
        if episode_id in train_data:
            curr_split = "train"
        elif episode_id in val_data:
            curr_split = "val"
        elif episode_id in test_data:
            curr_split = "test"
        else:
            assert "error episode"
        
        all_parsed_episode[curr_split].append({"episode_id":episode_id, "data":output})
        total_screens[curr_split] += len(episode)

    print(len(all_parsed_episode["train"]), total_screens["train"], len(all_parsed_episode["val"]), total_screens["val"], len(all_parsed_episode["test"]), total_screens["test"])
    return all_parsed_episode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='general')
    parser.add_argument("--split_file", type=str, default="dataset/general_texts_splits.json")
    parser.add_argument('--output_dir', type=str, default='dataset/t5/general_parsed_episode_t5_clip')
    # parser.add_argument('--get_images', action='store_true')
    # parser.add_argument('--get_annotations', action='store_true')
    # parser.add_argument('--get_actions', action='store_true')

    parser.add_argument('--get_images', default=True, action='store_true')
    parser.add_argument('--get_annotations', default=True, action='store_true')
    parser.add_argument('--get_actions', default=True, action='store_true')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    all_parsed_episode = fetch_episode(args.dataset, args.split_file, args.get_images, args.get_annotations, args.get_actions, args.output_dir)
    
    with open(f"{args.output_dir}_train.obj", "wb") as wp:
        pickle.dump(all_parsed_episode["train"],wp)
    with open(f"{args.output_dir}_val.obj", "wb") as wp:
        pickle.dump(all_parsed_episode["val"],wp)
    with open(f"{args.output_dir}_test.obj", "wb") as wp:
        pickle.dump(all_parsed_episode["test"],wp)

# python fetch_dataset_for_t5_blipv2.py --split_file "dataset/splits/standard.json" --output_dir "dataset/owl/general_parsed_episode_owl"
# python fetch_dataset_for_t5_blipv2.py --split_file "dataset/splits/standard.json" --output_dir "dataset/owl/install_parsed_episode_owl" --dataset install
# python fetch_dataset_for_t5_blipv2.py --split_file "dataset/splits/standard.json" --output_dir "dataset/owl/google_apps_parsed_episode_owl" --dataset google_apps
# CUDA_VISIBLE_DEVICES=5 python fetch_dataset_for_t5_blipv2.py --split_file "dataset/splits/standard.json" --output_dir "dataset/owl/single_parsed_episode_owl" --dataset single ---------failed, the episode id is not inttt
# python fetch_dataset_for_t5_blipv2.py --split_file "dataset/splits/standard.json" --output_dir "dataset/owl/web_shopping_parsed_episode_owl" --dataset web_shopping

