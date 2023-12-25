import argparse
import torch
import os
import json, jsonlines
from tqdm import tqdm
# import shortuuid

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from utils_data_for_owl import load_for_owl

from PIL import Image
import math
import action_type
import action_matching

# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
scroll_map = {
    "up": [[0.8, 0.5], [0.2, 0.5]],
    "down": [[0.2, 0.5], [0.8, 0.5]],
    "left": [[0.5, 0.8], [0.5, 0.2]],
    "right": [[0.5, 0.2], [0.5, 0.8]]
}

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_for_metrics(args):
    pred_data = []
    if args.answers_file.endswith('.jsonl'):
        with jsonlines.open(args.answers_file, 'r') as pf:
            for line in pf:
                pred_data.append(line)
    else:
        with open(args.answers_file, 'r') as pf:
            pred_data = json.load(pf)

    data = load_for_owl('.', 'test')
    data = data[:len(pred_data)] # sequential
    # questions = data
    # questions = json.load(open(os.path.expanduser(args.question_file), "r"))[:10]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    preds = []
    targets_cot = []
    targets = []
    metrics = {}
    cot_acc = 0
    partial_correct = 0
    text_correct = 0
    type_correct = 0
    reference_test_positions = []
    # acc for each type
    cot_type_p = {
        '<STATUS_TASK_COMPLETE>': [0,0,0],
        '<STATUS_TASK_IMPOSSIBLE>': [0,0,0],
        '<PRESS_BACK>': [0,0,0],
        '<PRESS_HOME>': [0,0,0],
        '<PRESS_ENTER>': [0,0,0],
        '<TYPE>': [0,0,0],
        '<TAP>': [0,0,0],
        '<CLICK>': [0,0,0],
        '<SCROLL>': [0,0,0]
    }
    cot_type_r = {
        '<STATUS_TASK_COMPLETE>': [0,0,0],
        '<STATUS_TASK_IMPOSSIBLE>': [0,0,0],
        '<PRESS_BACK>': [0,0,0],
        '<PRESS_HOME>': [0,0,0],
        '<PRESS_ENTER>': [0,0,0],
        '<TYPE>': [0,0,0],
        '<TAP>': [0,0,0],
        '<CLICK>': [0,0,0],
        '<SCROLL>': [0,0,0]
    }
    cot_type_f = {
        '<STATUS_TASK_COMPLETE>': [0,0],
        '<STATUS_TASK_IMPOSSIBLE>': [0,0],
        '<PRESS_BACK>': [0,0],
        '<PRESS_HOME>': [0,0],
        '<PRESS_ENTER>': [0,0],
        '<TYPE>': [0,0],
        '<TAP>': [0,0],
        '<CLICK>': [0,0],
        '<SCROLL>': [0,0]
    }
    if 'question_id' not in pred_data[0].keys() and 'id' not in pred_data[0].keys() :
        pred_data = pred_data[1:] # the first line is the result line
    for i, line in enumerate(tqdm(pred_data)):
        # if i > 10:
        #     break
        targets.append(data[i]['target_text'])
        reference_test_positions.append(data[i]['anno_pos'])
        # print('assert: ', targets[-1], line['conversations'][-1]['value'])
        idx = line["question_id"]
        # idx = line['id']
        # print('assert: ', idx, data[i]['image'])
        assert idx == data[i]['image']
        preds.append(line['text'])
        targets_cot.append(line['ground_truth'])
    
    print('file closed')
    # metrics
    output_data = []

    assert len(preds) == len(targets)   == len(reference_test_positions)
    for idx, pred in enumerate(preds):
        try:
            reference = eval("{" + targets[idx] + "}")
            reference_cot = targets_cot[idx]
        except:
            print("reference error")
            continue
        
        # cot type acc
        if ', "Correct image":' in preds[idx]:
            preds[idx] = preds[idx].split(', "Correct image":')[0]
        if '\nNext action:' in preds[idx]:
            preds[idx] = preds[idx].split('\nNext action:')[0]
        print(preds[idx])
        cot1 = '<' + preds[idx].split('<')[1].split(">")[0] + '>'
        cot2 ='<' + reference_cot.split('<')[1].split(">")[0] + '>'
        if cot1 == cot2:
            cot_acc += 1
        else:
            print('not match in cot action: ', cot1, cot2, reference_cot)
            if cot1 not in cot_type_f.keys():
                cot1 = '<STATUS_TASK_IMPOSSIBLE>'
        # print(preds[idx])
        # try:
        #     pred = eval("{" + preds[idx] + "}")
        #     action_1_touch_yx = eval(pred["touch_point"])
        #     action_1_lift_yx = eval(pred["lift_point"])
        #     action_1_action_type = action_type.ActionType[pred["action_type"]].value
        #     action_1_typed_text = pred["typed_text"].lower()
        #     action_1_typed_text = action_1_typed_text.strip()

        #     action_1_wrap = f'"action_type": "{action_1_action_type}", "touch_point": "{action_1_touch_yx}", "lift_point": "{action_1_lift_yx}", "typed_text": "{action_1_typed_text}"'
        #     action_1_wrap = action_1_wrap.replace('"', "'")
        # except:
        #     pred = eval('{ "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "Invalid"}')
        #     action_1_touch_yx = eval(pred["touch_point"])
        #     action_1_lift_yx = eval(pred["lift_point"])
        #     action_1_action_type = action_type.ActionType[pred["action_type"]].value
        #     action_1_typed_text = pred["typed_text"].lower()
        #     action_1_typed_text = action_1_typed_text.strip()

        #     action_1_wrap = f'"action_type": "{action_1_action_type}", "touch_point": "{action_1_touch_yx}", "lift_point": "{action_1_lift_yx}", "typed_text": "{action_1_typed_text}"'
        #     action_1_wrap = action_1_wrap.replace('"', "'")
            
        # parsing for CoT
        try:
            if '<STATUS_TASK_COMPLETE>' in preds[idx]:
                pred = eval('{ "action_type": "STATUS_TASK_COMPLETE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""}')
            elif '<PRESS_BACK>' in preds[idx]:
                pred = eval('{ "action_type": "PRESS_BACK", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""}')
            elif '<PRESS_HOME>' in preds[idx]:
                pred = eval('{ "action_type": "PRESS_HOME", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""}')
            elif '<PRESS_ENTER>' in preds[idx]:
                pred = eval('{ "action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""}')
            elif '<TYPE>' in preds[idx]:
                pred_type_text = preds[idx].split('"typed_text":')[1].strip()
                pred_type_text = pred_type_text.replace('"', '')
                pred = eval('{ "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""}')
                pred['typed_text'] = pred_type_text
            elif '<TAP>' in preds[idx] or '<CLICK>' in preds[idx]:
                tap_point = eval(preds[idx].split('"tap_point":')[1].strip())
                pred = eval('{ "action_type": "DUAL_POINT", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""}')
                pred['touch_point'] = tap_point
                pred['lift_point'] = tap_point
            elif '<SCROLL>' in preds[idx]:
                tp = eval(preds[idx].split('"touch_point": ')[1].split(', "lift_point": ')[0].strip())
                lp = eval(preds[idx].split(', "lift_point": ')[1].strip())
                pred = eval('{ "action_type": "DUAL_POINT", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""}')
                pred['touch_point'] = tp
                pred['lift_point'] = lp
            else:
                pred = eval('{ "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "Invalid"}')
        except:
            pred = eval('{ "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "Invalid"}')
        # print(pred)
        try:
            action_1_touch_yx = eval(pred["touch_point"])
            action_1_lift_yx = eval(pred["lift_point"])
        except:
            action_1_touch_yx = [-1.0, -1.0]
            action_1_lift_yx = [-1.0, -1.0]
        action_1_action_type = action_type.ActionType[pred["action_type"]].value
        action_1_typed_text = pred["typed_text"].lower()
        action_1_wrap = f'"action_type": "{action_1_action_type}", "touch_point": "{action_1_touch_yx}", "lift_point": "{action_1_lift_yx}, "typed_text": "{action_1_typed_text}"'
        action_1_wrap = action_1_wrap.replace('"', "'")
        
        action_2_touch_yx = eval(reference["touch_point"])
        action_2_lift_yx = eval(reference["lift_point"])
        action_2_action_type = action_type.ActionType[reference["action_type"]].value
        action_2_typed_text = reference["typed_text"].lower()
        
        action_2_wrap = f'"action_type": "{action_2_action_type}", "touch_point": "{action_2_touch_yx}", "lift_point": "{action_2_lift_yx}, "typed_text": "{action_2_typed_text}"'
        action_2_wrap = action_2_wrap.replace('"', "'")

        annotation_positions = reference_test_positions[idx]

        try:
            check_match = action_matching.check_actions_match(
                action_1_touch_yx,
                action_1_lift_yx,
                action_1_action_type,
                action_2_touch_yx,
                action_2_lift_yx,
                action_2_action_type,
                annotation_positions
            )

        except Exception as exc:
            print(idx, action_1_touch_yx, action_1_lift_yx)
            check_match = False
            match_label = "invalid"

        if check_match:
            partial_correct += 1
            match_label = 1
            # print(cot_type_acc[cot2])
            cot_type_p[cot2][2] += 1
            cot_type_r[cot1][2] += 1
        else:
            match_label = 0
        if check_match and (action_1_typed_text in action_2_typed_text or action_2_typed_text in action_1_typed_text):
            text_correct += 1
        if action_1_action_type == action_2_action_type:
            type_correct += 1
            cot_type_p[cot2][1] += 1
            cot_type_r[cot1][1] += 1
        if check_match and not (action_1_typed_text in action_2_typed_text or action_2_typed_text in action_1_typed_text):
            print(action_1_wrap, action_2_wrap)
        action_data = {"pred": action_1_wrap, "target": action_2_wrap, "match_label": match_label}
        output_data.append(action_data)
        
        # cot type acc
        cot_type_p[cot2][0] += 1  # ground cot type count
        cot_type_r[cot1][0] += 1 

    for k in cot_type_p.keys():
        cot_type_p[k][1] = cot_type_p[k][1] / cot_type_p[k][0] * 100
        cot_type_p[k][2] = cot_type_p[k][2] / cot_type_p[k][0] * 100
        if cot_type_r[k][0] == 0:
            cot_type_r[k][1] = 0
            cot_type_r[k][2] = 0
        else:
            cot_type_r[k][1] = cot_type_r[k][1] / cot_type_r[k][0] * 100
            cot_type_r[k][2] = cot_type_r[k][2] / cot_type_r[k][0] * 100
        # print( 2 * cot_type_p[k][1] * cot_type_r[k][1] / (cot_type_p[k][1] + cot_type_r[k][1]))
        # cot_type_f[k][0] = "{:.2f}".format(2 * cot_type_p[k][1] * cot_type_r[k][1] / (cot_type_p[k][1] + cot_type_r[k][1]))
        # cot_type_f[k][1] = "{:.2f}".format(2 * cot_type_p[k][2] * cot_type_r[k][2] / (cot_type_p[k][2] + cot_type_r[k][2]))
        cot_type_p[k][1] = "{:.2f}".format(cot_type_p[k][1])
        cot_type_p[k][2] = "{:.2f}".format(cot_type_p[k][2])
        cot_type_r[k][1] = "{:.2f}".format(cot_type_r[k][1])
        cot_type_r[k][2] = "{:.2f}".format(cot_type_r[k][2])
        
    metrics['cot_type_p'] = cot_type_p
    metrics['cot_type_r'] = cot_type_r
    metrics['cot_type_f'] = cot_type_f
    metrics["cot_acc"] = "{:.2f}".format(cot_acc/len(targets) * 100)
    metrics["partial_acc"] = "{:.2f}".format(partial_correct/len(targets) * 100)
    metrics["text_acc"] = "{:.2f}".format(text_correct/len(targets) * 100)
    metrics["type_acc"] = "{:.2f}".format(type_correct/len(targets) * 100)
    metrics["partial_correct"] = partial_correct
    metrics["text_correct"] = text_correct
    metrics["type_correct"] = type_correct
    metrics["total_num"] = len(targets)
    print(metrics)
    output_data_dic = {
        "metrics": metrics,
        "data": output_data
    }
    print(args.prd_output_path)
    if args.save_path:
        args.prd_output_path = os.path.join(args.save_path, args.prd_output_path)
    print(args.prd_output_path)
    if not os.path.exists(args.prd_output_path):
        os.mkdir(args.prd_output_path)
    if args.eval_name:
        output_prediction_file = os.path.join(args.prd_output_path,f"predictions_ans_test_{args.eval_name}.json")
    else:
        output_prediction_file = os.path.join(args.prd_output_path,"predictions_ans_test.json")
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_data_dic, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3")
    # parser.add_argument("--model-path", type=str, default="/data/maxb/tag/LLaVA/checkpoints/llava_try-llama-2-7b-chat-finetune")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="")
    # parser.add_argument("--question-file", type=str, default="/data/maxb/tag/LLaVA/scripts/llava_aitwv2_train_QCM-LEA.json")
    parser.add_argument("--answers-file", type=str, default="/data/maxb/mic_1025/checkpoints/instruct_BLIP_deepSpeed_t5xxl_unfreeze_Projection_LLM_QV_weight_without_instruct_qformer_v3_debug/checkpoint-1500/debug_generations.json")
    # parser.add_argument("--conv-mode", type=str, default="llava_llama_2")
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--prd_output_path", type=str, default='/data/maxb/mic_1025/checkpoints/instruct_BLIP_deepSpeed_t5xxl_unfreeze_Projection_LLM_QV_weight_without_instruct_qformer_v3_debug/checkpoint-1500')
    parser.add_argument("--eval_name", type=str, default='debug_generations_res')
    parser.add_argument("--eval_data", type=str, default='/data/maxb/mmcot2/dataset/owl/general_parsed_episode_owl_train.obj')
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    eval_for_metrics(args)

# conda activate mm
# python eval_aitw.py
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_llava_try-llama-2-7b-chat-finetune.jsonl
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/5k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_5k_llava_try-llama-2-7b-chat-finetune
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/10k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_10k_llava_try-llama-2-7b-chat-finetune
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/20hist_10k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_20hist_10k_llava_try-llama-2-7b-chat-finetune
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/centertype_10k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_centertype_10k_llava_try-llama-2-7b-chat-finetune
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/centertype_12k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_centertype_12k_llava_try-llama-2-7b-chat-finetune
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/centerfull_10k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_centerfull_10k_llava_try-llama-2-7b-chat-finetune
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/centerfull_5k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_centerfull_5k_llava_try-llama-2-7b-chat-finetune

# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/retrieve_5k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_retrieve_5k_llava_try-llama-2-7b-chat-finetune
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/highlevel_10k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_highlevel_10k_llava_try-llama-2-7b-chat-finetune

# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/retrieve_10k_v2_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_retrieve_10k_v2_llava_try-llama-2-7b-chat-finetune

# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/4histwimg_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/4histwimg_llava_try-llama-2-7b-chat-finetune

# python eval_aitw.py --answers-file /data/maxb/tag/answers_test.json  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_mplug_test
# python eval_aitw.py --answers-file /data/maxb/tag/answers_test_30ckp.json  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_mplug_test_30ckp
# python eval_aitw.py --answers-file /data/maxb/tag/answers_test_30ckp_1shot.json  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_mplug_test_30ckp_1shot
# python eval_aitw.py --answers-file /data/maxb/tag/answers_test_30ckp_1shot_hist.json  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_mplug_test_30ckp_1shot_hist
# python eval_aitw.py --answers-file /data/maxb/tag/answers_test_30ckp_1shot_hist.json  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_out/res_mplug_test_30ckp_1shot_hist

# python eval_aitw.py --answers-file /data/maxb/tag/fewshot_mplug/orig_nothing.jsonl  --prd_output_path /data/maxb/tag/fewshot_mplug --eval_name orig_nothing
# python eval_aitw.py --answers-file /data/maxb/tag/fewshot_mplug/orig_1shot.jsonl  --prd_output_path /data/maxb/tag/fewshot_mplug --eval_name orig_1shot
# python eval_aitw.py --answers-file /data/maxb/tag/fewshot_mplug/orig_1shot_hist2.jsonl  --prd_output_path /data/maxb/tag/fewshot_mplug --eval_name orig_1shot_hist2
# python eval_aitw.py --answers-file /data/maxb/tag/fewshot_mplug/orig_1shot_hist_memo.jsonl  --prd_output_path /data/maxb/tag/fewshot_mplug --eval_name orig_1shot_hist_memo
# python eval_aitw.py --answers-file /data/maxb/tag/fewshot_mplug/orig_1shot_hist_memo_norm.jsonl  --prd_output_path /data/maxb/tag/fewshot_mplug --eval_name orig_1shot_hist_memo_norm

# python eval_aitw.py --answers-file /data/maxb/tag/fewshot_mplug/orig_1shot_1k.jsonl  --prd_output_path /data/maxb/tag/fewshot_mplug --eval_name orig_1shot_1k
# python eval_aitw.py --answers-file /data/maxb/tag/fewshot_mplug/orig_1shot_hist2_1k.jsonl  --prd_output_path /data/maxb/tag/fewshot_mplug --eval_name orig_1shot_hist2_1k
# python eval_aitw.py --answers-file /data/maxb/tag/fewshot_mplug/orig_1shot_hist_memo_1k.jsonl  --prd_output_path /data/maxb/tag/fewshot_mplug --eval_name orig_1shot_hist_memo_1k


# python eval_aitw.py --answers-file /data/maxb/mic_1025/debug_generations.json  --prd_output_path /data/maxb/mic_1025/ --eval_name debug_generations_res

# python eval_aitw.py --answers-file /data/maxb/mic_1025/checkpoints/instruct_BLIP_deepSpeed_t5xxl_unfreeze_Projection_LLM_QV_weight_without_instruct_qformer_v3_debug/checkpoint-1500/debug_generations.json  --prd_output_path /data/maxb/mic_1025/checkpoints/instruct_BLIP_deepSpeed_t5xxl_unfreeze_Projection_LLM_QV_weight_without_instruct_qformer_v3_debug/checkpoint-1500 --eval_name debug_generations_res


# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/debug_dialog_answer_test.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_debug_dialog_answer_test.jsonl --eval_name debug_res
# python eval_aitw.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/dialog_4histwimg_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_dialog_out --eval_name llava_aitwfull_dialog_4histwimg_test_QCM-LEA

# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_12k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_enter_quotation
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_3k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_3k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_6k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_9k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_9k

# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_norm_12k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_norm_12k

# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_norm_9k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_norm_9k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_norm_6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_norm_6k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_norm_3k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_norm_3k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_out/cotv0_norm_last_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_norm_last

# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8hist_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8hist_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8hist3k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8hist3k_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8hist9k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8hist9k_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8hist12k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8hist12k_scores

# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8histlocation6k_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8histlocation3k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8histlocation3k_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8histlocation9k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8histlocation9k_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8histlocation12k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8histlocation12k_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8histlocationtotal_scores

# python eval_aitw_cot.py --answers-file ./res_continue/30k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv0_30k_scores_plus
# python eval_aitw_cot.py --answers-file ./res_continue/60k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv0_60k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/90k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv0_90k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/12k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv1_12k_scores_2
# python eval_aitw_cot.py --answers-file ./res_continue/9k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv1_9k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/6k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv1_6k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/3k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv1_3k_scores


# python eval_aitw_cot.py --answers-file ./res_continue/6k_debug_ep1_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv1_debug_6k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/6k_v2_ep1_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name contiv1_v2_6k_scores
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_continue/6k_debug_ep1_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name debug_6k_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name debug_cotv0_8histlocation6k_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/debug_t0_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name debug_t0_cotv0_8histlocation6k_scores

# python eval_aitw_cot.py --answers-file ./res_cot_out/cotv0_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name cotv0_8histlocationtotal_scores_plus
# python eval_aitw_cot.py --answers-file ./res_continue/conv2_6k_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name conv2_t0_6k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/conv2_6k_t0_savevision_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name conv2_t0_savevision_6k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/conv2_9k_t0_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name conv2_t0_9k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/conv2_12k_t0_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name conv2_t0_12k_scores
# python eval_aitw_cot.py --answers-file ./res_continue/conv2_total_t0t0_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl  --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_continue --eval_name conv2_t0t0_total_scores
# python eval_aitw_cot.py --answers-file ./res_cot_out/debugt0_cotv0_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_cot_out --eval_name debugt0_cotv0_8histlocationtotal

# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_multitask/continue_ep5_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name continue_multi_ep5_cotv0_8histlocationtotal
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_multitask/continue_ep6_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name continue_multi_ep6_cotv0_8histlocationtotal
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_multitask/continue_ep3_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name continue_multi_ep3_cotv0_8histlocationtotal
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_multitask/continue_ep4_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name continue_multi_ep4_cotv0_8histlocationtotal
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_multitask/continue_ep1_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name continue_multi_ep1_cotv0_8histlocationtotal
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_multitask/continue_ep2_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name continue_multi_ep2_cotv0_8histlocationtotal
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_multitask/continue_ep7_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name continue_multi_ep7_cotv0_8histlocationtotal
# python eval_aitw_cot.py --answers-file ./res_multitask/nocorrupt_v0_ep6_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name transfer_nocorrupt_v0_ep6

# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_onlytype/ep0_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_onlytype --eval_name ep0
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_onlytype/ep2_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_onlytype --eval_name ep2
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_onlytype/ep3_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_onlytype --eval_name ep3
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_onlytype/ep4_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_onlytype --eval_name ep4

# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_onlytype/continue_6k_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_onlytype --eval_name continue_6k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_onlytype/continue_9k_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_onlytype --eval_name continue_9k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_onlytype/continue_12k_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_onlytype --eval_name continue_12k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_onlytype/continue_total_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_onlytype --eval_name continue_total

# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_future/6k_future2_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_future --eval_name 6k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_future/9k_future2_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_future --eval_name 9k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_future/12k_future2_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_future --eval_name 12k
# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_future/total_future2_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_future --eval_name total

# python eval_aitw_cot.py --answers-file /data/maxb/tag/LLaVA/llava/eval/res_multitask/bs16_6k_llava_try-llama-2-7b-chat-finetune.jsonl --prd_output_path /data/maxb/tag/LLaVA/llava/eval/res_multitask --eval_name bs16_6k