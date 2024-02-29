import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from utils_data_for_owl import load_for_owl

from PIL import Image
import math
import action_type
import action_matching
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    print(model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    data = load_for_owl('.', 'test', data_name=args.data_name)
    # questions = data
    # questions = json.load(open(os.path.expanduser(args.question_file), "r"))[:1000]
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = random.sample(questions, 5000)
    if args.num != None:
        questions = questions[:args.num]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    preds = []
    targets = []
    metrics = {}
    partial_correct = 0
    text_correct = 0
    type_correct = 0
    reference_test_positions = []
    for i, line in enumerate(tqdm(questions)):
        targets.append(data[i]['target_text'])
        reference_test_positions.append(data[i]['anno_pos'])
        print('assert: ', targets[-1], line['conversations'][-1]['value'])
        # assert targets[-1] == line['conversations'][-1]['value']
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        # print("original dataset: ", line['conversations'])
        gt = line['conversations'][1]['value']
        if 'image' in line:
            image_file = line["image"]
            if type(image_file) == list:
                image = [ Image.open(os.path.join(args.image_folder, f)) for f in image_file ] 
            else:
                image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            qs = question['value'].strip()
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                # do_sample=True,
                # num_beams = 4,
                do_sample=False,
                temperature=0.2,
                # temperature=0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(outputs)

        # prompt for answer
        # if args.answer_prompter:
        #     outputs_reasoning = outputs
        #     input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        #     with torch.inference_mode():
        #         output_ids = model.generate(
        #             input_ids,
        #             images=images,
        #             do_sample=True,
        #             # temperature=0.2,
        #             temperature=0,
        #             max_new_tokens=64,
        #             use_cache=True,
        #             stopping_criteria=[stopping_criteria])

        #     input_token_len = input_ids.shape[1]
        #     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        #     if n_diff_input_output > 0:
        #         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        #     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        #     outputs = outputs.strip()
        #     if outputs.endswith(stop_str):
        #         outputs = outputs[:-len(stop_str)]
        #     outputs = outputs.strip()
        #     outputs = outputs_reasoning + '\n The answer is ' + outputs

        preds.append(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "ground_truth": gt,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    
    print('file closed')
    # # metrics
    # output_data = []

    # assert len(preds) == len(targets)   == len(reference_test_positions)
    # for idx, pred in enumerate(preds):
    #     try:
    #         reference = eval("{" + targets[idx] + "}")
    #     except:
    #         print("reference error")
    #         continue

    #     try:
    #         pred = eval("{" + preds[idx] + "}")
    #         action_1_touch_yx = eval(pred["touch_point"])
    #         action_1_lift_yx = eval(pred["lift_point"])
    #         action_1_action_type = action_type.ActionType[pred["action_type"]].value
    #         action_1_typed_text = pred["typed_text"].lower()
    #         action_1_typed_text = action_1_typed_text.strip()

    #         action_1_wrap = f'"action_type": "{action_1_action_type}", "touch_point": "{action_1_touch_yx}", "lift_point": "{action_1_lift_yx}", "typed_text": "{action_1_typed_text}"'
    #         action_1_wrap = action_1_wrap.replace('"', "'")
    #     except:
    #         pred = '{ "action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "Invalid"}'
        
    #     action_2_touch_yx = eval(reference["touch_point"])
    #     action_2_lift_yx = eval(reference["lift_point"])
    #     action_2_action_type = action_type.ActionType[reference["action_type"]].value
    #     action_2_typed_text = reference["typed_text"].lower()
        
    #     action_2_wrap = f'"action_type": "{action_2_action_type}", "touch_point": "{action_2_touch_yx}", "lift_point": "{action_2_lift_yx}, "typed_text": "{action_2_typed_text}"'
    #     action_2_wrap = action_2_wrap.replace('"', "'")

    #     annotation_positions = reference_test_positions[idx]

    #     try:
    #         check_match = action_matching.check_actions_match(
    #             action_1_touch_yx,
    #             action_1_lift_yx,
    #             action_1_action_type,
    #             action_2_touch_yx,
    #             action_2_lift_yx,
    #             action_2_action_type,
    #             annotation_positions
    #         )

    #     except Exception as exc:
    #         print(idx, action_1_touch_yx, action_1_lift_yx)
    #         check_match = False
    #         match_label = "invalid"

    #     if check_match:
    #         partial_correct += 1
    #         match_label = 1
    #     else:
    #         match_label = 0
    #     if check_match and (action_1_typed_text in action_2_typed_text or action_2_typed_text in action_1_typed_text):
    #         text_correct += 1
    #     if action_1_action_type == action_2_action_type:
    #         type_correct += 1

    #     action_data = {"pred": action_1_wrap, "target": action_2_wrap, "match_label": match_label}
    #     output_data.append(action_data)

    # metrics["partial_acc"] = "{:.2f}".format(partial_correct/len(targets) * 100)
    # metrics["text_acc"] = "{:.2f}".format(text_correct/len(targets) * 100)
    # metrics["type_acc"] = "{:.2f}".format(type_correct/len(targets) * 100)
    # metrics["partial_correct"] = partial_correct
    # metrics["text_correct"] = text_correct
    # metrics["type_correct"] = type_correct
    # metrics["total_num"] = len(targets)
    # print(metrics)
    # output_data_dic = {
    #     "metrics": metrics,
    #     "data": output_data
    # }
    # print(args.prd_output_path)
    # args.prd_output_path = os.path.join(args.save_path, args.prd_output_path)
    # print(args.prd_output_path)
    # if not os.path.exists(args.prd_output_path):
    #     os.mkdir(args.prd_output_path)
    # if args.eval_name:
    #     output_prediction_file = os.path.join(args.prd_output_path,f"predictions_ans_test_{args.eval_name}.json")
    # else:
    #     output_prediction_file = os.path.join(args.prd_output_path,"predictions_ans_test.json")
    # with open(output_prediction_file, "w") as writer:
    #     writer.write(json.dumps(output_data_dic, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3")
    parser.add_argument("--model-path", type=str, default="/data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12000")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="/data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json")
    parser.add_argument("--answers-file", type=str, default="./res_continue/12k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--num', type=int, default=None)
    # parser.add_argument("--prd_output_path", type=str, default='.')
    # parser.add_argument("--eval_name", type=str, default=None)
    # parser.add_argument("--eval_data", type=str, default='/data/maxb/mmcot2/dataset/owl/general_parsed_episode_owl_test.obj')
    # parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    eval_model(args)

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --question-file ../../scripts/llava_aitwv2_train_QCM-LEA.json 
# CUDA_VISIBLE_DEVICES=4,5,6,7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/llava_aitwfullhist12_test_QCM-LEA.json  --answers-file ./res_out/llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5,6,7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12-llama-2-7b-chat-finetune/checkpoint-5000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/llava_aitwfullhist12_test_QCM-LEA.json  --answers-file ./res_out/5k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5,6,7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/llava_aitwfullhist12_test_QCM-LEA.json  --answers-file ./res_out/10k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5,6,7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist20-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/llava_aitwfullhist20_test_QCM-LEA.json  --answers-file ./res_out/20hist_10k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist20-llama-2-7b-chat-finetune/ --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/llava_aitwfullhist20_test_QCM-LEA.json  --answers-file ./res_out/20hist_totalstep_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=2,3,1,0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_centertype-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_centertype_test_QCM-LEA.json  --answers-file ./res_out/centertype_12k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2,3,1,0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_centertype-llama-2-7b-chat-finetune/checkpoint-5000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_centertype_test_QCM-LEA.json  --answers-file ./res_out/centertype_5k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5,6,7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_centertype-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_centerfull_test_QCM-LEA.json  --answers-file ./res_out/centerfull_10k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=4,5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_ret-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_retrieve_test_QCM-LEA.json  --answers-file ./res_out/retrieve_10k_v2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_ret-llama-2-7b-chat-finetune/checkpoint-5000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_retrieve_test_QCM-LEA.json  --answers-file ./res_out/retrieve_5k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_highlevel-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_highlevel_test_QCM-LEA.json  --answers-file ./res_out/highlevel_10k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/debug/llava-checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/histwimg/llava_aitwfull_4histwimg_test_QCM-LEA.json  --answers-file ./res_out/4histwimg_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_ret-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_retrieve_test_QCM-LEA.json  --answers-file ./res_out/retrieve_10k_v2_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_ret-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_retrieve_test_QCM-LEA.json  --answers-file ./res_out/retrieve_10k_v2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot-llama-2-7b-chat-finetune/checkpoint-12000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_test_QCM-LEA.json  --answers-file ./res_out/cotv0_12k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_test_QCM-LEA.json  --answers-file ./res_out/cotv0_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_test_QCM-LEA.json  --answers-file ./res_out/cotv0_9k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_test_QCM-LEA.json  --answers-file ./res_out/cotv0_6k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_test_QCM-LEA.json  --answers-file ./res_out/cotv0_3k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot_norm-llama-2-7b-chat-finetune/checkpoint-12000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_norm_short_test_QCM-LEA.json  --answers-file ./res_out/cotv0_norm_12k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot_norm-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_norm_short_test_QCM-LEA.json  --answers-file ./res_out/cotv0_norm_9k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot_norm-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_norm_short_test_QCM-LEA.json  --answers-file ./res_out/cotv0_norm_6k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot_norm-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_norm_short_test_QCM-LEA.json  --answers-file ./res_out/cotv0_norm_3k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot_norm-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_norm_short_test_QCM-LEA.json  --answers-file ./res_out/cotv0_norm_last_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_cot_norm-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_nohist_cot_norm_short_test_QCM-LEA.json  --answers-file ./res_out/cotv0_norm_last_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_vision_debug-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_debug_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out/pret_debug_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out/pretv0_debug_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-55000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_55k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-60000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_60k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-45000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_45k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-40000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_40k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-35000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_35k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-50000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_50k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-80000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_80k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-85000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_85k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-90000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_90k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-95000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_95k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-100000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_100k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-105000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_105k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-110000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_110k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-115000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_115k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-120000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_120k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_pret_after_no_space-llama-2-7b-chat-finetune/checkpoint-125000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/pret/llava_aitwfull_pretrainv0_test_QCM-LEA.json  --answers-file ./res_out_pretrain/pretv0_125k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8hist_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8hist_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8hist_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8hist3k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8hist_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8hist9k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm-llama-2-7b-chat-finetune/checkpoint-12000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8hist_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8hist12k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8histlocation3k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8histlocation3k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8histlocation12k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/cotv0_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-20000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/20k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-25_partial --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/25k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-25000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/25k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-30000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/30k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-35000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/35k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-40000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/40k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-45000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/45k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-65000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/65k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-70000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/70k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/75k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-80000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/80k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-85000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/85k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-90000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/90k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-95000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/95k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-100000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/100k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-105000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/105k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_150k_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/30k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_150k_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/60k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_150k_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/90k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/12k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/9k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/6k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/3k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinuev2_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/6k_v2_ep1_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-debug_CONtinue_ep1_aitwfull_8hist_cot_norm_location_wadapter-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/6k_debug_ep1_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinue_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_continue/3k_ep1_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_continue/debug_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/debug1_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinuev2_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_continue/conv2_6k_t0_savevision_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinuev2_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_continue/conv2_9k_t0_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinuev2_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_continue/conv2_12k_t0_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONtinuev2_ep1_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/ --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_continue/conv2_total_t0t0_cotv0_8histlocation6k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=4 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_cot_out/debugt0_cotv0_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
#
# multi task
# CUDA_VISIBLE_DEVICES=4 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_multitask-llama-2-7b-chat-bs24-ep6/checkpoint-713 --question-file /data/maxb/tag/LLaVA/scripts/data4pretrain/llava_aitw_corrupted_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/v0_ep1_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_multitask-llama-2-7b-chat-bs24-ep6/checkpoint-1426 --question-file /data/maxb/tag/LLaVA/scripts/data4pretrain/llava_aitw_corrupted_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/v0_ep2_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=6 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_multitask-llama-2-7b-chat-bs24-ep6/checkpoint-2139 --question-file /data/maxb/tag/LLaVA/scripts/data4pretrain/llava_aitw_corrupted_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/v0_ep3_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_multitask-llama-2-7b-chat-bs24-ep6/checkpoint-2852 --question-file /data/maxb/tag/LLaVA/scripts/data4pretrain/llava_aitw_corrupted_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/v0_ep4_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_multitask-llama-2-7b-chat-bs24-ep6/checkpoint-3565 --question-file /data/maxb/tag/LLaVA/scripts/data4pretrain/llava_aitw_corrupted_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/v0_ep5_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_multitask-llama-2-7b-chat-bs24-ep6/checkpoint-4278 --question-file /data/maxb/tag/LLaVA/scripts/data4pretrain/llava_aitw_corrupted_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/v0_ep6_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl



# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3565 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/continue_ep5_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-4278 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/continue_ep6_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-2139 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/continue_ep3_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-2852 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/continue_ep4_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-713 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/continue_ep1_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-1426 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/continue_ep2_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-4991 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/continue_ep7_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_multitask-llama-2-7b-chat-bs24-ep6/checkpoint-4278 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/nocorrupt_v0_ep6_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_onlytype-llama-2-7b-chat-finetune/checkpoint-1426 --question-file /data/maxb/tag/LLaVA/scripts/curri/llava_aitw_type_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_onlytype/ep2_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_onlytype-llama-2-7b-chat-finetune/checkpoint-2139 --question-file /data/maxb/tag/LLaVA/scripts/curri/llava_aitw_type_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_onlytype/ep3_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-pretrain_onlytype-llama-2-7b-chat-finetune/checkpoint-2852 --question-file /data/maxb/tag/LLaVA/scripts/curri/llava_aitw_type_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_onlytype/ep4_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_onlytype4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_onlytype/continue_3k_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_onlytype4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_onlytype/continue_6k_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_onlytype4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_onlytype/continue_9k_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_onlytype4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_onlytype/continue_12k_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_onlytype4_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_onlytype/continue_total_8histlocationtotal_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_future2_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/prophet/llava_aitwfull_future2_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_future/6k_future2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_future2_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/prophet/llava_aitwfull_future2_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_future/9k_future2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_future2_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/prophet/llava_aitwfull_future2_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_future/3k_future2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_future2_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/prophet/llava_aitwfull_future2_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_future/12k_future2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_future2_8hist_cot_norm_location-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/prophet/llava_aitwfull_future2_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_future/total_future2_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location_bs16-llama-2-7b-chat-finetune/checkpoint-6000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/bs16_6k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep4_aitwfull_8hist_cot_norm_location_bs16-llama-2-7b-chat-finetune/checkpoint-9000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/bs16_9k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_future2_9k_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/3k_conti_9k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-8556 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep6_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-7130 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep5_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-5704 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep4_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-4278 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep3_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-2852 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep2_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-1426 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep1_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-9982 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep7_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-11408 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep8_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-12834 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep9_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-14260 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep10_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-15686 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep11_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/checkpoint-17112 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/ep12_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_multiep2_aitwfull_8hist_cot_norm_location_bs12-llama-2-7b-chat-finetune/ --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json  --answers-file ./res_multitask/total_conti_bs12ep2_llava_try-llama-2-7b-chat-finetune.jsonl

# full episode pretrain
# CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-5704 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/5704_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-7130 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/7130_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-8556 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/8556_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-9982 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/9982_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-11408 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/11408_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12834 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/12834_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-14260 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/14260_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-15686 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/15686_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-17112 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/17112_conti_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullsetgoopre_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-1069 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/1069_conti_fullset_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullsetgoopre_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-2138 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/2138_conti_fullset_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullsetgoopre_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-3207 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/3207_conti_fullset_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullsetgoopre_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-4276 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/4276_conti_fullset_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullsetgoopre_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-5345 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/5345_conti_fullset_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullsetgoopre_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6414 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/6414_conti_fullset_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_fullsetgoopre_fullepisode_v1_ep3_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-7483 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/7483_conti_fullset_fullepisode_v1_ep3_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-7130 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/7130_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=6 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-8556 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/8556_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-9982 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/9982_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=1 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-11408 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/beam4_11408_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12834 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/12834_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-14260 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/14260_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-15686 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/15686_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-17112 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/17112_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-17112 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/17112_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_bs8*4_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-2138 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/8*4_2138_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_bs8*4_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-6414 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/8*4_6416_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_bs8*4_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-8552 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/8*4_8552_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_bs8*4_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-10690 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/8*4_10690_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_bs8*4_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-12828 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/8*4_12828_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-CONTINUE_bs8*4_nocoordinate_fullsetgoopre_fullepisode_v2_ep1_changed_aitwfull_8hist_cot_norm_location-llama-2-7b-chat-finetune/checkpoint-14966 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/changed_llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_episode/8*4_14966_nocoordinate_fullsetgoopre_try-llama-2-7b-chat-finetune.jsonl

# baseline 12hist 
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_goopre10_install_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/install_75k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_goopre10_single_fixed_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/fixed_single_75k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_goopre10_googleapp_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/googleapp_75k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_goopre10_webshop_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/webshop_75k_llava_try-llama-2-7b-chat-finetune.jsonl

# percept fullset
# CUDA_VISIBLE_DEVICES=6 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_percept_goopre10-llama-2-7b-chat-finetune/checkpoint-5000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/percept_5k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_percept_goopre10_fixed-llama-2-7b-chat-finetune/checkpoint-35000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/cot/llava_aitwfull_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/percept/percept_35k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_goopre10_install_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/install_75k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_goopre10_single_fixed_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/fixed_single_75k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_goopre10_googleapp_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/googleapp_75k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava-aitwfull_fullset_12hist_goopre10-llama-2-7b-chat-finetune/checkpoint-75000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/llava_aitwfull_fullset_zzs_goopre10_webshop_test_QCM-LEA.json --answers-file ./res_fullsetgoopre10/webshop_75k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw.py --model-path /data/share/maxb/LLaVA/checkpoints_16cp/llava-aitwfull_googleapps_percept-llama-2-7b-chat-finetune/checkpoint-30000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/fullset/fullset_8hist_cot_norm/llava_aitwfull_google_fullsetgoopre_8histlocation_cot_norm_truncted_test_QCM-LEA.json --answers-file ./res_googleapp/30k_llava_try-llama-2-7b-chat-finetune.jsonl