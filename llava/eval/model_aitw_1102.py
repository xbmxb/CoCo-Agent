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

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

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

    data = load_for_owl('.', 'test')
    # questions = data
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))[:1000]
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
        idx = line["id"]
        # question = line['conversations'][0]
        # qs = question['value'].replace('<image>', '').strip()
        # cur_prompt = qs
        # print("original dataset: ", line['conversations'])
        if 'image' in line:
            image_file = line["image"]
            if type(image_file) == list:
                image = [ Image.open(os.path.join(args.image_folder, f)) for f in image_file ] 
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            else:
                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            # if getattr(model.config, 'mm_use_im_start_end', False):
            #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            # else:
            #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            # qs = question['value'].strip()
            # cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
        conv = conv_templates[args.conv_mode].copy()
        for j, question in enumerate(line['conversations'][:-1]):
            conv.append_message(conv.roles[j%2], question['value'])
            # conv.append_message(conv.roles[j%2], None)
        conv.append_message(conv.roles[len(line['conversations'])%2], None)
        prompt = conv.get_prompt()
        # print(prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True,
                temperature=0.2,
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
        if args.answer_prompter:
            outputs_reasoning = outputs
            input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=64,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs

        preds.append(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    
    print('file closed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3")
    parser.add_argument("--model-path", type=str, default="/data/maxb/tag/LLaVA/checkpoints/aitwfull_4histwimg_dialog-llama-2-7b-chat-finetune/llava-checkpoint-5000")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="/data/maxb/tag/LLaVA/scripts/aitw_data/histwimg/llava_aitwfull_debug_dialog_4histwimg_test_QCM-LEA.json")
    parser.add_argument("--answers-file", type=str, default="./debug_dialog_answer_test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
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
# CUDA_VISIBLE_DEVICES=2,3,1,0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_centertype-llama-2-7b-chat-finetune --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_centertype_test_QCM-LEA.json  --answers-file ./res_out/centertype_12k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=2,3,1,0 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_centertype-llama-2-7b-chat-finetune/checkpoint-5000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_centertype_test_QCM-LEA.json  --answers-file ./res_out/centertype_5k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5,6,7 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_centertype-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_centerfull_test_QCM-LEA.json  --answers-file ./res_out/centerfull_10k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=4,5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_ret-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_retrieve_test_QCM-LEA.json  --answers-file ./res_out/retrieve_10k_v2_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_ret-llama-2-7b-chat-finetune/checkpoint-5000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_retrieve_test_QCM-LEA.json  --answers-file ./res_out/retrieve_5k_llava_try-llama-2-7b-chat-finetune.jsonl
# CUDA_VISIBLE_DEVICES=4,5 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_highlevel-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_highlevel_test_QCM-LEA.json  --answers-file ./res_out/highlevel_10k_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/debug/llava-checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/histwimg/llava_aitwfull_4histwimg_test_QCM-LEA.json  --answers-file ./res_out/4histwimg_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=3 python model_aitw.py --model-path /data/maxb/tag/LLaVA/checkpoints/llava_hist12_ret-llama-2-7b-chat-finetune/checkpoint-10000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/memory/llava_aitwfull_hist12_retrieve_test_QCM-LEA.json  --answers-file ./res_out/retrieve_10k_v2_llava_try-llama-2-7b-chat-finetune.jsonl

# CUDA_VISIBLE_DEVICES=7 python model_aitw_1102.py --model-path /data/maxb/tag/LLaVA/checkpoints/aitwfull_4histwimg_dialog-llama-2-7b-chat-finetune/llava-checkpoint-5000 --question-file /data/maxb/tag/LLaVA/scripts/aitw_data/histwimg/llava_aitwfull_dialog_4histwimg_test_QCM-LEA.json  --answers-file ./res_out/dialog_4histwimg_llava_try-llama-2-7b-chat-finetune.jsonl