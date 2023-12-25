import os, json
from utils_data_for_owl_cot_hist import load_for_owl
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer

def convert_to_llava(base_dir, split, prompt_format="QCM-LEA", name=''):
    # split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[split]
    # problems = json.load(open(os.path.join(base_dir, "problems.json")))

    # split_problems = build_prompt_chatbot(
    #     problems, split_indices, prompt_format,
    #     use_caption=False, is_test=False)
    model_path = '/data/maxb/tag/LLaVA/checkpoints/llava-7b-llama-2-7b-chat'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    orig_data = load_for_owl(base_dir, split)
    conv = conv_templates['llava_llama_2'].copy()

    target_format = []
    trunced = 0
    for idx, text in enumerate(orig_data):
        input = text['text'].split('AI: ')[0]
        output = 'AI: ' + text['text'].split('AI: ')[1]
        
        if input.startswith('Human: '):
            input = input.replace('Human: ', '')
        if output.startswith('AI: '):
            output = output.replace('AI: ', '')
        assert 'image' in text.keys()
        conv = conv_templates['llava_llama_2'].copy()
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], output)
        prompt = conv.get_prompt()
        # print("first:", prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)#.unsqueeze(0)
        # print(input_ids)
        # print(len(input_ids))
        if len(input_ids) > 2048:
            # print('pick out layout to keep less than 2048')
            trunced += 1
            while len(input_ids) > 2048:
                input_locs = input.split('<image>')[-1].split('\nPrevious Actions')[0]
                # print(input_locs)
                input_locs_ = '\n'.join(input_locs.strip().split('\n')[:-1])
                # print(input_locs_)
                input = input.replace(input_locs, input_locs_)
                conv = conv_templates['llava_llama_2'].copy()
                conv.append_message(conv.roles[0], input)
                conv.append_message(conv.roles[1], output)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
                # print("trunc:", len(input_ids))
                # os._exit(0)
        target_format.append({
            "id": text['image'],
            "image": text['image'],
            "conversations": [
                {'from': 'human', 'value': f"{input}"},
                {'from': 'gpt', 'value': f"{output}"},
            ],
            # "target_text": text['target_text'],
            # "annos": text['anno_pos']
        })
    print(f'Number of samples: {len(target_format)}, trunced samples: {trunced}')
    outputfile = os.path.join('./aitw_data/cot/', f"llava_aitwfull{name}_{split}_{prompt_format}.json")
    if os.path.exists(outputfile):
        raise FileExistsError
    with open(outputfile, "w") as f:
        json.dump(target_format, f, indent=2)

convert_to_llava('.', 'train', name='_8histlocation_cot_norm_truncted')