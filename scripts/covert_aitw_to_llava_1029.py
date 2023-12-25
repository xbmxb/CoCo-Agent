import os, json
from utils_data_for_owl_1029 import load_for_owl

def convert_to_llava(base_dir, split, prompt_format="QCM-LEA", name=''):
    # split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[split]
    # problems = json.load(open(os.path.join(base_dir, "problems.json")))

    # split_problems = build_prompt_chatbot(
    #     problems, split_indices, prompt_format,
    #     use_caption=False, is_test=False)
    orig_data = load_for_owl(base_dir, split)
    

    target_format = []
    for idx, text in enumerate(orig_data):
        # input = text['text'].split('AI: ')[0]
        # output = 'AI: ' + text['text'].split('AI: ')[1]
        
        # if input.startswith('Human: '):
        #     input = input.replace('Human: ', '')
        # if output.startswith('AI: '):
        #     output = output.replace('AI: ', '')
        # assert 'image' in text.keys()
        # target_format.append({
        #     "id": text['image'],
        #     "image": text['image'],
        #     "conversations": [
        #         {'from': 'human', 'value': f"{input}"},
        #         {'from': 'gpt', 'value': f"{output}"},
        #     ],
        #     # "target_text": text['target_text'],
        #     # "annos": text['anno_pos']
        # })
        a_sample = {
            'id': text['image'][-1],
            'image': text['image']
        }
        cov = []
        for line in text['text'].split('\n'):
            #utterance
            # spk, utt = line.split(': ')
            if line.startswith('Human:'):
                spk = 'human'
                utt = line.split('Human: ')[-1]
            if line.startswith('AI:'):
                spk = 'gpt'
                utt = line.split('AI: ')[-1]
            cov.append({
                'from':spk,
                'value':utt
            })
        a_sample['conversations'] = cov
        target_format.append(a_sample)
    print(f'Number of samples: {len(target_format)}')
    outputfile = os.path.join('./aitw_data/histwimg/', f"llava_aitwfull{name}_{split}_{prompt_format}.json")
    if os.path.exists(outputfile):
        raise FileExistsError
    with open(outputfile, "w") as f:
        json.dump(target_format, f, indent=2)

convert_to_llava('.', 'test', name='_dialog_4histwimg')

