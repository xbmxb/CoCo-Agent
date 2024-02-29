# splits 
# standard.json from https://github.com/google-research/google-research/tree/master/android_in_the_wild
# single.json from https://github.com/cooelf/Auto-UI/tree/main

# prepare data
python fetch_dataset_for_t5_blipv2.py --split_file "dataset/splits/standard.json" --output_dir "dataset/owl/general_parsed_episode_owl"
python covert_aitw_to_llavacot_hist_fullset.py