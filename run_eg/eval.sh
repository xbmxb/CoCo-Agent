# evaluation example

CUDA_VISIBLE_DEVICES=0 python model_aitw.py --model-path xx --question-file xx --answers-file xx --data_name google_apps_parsed_episode_owl_pre10_pre10
python eval_aitw_cot.py --answers-file xx  --prd_output_path xx --eval_name xx --data_name google_apps_parsed_episode_owl_pre10_pre10