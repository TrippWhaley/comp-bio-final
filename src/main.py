import os.path
from warnings import catch_warnings

from data_loader import get_training_datasets, get_training_dataset
from train_patch import train_engen
from EnGen_iteration_preprocess_patch import our_matching
from EnGen_generate_patch import GenerateEnGen

num_iters = 30  # use 30 for real run
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# iteration num hardcoded for now, sorry
num_iters_str = "0" + str(num_iters) if num_iters < 10 else num_iters
MODEL_PATH = os.path.join(BASE_PATH, '..', f'EnGen/EnGen_train_iterations/engen_output/iter_{num_iters_str}/saved_model/best_model_engen.pth')
MODEL_PATH_2 = os.path.join(BASE_PATH, '..', f'EnGen_train_iterations/engen_output/iter_{num_iters_str}/saved_model/best_model_engen.pth')


# train_ids = ['02', '03', '12'] # 14D
train_ids = ['03', '09'] # 30D
generate_ids = ['0'+str(i) if i < 10 else str(i) for i in range(2, 26)]
generate_ids = [x for x in generate_ids if x not in train_ids]
def main(ts1: str, ts2: str, run_only=False, model_path=MODEL_PATH):
    if not run_only:
        # Only run if we haven't already preprocesses
        if not os.path.exists(os.path.join(BASE_PATH, f'Func_Pheno_45k_scaled_with_{ts1}_{ts2}_tps_source_{ts1}_target_{ts2}_matched.csv')):
            # Preprocessing
            with catch_warnings(action='ignore'):
                t1_df, t2_df = get_training_datasets(ts1, ts2)
            # Seems like only these 3 IDs exist in both 24H and 14D
            for iter_id in range(num_iters):
                our_matching(iter_id, t1_df, t2_df, AE_train_ids=train_ids, timepoints=[ts1, ts2])

        # Only run if we don't have a model
        if not os.path.exists(MODEL_PATH):
            train_engen(num_iters)

    with catch_warnings(action='ignore'):
        source_df = get_training_dataset(ts1)
    # Always run generation for now
    generator = GenerateEnGen(model_path=model_path, test_patient_ids=generate_ids, source=source_df, iter_id=num_iters)
    generator.generate_csv()
if __name__ == "__main__":
    # main('24H', '30D')
    main('24H', '30D', True, MODEL_PATH_2)