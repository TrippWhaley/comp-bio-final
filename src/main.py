import os.path
from warnings import catch_warnings

from data_loader import get_training_datasets, get_training_dataset
# from EnGen.EnGen_Iteration_preprocess import our_matching
from EnGen.EnGen_model.train import train_engen
from EnGen_iteration_preprocess_patch import our_matching
from EnGen_generate_patch import GenerateEnGen

num_iters = 3  # use 30 for real run
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# iteration num hardcoded for now, sorry
MODEL_PATH = os.path.join(BASE_PATH, '..', f'EnGen/EnGen_train_iterations/engen_output/iter_{'0' + str(num_iters) if num_iters < 10 else num_iters}/saved_model/best_model_engen.pth')

def main(ts1: str, ts2: str):
    # Only run if we haven't already preprocesses
    if not os.path.exists(os.path.join(BASE_PATH, f'Func_Pheno_45k_scaled_with_{ts1}_{ts2}_tps_source_{ts1}_target_{ts2}_matched.csv')):
        # Preprocessing
        with catch_warnings(action='ignore'):
            t1_df, t2_df = get_training_datasets(ts1, ts2)
        # Seems like only these 3 IDs exist in both 24H and 14D
        for iter_id in range(num_iters):
            our_matching(iter_id, t1_df, t2_df, AE_train_ids=['02', '03', '12'], timepoints=[ts1, ts2])

    # Only run if we don't have a model
    if not os.path.exists(MODEL_PATH):
        train_engen(num_iters)

    with catch_warnings(action='ignore'):
        source_df = get_training_dataset(ts1)
    # Always run generation for now
    generator = GenerateEnGen(model_path=MODEL_PATH, test_patient_ids=['04', '05'], source=source_df, iter_id=num_iters)
    generator.generate_csv()
if __name__ == "__main__":
    main('24H', '14D')