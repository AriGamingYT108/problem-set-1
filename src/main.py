'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.ETL().run()
    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.build_df_arrests()
    # PART 3: Call functions/instanciate objects from logistic_regression
    X_train, y_train, df_test_lr, gs_cv = logistic_regression.run_logistic(df_arrests)
    # PART 4: Call functions/instanciate objects from decision_tree
    df_test_dt, gs_cv_dt = decision_tree.run_tree(
        X_train,
        y_train,
        df_test_lr[['num_fel_arrests_last_year', 'current_charge_felony']]
    )
    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.calibration_plot(df_test_lr['y_true'], df_test_lr['pred_lr'], n_bins=5)
    calibration_plot.calibration_plot(df_test_lr['y_true'], df_test_dt['pred_dt'], n_bins=5)

if __name__ == "__main__":
    main()