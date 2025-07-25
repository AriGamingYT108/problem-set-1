'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

df_test = pd.read_csv("data/df_arrests_test.csv")

calibration_plot(df_test['y'], df_test['pred_lr'], n_bins=5)
calibration_plot(df_test['y'], df_test['pred_dt'], n_bins=5)

def calib_error(y_true, y_prob):
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=5)
    return abs(bin_means - prob_true).mean()

err_lr = calib_error(df_test['y'], df_test['pred_lr'])
err_dt = calib_error(df_test['y'], df_test['pred_dt'])
better = "Logistic Regression" if err_lr < err_dt else "Decision Tree"
print("Which model is more calibrated?", better)

# Extra Credit: PPV and AUC calculations

# 1) PPV for top 50 arrestees
top50_lr = df_test.nlargest(50, 'pred_lr')
ppv_lr   = top50_lr['y'].mean()

top50_dt = df_test.nlargest(50, 'pred_dt')
ppv_dt   = top50_dt['y'].mean()

print(f"PPV for top 50 (Logistic Regression): {ppv_lr:.3f}")
print(f"PPV for top 50 (Decision Tree):          {ppv_dt:.3f}")

# AUC calculations
auc_lr = roc_auc_score(df_test['y'], df_test['pred_lr'])
auc_dt = roc_auc_score(df_test['y'], df_test['pred_dt'])

print(f"AUC (Logistic Regression): {auc_lr:.3f}")
print(f"AUC (Decision Tree):       {auc_dt:.3f}")

# 3) Do both metrics agree?
better_ppv = "Logistic Regression" if ppv_lr > ppv_dt else "Decision Tree"
better_auc = "Logistic Regression" if auc_lr > auc_dt else "Decision Tree"
agreement  = (better_ppv == better_auc)

print(
    "Do both PPV and AUC agree on which model is more accurate?",
    "Yes" if agreement else "No"
)