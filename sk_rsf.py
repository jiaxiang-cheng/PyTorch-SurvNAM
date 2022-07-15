import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest


# First, we need to load the data and transform it into numeric values.
X, y = load_gbsg2()

grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

X_no_grade = X.drop("tgrade", axis=1)
Xt = OneHotEncoder().fit_transform(X_no_grade)
Xt.loc[:, "tgrade"] = grade_num

# Next, the data is split into 75% for training and 25% for testing, so we can determine how well our model generalizes.
random_state = 20
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)

# Training =============================================================================================================

# Several split criterion have been proposed in the past, but the most widespread one is based on the
# log-rank test, which you probably know from comparing survival curves among two or more groups. Using the training
# data, we fit a Random Survival Forest comprising 1000 trees.
rsf = RandomSurvivalForest(n_estimators=1000,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=-1,
                           random_state=random_state)
rsf.fit(X_train, y_train)

# We can check how well the model performs by evaluating it on the test data.
rsf.score(X_test, y_test)

# Predicting ===========================================================================================================

# For prediction, a sample is dropped down each tree in the forest until it reaches a terminal node. Data
# in each terminal is used to non-parametrically estimate the survival and cumulative hazard function using the
# Kaplan-Meier and Nelson-Aalen estimator, respectively. In addition, a risk score can be computed that represents
# the expected number of events for one particular terminal node. The ensemble prediction is simply the average
# across all trees in the forest.

# Let’s first select a couple of patients from the test data according to the number of positive lymph nodes and age.
X_test_sorted = X_test.sort_values(by=["pnodes", "age"])
X_test_sel = pd.concat((X_test_sorted.head(3), X_test_sorted.tail(3)))

# The predicted risk scores indicate that risk for the last three patients is quite a bit higher than that of the
# first three patients.
pd.Series(rsf.predict(X_test_sel))

# We can have a more detailed insight by considering the predicted survival function. It shows that the biggest
# difference occurs roughly within the first 750 days.
surv = rsf.predict_survival_function(X_test_sel, return_array=True)

for i, s in enumerate(surv):
    plt.step(rsf.event_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)
plt.show()

# Alternatively, we can also plot the predicted cumulative hazard function.
surv = rsf.predict_cumulative_hazard_function(X_test_sel, return_array=True)

for i, s in enumerate(surv):
    plt.step(rsf.event_times_, s, where="post", label=str(i))
plt.ylabel("Cumulative hazard")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)
plt.show()

# Permutation-based Feature Importance =================================================================================

# The implementation is based on scikit-learn’s Random Forest implementation and
# inherits many features, such as building trees in parallel. What’s currently missing is feature importance via the
# feature_importance_ attribute. This is due to the way scikit-learn’s implementation computes importance. It relies
# on a measure of impurity for each child node, and defines importance as the amount of decrease in impurity due to a
# split. For traditional regression, impurity would be measured by the variance, but for survival analysis there is
# no per-node impurity measure due to censoring. Instead, one could use the magnitude of the log-rank test statistic
# as an importance measure, but scikit-learn’s implementation does not seem to allow this.

# Fortunately, this is not a big concern though, as scikit-learn’s definition of feature importance is non-standard
# and differs from what Leo Breiman proposed in the original Random Forest paper. Instead, we can use permutation to
# estimate feature importance, which is preferred over scikit-learn’s definition. This is implemented in the ELI5
# library, which is fully compatible with scikit-survival.
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rsf, n_iter=15, random_state=random_state)
perm.fit(X_test, y_test)
eli5.show_weights(perm, feature_names=Xt.columns.tolist())