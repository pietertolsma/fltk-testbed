import copy
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import *

train = pd.read_csv("D:/logging/results/data_train.csv")
test = pd.read_csv("D:/logging/results/data_test.csv")

train_y = train.pop('accuracy_mean')
train_X = train
test_y = test.pop('accuracy_mean')
test_X = test

model_selection = False
if model_selection:
    models = {"RandomForest": [RandomForestRegressor(), [{"n_estimators": [10, 20, 50, 100, 200, 500]}]],
              "GBDT": [GradientBoostingRegressor(), [{"n_estimators": [10, 20, 50, 100, 200, 500],
                                                      "learning_rate": [1, 0.1, 0.01]}]],
              "SVM": [SVR(), [{'kernel': ['linear', 'poly', 'rbf'], 'C': [0.1, 1, 10]}]]}

    for k, v in models.items():
        model = GridSearchCV(v[0], v[1], scoring='neg_mean_squared_error', cv=5)
        model.fit(train_X, train_y)

        print(k)
        print(model.best_params_)
        pred = model.predict(test_X)
        print(mean_squared_error(pred, test_y))

lg = LinearRegression()
rf = RandomForestRegressor(n_estimators=200, random_state=42)
gbdt = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, random_state=42)
svm = SVR(kernel='rbf', C=10)

for model in [lg, rf, gbdt, svm]:
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    print(mean_squared_error(pred, test_y))


train = pd.read_csv("D:/logging/results/data_train.csv")
test = pd.read_csv("D:/logging/results/data_test.csv")

df = pd.concat([train, test], ignore_index=True)
df2 = copy.deepcopy(df)
all_y = df.pop('accuracy_mean')
all_X = df


# for t in range(1, 5):
#     remain = df2.loc[lambda x: x['time_level'] == t, :]['accuracy_mean']
#     print(len(remain))
#     print(np.min(remain))
#     print(np.max(remain))
#     print(np.mean(remain))
#     print(np.std(remain))

groups = df.groupby(["layers", "nodes", "cores"])
group_num = np.zeros((len(df),), dtype=np.int64)
for i, (key, group) in enumerate(groups):
    for idx in group.axes[0]:
        group_num[idx] = i


for model in [lg, rf, gbdt, svm]:
    scores = cross_val_score(model, all_X, all_y, groups=group_num,
                             scoring='neg_mean_squared_error', cv=LeaveOneGroupOut())
    print(-np.mean(scores))


logo = LeaveOneGroupOut()
for model in [rf, gbdt, svm]:
    res = []
    cnt = 0
    cnt_t = 0
    y_pred_labels = []
    y_true_labels = []
    for train_index, test_index in logo.split(all_X, all_y, group_num):
        X_train, X_test = all_X.iloc[train_index], all_X.iloc[test_index]
        y_train, y_test = all_y.iloc[train_index], all_y.iloc[test_index]
        all_test = df2.iloc[test_index]
        model.fit(X_train, y_train)
        d = {}
        for t in range(1, 5):
            for bs in range(1, 4):
                for lr in range(1, 4):
                    tpl = (t, X_test.iloc[0, 1], X_test.iloc[0, 2], X_test.iloc[0, 3], bs, lr)
                    score = model.predict([tpl])
                    d[tpl] = score[0]
        for tmax in range(1, 5):
            remain = all_test.loc[lambda x: x['time_level'] <= tmax, :]  #
            remain_acc = remain['accuracy_mean']
            remain_avg = np.mean(remain_acc)
            remain_max = np.max(remain_acc)
            pred_list = sorted([(v, k) for k, v in d.items() if k[0] <= tmax], reverse=True)  #
            pred_list2 = sorted([(v, k) for k, v in d.items() if k[0] == tmax], reverse=True)
            pred_max = pred_list[0]
            pred_max2 = pred_list2[0]
            pred_max_act = all_test.loc[lambda x: (x['time_level'] == pred_max[1][0]) &
                                                  (x['batch_size'] == pred_max[1][4]) &
                                                  (x['learning_rate'] == pred_max[1][5]), :].iloc[0, 6]
            pred_max_act2 = all_test.loc[lambda x: (x['time_level'] == pred_max2[1][0]) &
                                                   (x['batch_size'] == pred_max2[1][4]) &
                                                   (x['learning_rate'] == pred_max2[1][5]), :].iloc[0, 6]
            if pred_max_act > pred_max_act2:
                cnt_t += 1

            fix_max_act = all_test.loc[lambda x: (x['time_level'] == tmax) &
                                                 (x['batch_size'] == 1) &
                                                 (x['learning_rate'] == 1), :].iloc[0, 6]

            res.append([fix_max_act, remain_max, pred_max_act])
            if pred_max_act >= fix_max_act:
                cnt += 1
            # y_pred_labels.append(3 * (pred_max_act[1][4] - 1) + pred_max_act[1][5] - 1)
            # y_true_labels.append(3 * (pred_max_act[1][4] - 1) + pred_max_act[1][5] - 1)
    print(np.mean(np.array(res), axis=0))
    print(cnt / len(res))
    print(cnt_t / len(res))
    # print(confusion_matrix(y_pred_labels, y_true_labels))
    # print(sum([line[1] - line[2] for line in res]) / len(res), sum([line[2] - line[0] for line in res]) / len(res))





