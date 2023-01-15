import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import sys

def get_auc(Y,y_score,classes):
    y_test_binarize=np.array([[1 if i ==c else 0 for c in classes] for i in Y])
    fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    return auc(fpr, tpr)





def analysis_tree(new_tree, original_RF, test_X, test_Y):
    new_tree_intervals_n = new_tree.tree_.n_leaves
    original_RF_class = original_RF.predict(test_X)
    new_tree_class = new_tree.predict(test_X)
    original_label = test_Y

    original_RF_acc = accuracy_score(original_label, original_RF_class)
    new_tree_acc = accuracy_score(original_label, new_tree_class)

    faithfulness = accuracy_score(original_RF_class, new_tree_class)
    # 防止产生单类别树时auc无法计算
    if new_tree.n_classes_ == 1:
        curr_pre_proba = new_tree.predict_proba(test_X)
        new_tree_pre_proba = np.zeros((test_X.shape[0], original_RF.n_classes_))
        j = np.where(original_RF.classes_ == new_tree.classes_)[0]
        new_tree_pre_proba[:, j] = curr_pre_proba
    else:
        curr_pre_proba = new_tree.predict_proba(test_X)
        new_tree_pre_proba = np.zeros((test_X.shape[0], original_RF.n_classes_))
        # print(curr_pre_proba.shape, new_tree_pre_proba.shape)
        for i in range(new_tree.n_classes_):
            j = np.where(original_RF.classes_ == new_tree.classes_[i])[0][0]
            new_tree_pre_proba[:, j] = curr_pre_proba[:, i]
        # print(new_tree.classes_, original_RF.classes_)
        # print(curr_pre_proba, new_tree_pre_proba)
        # sys.exit()

            
    AUC_faithfulness = get_auc(original_RF.predict(test_X), new_tree_pre_proba, original_RF.classes_)
    AUC_label = get_auc(test_Y, new_tree_pre_proba, original_RF.classes_)
    return new_tree_intervals_n, faithfulness, new_tree_acc, AUC_faithfulness, AUC_label, original_RF_acc


def road_predict(road, road_proba, data):  # 所有区间都是左开闭的区间

    # mask_feature = ((road[:,:,0] - data) * (road[:,:,1] - data) <= 0) + 0
    mask_feature = np.logical_and(data >= road[:, :, 0], data < road[:, :, 1]) + 0
    mask_road = np.prod(mask_feature, axis=1)
    mask_num = np.where(mask_road == 1)
    proba = np.sum(road_proba[mask_num], axis=0)
    # proba = np.average(road_proba[mask_num], axis=0)
    return proba, mask_road


def predict_for_nan_L2(new_interval, new_proba, data, upper_bound, lower_bound):
    feature_in_interval_mask = np.logical_and((data > new_interval[:, :, 0]),
                                              (data <= new_interval[:, :, 1]))
    feature_not_in_interval_mask = feature_in_interval_mask - 1
    feature_range_matrix = upper_bound - lower_bound

    dis_matrix = np.minimum(np.abs((data - new_interval[:, :, 0])),
                            np.abs(data - new_interval[:, :, 1])) / (feature_range_matrix)

    dis_matrix = np.abs(dis_matrix) * feature_not_in_interval_mask
    dis = np.linalg.norm(dis_matrix, axis=1)
    proba = new_proba[np.argmin(dis)]
    return proba


def analysis_interval(new_interval, new_proba, rest_interval, rest_proba, RF, test_X, test_Y, lower_bound, upper_bound,):
    new_intervals_n = new_interval.shape[0]
    intervals_predict_proba = []
    intervals_predict_class = []
    num_predict_nan = 0
    for i in range(test_X.shape[0]):
        pre_proba, _ = road_predict(new_interval, new_proba, test_X[i])
        # print('路径预测', pre_proba)
        if np.sum(pre_proba) == 0:
            pre_proba = predict_for_nan_L2(new_interval, new_proba, test_X[i], upper_bound, lower_bound)
            # print('无路径匹配预测', pre_proba)
            num_predict_nan += 1
        intervals_predict_proba.append(pre_proba)
        intervals_predict_class.append(RF.classes_[np.argmax(pre_proba)])
    intervals_predict_class = np.array(intervals_predict_class)
    original_RF_class = RF.predict(test_X)
    original_label = test_Y
    original_RF_acc = accuracy_score(original_label, original_RF_class)
    intervals_predict_acc = accuracy_score(original_label, intervals_predict_class)

    faithfulness = accuracy_score(original_RF_class, intervals_predict_class)

    
    AUC_faithfulness = get_auc(RF.predict(test_X), np.array(intervals_predict_proba), RF.classes_)
    AUC_label = get_auc(test_Y, np.array(intervals_predict_proba), RF.classes_)
    print('未覆盖样本测试数量及比例：', num_predict_nan, num_predict_nan/test_X.shape[0])

    return new_intervals_n, faithfulness, intervals_predict_acc, AUC_faithfulness, AUC_label, original_RF_acc




