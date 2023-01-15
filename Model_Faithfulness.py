import numpy as np
from Interval_IOU import IoU
from RF_to_interval import _parse_forest_interval_with_flux

import sys


def model_faithfulness(interval_set, proba_set, RF, whole_interval):
    tree_num = RF.n_estimators
    all_interval, all_proba, _ = _parse_forest_interval_with_flux(RF, left_bounds=whole_interval[:, 0],
                                                                  right_bounds=whole_interval[:, 1])

    conflict = np.zeros(RF.n_classes_)
    for k in range(len(interval_set)):
        interval, proba = interval_set[k], proba_set[k]
        axis_intersect_rate_matrix = np.zeros(all_interval.shape[0])
        axis_area = IoU(whole_interval, interval).intersect_rate_for_interval1()
        for i in range(all_interval.shape[0]):
            axis_intersect_rate_matrix[i] = IoU(interval,
                                                all_interval[i]).intersect_rate_for_interval1()  # 交集面积相对比
        axis_intersect_area_matrix = axis_intersect_rate_matrix * axis_area  # 交集面积绝对大小
        for curr_class in range(RF.n_classes_):
            proba_diff = proba[curr_class] - all_proba[:, 0, curr_class]
            conflict[curr_class] = conflict[curr_class] + np.abs(np.sum(axis_intersect_area_matrix * proba_diff) / tree_num)
        # print(conflict)

    # print(conflict)
    faithfulness = 1 - (conflict)
    return faithfulness

def model_faithfulness_class(interval_set, proba_set, RF, whole_interval):
    tree_num = RF.n_estimators
    all_interval, all_proba, _ = _parse_forest_interval_with_flux(RF, left_bounds=whole_interval[:, 0],
                                                                  right_bounds=whole_interval[:, 1])

    conflict = 0
    for k in range(len(interval_set)):
        interval, proba = interval_set[k], proba_set[k]
        proba_class = np.argmax(proba)
        # print(proba_class)
        axis_intersect_rate_matrix = np.zeros(all_interval.shape[0])
        axis_area = IoU(whole_interval, interval).intersect_rate_for_interval1()
        for i in range(all_interval.shape[0]):
            axis_intersect_rate_matrix[i] = IoU(interval,
                                                all_interval[i]).intersect_rate_for_interval1()  # 交集面积相对比
        axis_intersect_area_matrix = axis_intersect_rate_matrix * axis_area  # 交集面积绝对大小

        proba_diff = 1 - all_proba[:, 0, proba_class]
        conflict = conflict + np.abs(np.sum(axis_intersect_area_matrix * proba_diff) / tree_num)
        # print(conflict)

    # print(conflict)
    faithfulness = 1 - (conflict)
    return faithfulness

if __name__ == '__main__':
    from build_RF_model import RF_build
    import random
    from RF_to_interval import _parse_forest_interval_with_flux
    from all_region_sample_tree import all_region_sample
    from exp_gini_interval_build_tree import *

    np.random.seed(1)
    random.seed(1)
    dataset_list = ['iris', 'aust_credit', 'banknote', 'haberman', 'mammographic',
                    'breast_cancer', 'compas', 'shop', 'heloc', 'wine_data', '2D_data', 'liver', 'bank', '2D_data_sin']
    dataset_T = ['2D_data_sin']
    ana_dic = {}
    ana_fixed_dic = {}
    for dataset_name in dataset_T:
        print(dataset_name)
        RF_init = RF_build(dataset_name)
        RF = RF_init.build_RF_model()
        test_X = RF_init.test_X
        test_Y = RF_init.test_Y
        val_X = RF_init.val_X
        val_Y = RF_init.val_Y
        lower_bounds = RF_init.data_lower_bounds
        upper_bounds = RF_init.data_upper_bounds
        interval_lower_bound = lower_bounds - 0.01 * (upper_bounds - lower_bounds)
        interval_upper_bound = upper_bounds + 0.01 * (upper_bounds - lower_bounds)
        whole_interval = np.array([interval_lower_bound, interval_upper_bound]).T
        interval, proba, flux = _parse_forest_interval_with_flux(RF, left_bounds=interval_lower_bound,
                                                                 right_bounds=interval_upper_bound)
        random_test_X = all_region_sample(lower_bounds, upper_bounds, num=50)

        # gini_switch = 'proba and flux gini'
        gini_switch = 'proba gini'
        # gini_switch = 'proba and tree test flux gini'

        test_flux_data = val_X

        max_depth = 11

        SPS = interval_to_split_point_set(interval)
        NEW_TREE = Node(max_depth, SPS, whole_interval, interval, proba, flux, whole_interval, test_flux_data,
                        RF.n_estimators,
                        gini_switch=gini_switch, ablation_switch=False)
        NEW_TREE.split(0)
        RESULT = test_New_tree(NEW_TREE, RF, test_X, test_Y, random_test_X)
        A = model_faithfulness_class(RESULT['New tree interval'], RESULT['New tree interval proba'], RF, whole_interval)