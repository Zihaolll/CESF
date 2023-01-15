import numpy as np
from Interval_IOU import IoU, IoU_set
import sys
from calculate_interval_flux import interval_flux
from Model_Faithfulness import model_faithfulness_class
import random
import time
def interval_to_split_point_set(interval):  # 选取候选区间中的所有可能切分点，同时已经去除可能的边界点
    I = interval
    feature_n = interval.shape[1]
    SPS = []
    for i in range(feature_n):
        point_set = np.unique(I[:, i, :])
        point_set = np.sort(point_set)[1:-2]
        for value in point_set:
            SPS.append([i, value])
    return SPS


def exp_gini_calculate(interval, true_interval, true_proba, true_flux, whole_interval, tree_num=50,
                       switch='proba gini', add_flux_weight=None):
    all_interval = true_interval.copy()
    all_proba = true_proba.copy()
    all_flux = true_flux.copy()
    all_flux_rate = all_flux / max(all_flux)
    # avg_exp_conflict = np.ones(true_proba.shape[2])
    interval_proba = np.ones(true_proba.shape[2])

    for curr_class in range(true_proba.shape[2]):
        # axis_area = IoU(whole_interval, interval).intersect_rate_for_interval1() + 10**(-(interval.shape[1]+5))
        axis_area = IoU(whole_interval, interval).intersect_rate_for_interval1()
        axis_intersect_rate_matrix = IoU_set(interval, all_interval).intersect_rate_for_interval1()  # 交集面积相对比

        # for i in range(all_interval.shape[0]):
        #     axis_intersect_rate_matrix[i] = IoU(interval,
        #                                         all_interval[i]).intersect_rate_for_interval1()  # 交集面积相对比
        #     test_flux_rate_matrix[i] = interval_flux(all_interval[i], flux_test_data) / flux_test_data.shape[
        #         0]  # 通量测试集下通量比例
        axis_intersect_area_matrix = axis_intersect_rate_matrix * axis_area  # 交集面积绝对大小

        if switch == 'uniform gini':
            weight = 1

        elif switch == 'proba gini':
            weight = all_proba[:, 0, curr_class]

        elif switch == 'proba and flux gini':  # 利用训练集原始子树的通量
            weight = (all_proba[:, 0, curr_class]) * all_flux_rate[:]

        elif switch == 'proba and tree add flux gini':  # 利用额外测试的子树的通量
            if add_flux_weight is not None:
                weight = (all_proba[:, 0, curr_class]) * add_flux_weight[:]

        # weight_exp_conflict = axis_intersect_area_matrix * weight
        interval_proba[curr_class] = np.sum(axis_intersect_area_matrix * weight) / (tree_num * axis_area)

        # print(avg_exp_conflict / axis_area)
        # print(avg_exp_conflict)
        # sys.exit()

    avg_exp_conflict = (1 - interval_proba) * axis_area
    predict_class = np.argmax(interval_proba)
    predict_proba = interval_proba
    if 'flux' in switch:
        predict_proba = interval_proba/np.sum(interval_proba)


    # conflict_ratio = avg_exp_conflict / axis_area
    # predict_proba = np.exp(1 - conflict_ratio) / np.sum(np.exp(1 - conflict_ratio)) #  softmax
    # predict_proba = (1 - conflict_ratio) / np.sum((1 - conflict_ratio))
    # print(predict_class, predict_proba, avg_exp_conflict)

    exp_gini = avg_exp_conflict[predict_class]


    return predict_class, predict_proba, exp_gini


class Node:
    def __init__(self, max_depth, split_point_set, curr_interval, truth_interval, truth_proba, truth_flux,
                 whole_interval, add_flux_data, tree_num=50, gini_switch='proba gini', node_num=0, ablation_switch=False):
        random.seed(1)
        np.random.seed(1)
        # print('new node****************')
        self.SPS = split_point_set
        self.curr_interval = curr_interval
        self.truth_interval = truth_interval
        self.truth_proba = truth_proba
        self.truth_flux = truth_flux
        self.add_flux_data = add_flux_data
        self.add_flux_weight = np.array([interval_flux(self.truth_interval[i], add_flux_data) /
                                                  add_flux_data.shape[0] for i in range(self.truth_interval.shape[0])])



        self.max_depth = max_depth
        self.whole_interval = whole_interval
        self.gini_switch = gini_switch
        self.ablation_switch = ablation_switch
        self.tree_num = tree_num

        self.node_num = node_num
        self.predict_class, self.predict_proba, self.curr_gini = exp_gini_calculate(curr_interval, truth_interval, truth_proba, truth_flux,
                                                                whole_interval,
                                                                tree_num=self.tree_num,
                                                                switch=self.gini_switch,
                                                                add_flux_weight=self.add_flux_weight)



        # print(self.predict_class)
        # print(self.curr_interval)

    def split(self, curr_depth):
        self.curr_depth = curr_depth
        if len(self.SPS) == 0:
            self.left = None
            self.right = None
            # print('stop: no split value')
            return
        self.split_information = self.find_split(self.SPS, self.curr_interval, self.ablation_switch)

        self.split_feature = self.split_information['feature id']
        self.split_value = self.split_information['value']
        self.split_gini = self.split_information['split gini']

        # print(self.split_information)

        is_splitable = self.is_splitable()
        if is_splitable == False:
            self.left = None
            self.right = None
            # print('stop')
            return
        # print(self.split_feature + 1, self.split_value)
        left_SPS, right_SPS = self.SPS_update(self.SPS, self.split_feature, self.split_value)
        # print('left')
        self.left = Node(self.max_depth, left_SPS, self.split_information['left interval'],
                         self.truth_interval, self.truth_proba, self.truth_flux, self.whole_interval,
                         self.add_flux_data, self.tree_num, self.gini_switch,
                         (2*self.node_num)+1, self.ablation_switch)
        self.left.split(self.curr_depth + 1)
        # print('right')
        self.right = Node(self.max_depth, right_SPS, self.split_information['right interval'],
                          self.truth_interval, self.truth_proba, self.truth_flux, self.whole_interval,
                          self.add_flux_data, self.tree_num, self.gini_switch,
                          (2*self.node_num)+2, self.ablation_switch)
        self.right.split(self.curr_depth + 1)

    def sample_split(self, SPS, sample_num=10):
        SPS = np.array(SPS)
        np.random.shuffle(SPS)
        num_max = SPS.shape[0]
        num = np.min((sample_num, num_max))
        # print(sample_num, num_max, num)
        SPS_sample = SPS[:num].tolist()
        return SPS_sample

    def split_calculate(self, feature_id, value, interval, truth_interval, truth_proba, truth_flux):
        split_information = {}

        left_interval = interval.copy()
        right_interval = interval.copy()

        left_interval[feature_id][1] = value
        right_interval[feature_id][0] = value

        left_class, left_proba, left_gini = exp_gini_calculate(left_interval, truth_interval, truth_proba, truth_flux,
                                                   self.whole_interval,
                                                   tree_num=self.tree_num,
                                                   switch=self.gini_switch,
                                                   add_flux_weight=self.add_flux_weight)
        right_class, right_proba, right_gini = exp_gini_calculate(right_interval, truth_interval, truth_proba, truth_flux,
                                                     self.whole_interval,
                                                     tree_num=self.tree_num,
                                                     switch=self.gini_switch,
                                                     add_flux_weight=self.add_flux_weight)


        # left_area_ratio = IoU(interval, left_interval).intersect_rate_for_interval1()
        # right_area_ratio = IoU(interval, right_interval).intersect_rate_for_interval1()

        # split_gini = right_area_ratio * left_gini + left_area_ratio * right_gini
        # 切分系数直接加和会导致小面积偏好，因此在切分时利用剩余面积对当前侧gini加权


        split_gini = left_gini + right_gini
        # print("split_gini", split_gini)


        split_information['split gini'] = split_gini
        split_information['feature id'] = feature_id
        split_information['value'] = value
        split_information['left interval'] = left_interval
        split_information['left class'] = left_class
        split_information['right interval'] = right_interval
        split_information['right class'] = right_class
        return split_information

    def find_split(self, SPS, interval, ablation_switch=False):  # 消融实验部分,True时是随机选，False是按照expgini选
        if ablation_switch==True:
            SPS_sample = self.sample_split(SPS)
            curr_SPS = SPS_sample[0]
            feature_id = int(curr_SPS[0])
            value = curr_SPS[1]
            curr_split_information = self.split_calculate(feature_id, value, interval, self.truth_interval, self.truth_proba,
                                                     self.truth_flux)
        elif ablation_switch==False:
            SPS_sample = self.sample_split(SPS)
            curr_split_information = None
            curr_exp_gini = np.inf  # 初始切分gini设为inf
            for curr_SPS in SPS_sample:
                feature_id = int(curr_SPS[0])
                value = curr_SPS[1]
                split_information = self.split_calculate(feature_id, value, interval, self.truth_interval, self.truth_proba,
                                                         self.truth_flux)

                split_gini = split_information['split gini']
                if split_gini < curr_exp_gini:  # gini越小越好
                    curr_exp_gini = split_gini
                    curr_split_information = split_information


        return curr_split_information


    def SPS_update(self, original_SPS, split_feature, split_value):
        original_SPS = np.array(original_SPS)
        left_SPS = original_SPS.copy()
        right_SPS = original_SPS.copy()

        feature_mask = original_SPS[:, 0] != split_feature
        left_value_mask = original_SPS[:, 1] < split_value
        right_value_mask = original_SPS[:, 1] > split_value
        left_mask = np.logical_or(feature_mask, left_value_mask)
        right_mask = np.logical_or(feature_mask, right_value_mask)

        left_SPS = (left_SPS[left_mask]).tolist()
        right_SPS = (right_SPS[right_mask]).tolist()

        return left_SPS, right_SPS

    def is_splitable(self):  # 终止条件
        if self.curr_depth >= self.max_depth:  # 最大路径树数
            # print('最大路径数')
            return False
        # elif self.curr_gini < self.split_gini:  # gini系数不再降低
        #     print('gini系数不再降低')
        #     return False
        return True

    def predict_and_depth(self, inst):
        if self.left is None and self.right is None:
            return self.predict_class, 1
        if inst[self.split_feature] <= self.split_value:
            # print(self.split_feature, self.split_value, inst[self.split_feature])
            prediction, depth = self.left.predict_and_depth(inst)
            # print('left', depth)
            return prediction, depth + 1
        else:
            prediction, depth = self.right.predict_and_depth(inst)
            # print(self.split_feature, self.split_value, inst[self.split_feature])
            # print('right', depth)
            return prediction, depth + 1

    def number_of_children(self):
        if self.right == None:
            return 1
        return 1 + self.right.number_of_children() + self.left.number_of_children()

    def number_of_leaves_children(self):
        if self.right == None:
            return 1
        return 0 + self.right.number_of_leaves_children() + self.left.number_of_leaves_children()

    def interval_and_predict(self):
        if self.right == None:
            interval = []
            predict_class = []
            predict_proba = []
            # I_l = [self.split_feature, self.split_value, 'l']
            # I_r = [self.split_feature, self.split_value, 'r']
            I_init = np.array(self.whole_interval)
            interval.append(I_init)
            predict_class.append(self.predict_class)
            predict_proba.append(self.predict_proba)
            return interval, predict_class, predict_proba
        else:
            interval = []
            predict_class = []
            predict_proba = []
            left_interval, left_class, left_proba = self.left.interval_and_predict()
            right_interval, right_class, right_proba = self.right.interval_and_predict()
            if (len(left_interval) != len(left_class)) or (len(right_interval) != len(right_class)):
                print("区间与预测数量不等")
                sys.exit()

            for i_l in range(len(left_interval)):
                curr_interval = left_interval[i_l]
                curr_class = left_class[i_l]
                curr_proba = left_proba[i_l]
                curr_value = min(curr_interval[self.split_feature][1], self.split_value)
                curr_interval[self.split_feature][1] = curr_value
                interval.append(curr_interval)
                predict_class.append(curr_class)
                predict_proba.append(curr_proba)

            for i_r in range(len(right_interval)):
                curr_interval = right_interval[i_r]
                curr_class = right_class[i_r]
                curr_proba = right_proba[i_r]
                curr_value = max(curr_interval[self.split_feature][0], self.split_value)
                curr_interval[self.split_feature][0] = curr_value
                interval.append(curr_interval)
                predict_class.append(curr_class)
                predict_proba.append(curr_proba)

            return interval, predict_class, predict_proba



    def pruning(self):
        # print('遍历', self.node_num)
        self.is_last_node = False
        if self.left is None:
            self.is_last_node = True
            return self.is_last_node

        self.left.is_last_node = self.left.pruning()
        self.right.is_last_node = self.right.pruning()

        if self.left.is_last_node and self.right.is_last_node:
            if self.left.predict_class == self.right.predict_class:
                # print('判断2成立')
                # print(self.left.node_num, self.right.node_num, '合并')
                self.left = None
                self.right = None
                self.is_last_node = True
            return self.is_last_node





def test_New_tree(New_tree, orginal_RF, test_X, test_Y, test_X_random):
    result_dic = {}
    test_X_pre_proba = []
    test_X_pre_class = []
    test_X_pre_depth = []
    test_X_pre_proba_random = []
    test_X_pre_class_random = []
    test_X_pre_depth_random = []
    RF = orginal_RF
    for indx in test_X:
        pre_proba, pre_depth = New_tree.predict_and_depth(indx)
        pre_class = RF.classes_[int(pre_proba)]
        test_X_pre_proba.append(pre_proba)
        test_X_pre_class.append(pre_class)
        test_X_pre_depth.append(pre_depth)
    faithfulness = (test_X_pre_class == RF.predict(test_X)).sum() / test_Y.shape[0]
    acc = (test_X_pre_class == test_Y).sum() / test_Y.shape[0]

    for indx in test_X_random:
        pre_proba_random, pre_depth_random = New_tree.predict_and_depth(indx)
        pre_class_random = RF.classes_[int(pre_proba_random)]
        test_X_pre_proba_random.append(pre_proba_random)
        test_X_pre_class_random.append(pre_class_random)
        test_X_pre_depth_random.append(pre_depth_random)
    faithfulness_random = (test_X_pre_class_random == RF.predict(test_X_random)).sum() / test_X_random.shape[0]

    leaf_num = New_tree.number_of_leaves_children()
    New_tree_interval, New_tree_interval_class, New_tree_interval_proba = New_tree.interval_and_predict()
    New_tree_interval_class_name = [RF.classes_[New_tree_interval_class[i]] for i in range(len(New_tree_interval_class))]



    result_dic['faithfulness'] = faithfulness
    result_dic['accuracy'] = acc
    result_dic['faithfulness random'] = faithfulness_random
    result_dic['leaf num'] = leaf_num
    result_dic['New tree interval'] = New_tree_interval
    result_dic['New tree interval proba'] = New_tree_interval_proba
    result_dic['New tree interval class'] = New_tree_interval_class
    result_dic['New tree interval class name'] = New_tree_interval_class_name


    return result_dic










if __name__ == '__main__':
    from build_RF_model import RF_build
    import random
    from RF_to_interval import _parse_forest_interval_with_flux
    # from AAA_NMS_Extract import interval_extract
    # from result_analysis import analysis_tree, analysis_interval
    # from intervals_similarity import intervals_similarity
    # import math
    # import pandas as pd
    from all_region_sample_tree import all_region_sample

    np.random.seed(1)
    random.seed(1)
    dataset_list = ['iris', 'aust_credit', 'banknote', 'haberman', 'mammographic',
                    'breast_cancer', 'compas', 'shop', 'heloc', 'wine_data', '2D_data', 'liver', 'bank', '2D_data_sin']
    dataset_T = ['heloc']
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
        # gini_switch = 'proba gini'
        gini_switch = 'proba and tree add flux gini'

        test_flux_data = val_X


        max_depth = 5

        begin_time = time.time()

        SPS = interval_to_split_point_set(interval)
        NEW_TREE = Node(max_depth, SPS, whole_interval, interval, proba, flux, whole_interval, test_flux_data, RF.n_estimators,
                        gini_switch=gini_switch, ablation_switch=False)
        NEW_TREE.split(0)
        RESULT = test_New_tree(NEW_TREE, RF, test_X, test_Y, random_test_X)
        A = model_faithfulness_class(RESULT['New tree interval'], RESULT['New tree interval proba'], RF, whole_interval)

        end_time = time.time()
        run_time = end_time - begin_time
        print(run_time)



        # SPS_best = interval_to_split_point_set(new_interval)
        # NEW_TREE_best = Node(max_depth, SPS_best, whole_interval, interval, proba, flux, whole_interval, RF.n_estimators,
        #                 gini_switch=gini_switch)
        # NEW_TREE_best.split(0)
        # RESULT_best = test_New_tree(NEW_TREE_best, test_X, test_Y, random_test_X)



