from build_RF_model import RF_build
import numpy as np
from RF_to_interval import _parse_forest_interval_with_flux, _parse_tree_interval_with_flux
from all_region_sample_tree import all_region_sample
from exp_gini_interval_build_tree import Node, interval_to_split_point_set, test_New_tree
from Model_Faithfulness import model_faithfulness_class




class CESF:
    def __init__(self, dataset_name, max_depth=5, switch='CESF', RF_n=20, RF_d=5):
        self.dataset_name = dataset_name
        self.max_depth = max_depth
        self.switch = switch

        self.RF_init = RF_build(dataset_name, n_estimators=RF_n, max_depth=RF_d)
        self.RF = self.RF_init.build_RF_model()
        self.train_X = self.RF_init.train_X
        self.train_Y = self.RF_init.train_Y
        self.val_X = self.RF_init.val_X
        self.val_Y = self.RF_init.val_Y
        self.test_X = self.RF_init.test_X
        self.test_Y = self.RF_init.test_Y

        self.upper_bound = self.RF_init.data_upper_bounds
        self.lower_bound = self.RF_init.data_lower_bounds
        self.interval_upper_bound = self.upper_bound + 0.01 * (self.upper_bound - self.lower_bound)
        self.interval_lower_bound = self.lower_bound - 0.01 * (self.upper_bound - self.lower_bound)
        self.feature_name = self.RF_init.x_columns

        self.whole_interval = np.array([self.interval_lower_bound, self.interval_upper_bound]).T
        self.interval, self.proba, self.flux = _parse_forest_interval_with_flux(self.RF,
                                                                                left_bounds=self.interval_lower_bound,
                                                                                right_bounds=self.interval_upper_bound)
        if switch=='CESF':
            self.gini_switch = 'proba gini'
        elif switch=='P-CESF':
            self.gini_switch = 'proba and tree add flux gini'

        self.random_test_X = all_region_sample(self.lower_bound, self.upper_bound, num=self.test_X.shape[0])

        self.RESULT_DIC = {}
        self.RESULT_DIC['Dataset name'] = dataset_name
        self.RESULT_DIC['RF Acc'] = self.RF.score(self.test_X, self.test_Y)
        self.RESULT_DIC['RF NumR'] = self.interval.shape[0]


    def run(self):
        curr_dic = {}
        SPS = interval_to_split_point_set(self.interval)
        NEW_TREE = Node(self.max_depth, SPS, self.whole_interval, self.interval, self.proba, self.flux,
                        self.whole_interval, self.val_X, self.RF.n_estimators,
                        gini_switch=self.gini_switch)
        NEW_TREE.split(0)
        NEW_TREE.pruning()
        RESULT = test_New_tree(NEW_TREE, self.RF, self.test_X, self.test_Y, self.random_test_X)
        new_interval = RESULT['New tree interval']
        new_proba = RESULT['New tree interval proba']
        model_faithfulness = model_faithfulness_class(new_interval, new_proba, self.RF, self.whole_interval)

        curr_dic['NumR'] = RESULT['leaf num']
        curr_dic['FModel'] = model_faithfulness
        curr_dic['FXtest'] = RESULT['faithfulness']
        curr_dic['FXRtest'] = RESULT['faithfulness random']
        curr_dic['Acc'] = RESULT['accuracy']
        self.RESULT_DIC.update(curr_dic)
        return self.RESULT_DIC
if __name__ == '__main__':
    Datasets = ['2D_data_sin', 'abalone', 'banknote', 'breast_cancer', 'ecoli', 'haberman', 'heloc', 'iris']
    CESF = CESF('2D_data_sin', 5)
    result = CESF.run()
