import numpy as np


def interval_flux(interval, dataset):  # 所有区间都是左开闭的区间


    left_in_mask = dataset >= interval[:, 0]
    right_in_mask = dataset < interval[:, 1]
    mask = np.logical_and(left_in_mask, right_in_mask)
    mask_dataset = np.prod(mask, axis=1)
    mask_num = np.sum(mask_dataset)

    # mask_feature = np.logical_and(dataset >= interval[:, :, 0], dataset < interval[:, :, 1]) + 0
    # mask_road = np.prod(mask_feature, axis=1)
    # mask_num = np.where(mask_road == 1)

    return mask_num

def interval_flux(interval, dataset):  # 所有区间都是左开闭的区间


    left_in_mask = dataset >= interval[:, 0]
    right_in_mask = dataset < interval[:, 1]
    mask = np.logical_and(left_in_mask, right_in_mask)
    mask_dataset = np.prod(mask, axis=1)
    mask_num = np.sum(mask_dataset)

    # mask_feature = np.logical_and(dataset >= interval[:, :, 0], dataset < interval[:, :, 1]) + 0
    # mask_road = np.prod(mask_feature, axis=1)
    # mask_num = np.where(mask_road == 1)

    return mask_num



if __name__ == '__main__':
    from build_RF_model import RF_build
    import random
    from RF_to_interval import _parse_forest_interval_with_flux

    from all_region_sample_tree import all_region_sample

    np.random.seed(3)
    random.seed(3)
    dataset_list = ['iris', 'aust_credit', 'banknote', 'haberman', 'mammographic',
                    'breast_cancer', 'compas', 'shop', 'heloc', 'wine_data', '2D_data', 'liver', 'bank']
    dataset_T = ['breast_cancer']
    ana_dic = {}
    ana_fixed_dic = {}
    for dataset_name in dataset_T:
        print(dataset_name)
        RF_init = RF_build(dataset_name)
        RF = RF_init.build_RF_model()
        test_X = RF_init.test_X
        test_Y = RF_init.test_Y
        lower_bounds = RF_init.data_lower_bounds
        upper_bounds = RF_init.data_upper_bounds
        interval_lower_bound = lower_bounds - 0.01 * (upper_bounds - lower_bounds)
        interval_upper_bound = upper_bounds + 0.01 * (upper_bounds - lower_bounds)
        whole_interval = np.array([interval_lower_bound, interval_upper_bound]).T
        interval, proba, flux = _parse_forest_interval_with_flux(RF, left_bounds=interval_lower_bound,
                                                                 right_bounds=interval_upper_bound)
        random_test_X = all_region_sample(lower_bounds, upper_bounds, num_base=100)

        flux_num = interval_flux(whole_interval, random_test_X)


