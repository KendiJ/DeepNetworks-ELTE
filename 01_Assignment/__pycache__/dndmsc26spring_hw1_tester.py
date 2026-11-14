
#
# Automatic tests for Homework#1 in ELTE IK, Deep Network Development course, 2026 spring
#
# Authors: Viktor Varga
#

import copy as copy_module
import os
import urllib
import numpy as np

import torch
import torch.nn as nn

class Tester:

    '''
    Member fields:

        TESTS_DICT: dict{test_name - str: test function - Callable}

        dataset_content: None OR str; dataset loaded into a string

        # STORED DATA FROM PREVIOUS TESTS
        
        self.dataset_noisy: ndarray(n_samples, n_features) fl32
        self.dataset: ndarray(n_samples, n_features) fl32; no NaNs

        # RESULT OF LASTEST PREVIOUS TEST RUNS
        test_results: dict{test_name - str: success - bool}

    '''

    def __init__(self):

        self.TESTS_DICT = {
              'dataset_load': self.__test_dataset_load,
              'dataset_fill_missing': self.__test_dataset_fill_missing,
              'dataset_split': self.__test_dataset_split,
              'reg_iter': self.__test_reg_model_iter,
              'reg_model_architecture': self.__test_reg_model_architecture,
              'reg_model_learning': self.__test_reg_model_learning,
              'cl_iter': self.__test_cl_model_iter,
              'cl_model_architecture': self.__test_cl_model_architecture,
              'cl_model_learning': self.__test_cl_model_learning}

        self.test_results = {k: False for k in self.TESTS_DICT.keys()}
        self.dataset_content = None

        self.dataset_noisy = None
        self.dataset = None

    def get_dataset_content(self):
        '''
        Sets self.dataset_content.
        Returns:
            dataset_content: str
        '''
        if self.dataset_content is None:
            url = "https://nipg12.inf.elte.hu/~vavsaai@nipg.lab/dndmsc26spring/files/student-mat_prep_noisy26hw1.csv"
            ftpstream = urllib.request.urlopen(url)
            self.dataset_content = ftpstream.read().decode('utf-8')
        return self.dataset_content

    def test(self, test_name, *args):
        '''
        Parameters:
            test_name: str
            *args: varargs; the arguments for the selected test
        '''
        if test_name not in self.TESTS_DICT:
            assert False, "Tester error: Invalid test name: " + str(test_name)

        self.test_results[test_name] = False
        test_func = self.TESTS_DICT[test_name]
        test_func(*args)
        self.test_results[test_name] = True    # only executed if no assert happened during test

    def print_all_tests_successful(self):
        if all(list(self.test_results.values())):
            print("\nTester: All tests were successful.")


    # TESTS

    def __test_dataset_load(self, *args):
        '''
        Expected parameters:
            dataset_noisy: ndarray(n_samples, n_features) of float32
        '''
        assert len(args) == 1, "Tester error: __test_dataset_load() expects 1 parameters: dataset_noisy. "
        dataset_noisy, = args

        assert type(dataset_noisy) == np.ndarray, "Tester: 'dataset_noisy' must be a numpy array."
        assert dataset_noisy.dtype == np.float32, "Tester: 'dataset_noisy' array must have float32 data type."
        expected_shape = (395, 33)
        assert dataset_noisy.shape == expected_shape, f"Tester: 'dataset_noisy' array must have a shape of {expected_shape}."

        nanmask = np.isnan(dataset_noisy)
        nanidxs = np.argwhere(nanmask)
        expected_nancount = 252
        expected_nancols = [8,9,11]
        expected_nanpos50_60 = [[85, 8], [85, 11], [89, 8], [90, 8], [92, 8], \
                           [94, 8], [99, 11], [102, 9], [103, 11], [105, 9]]

        assert np.count_nonzero(nanmask) == expected_nancount, "Tester: Incorrect NaN count in 'dataset_noisy' array."
        assert np.all(np.isin(nanidxs[:,1], expected_nancols)), "Tester: NaN values found in incorrect columns of 'dataset_noisy' array."
        assert np.array_equal(nanidxs[50:60], expected_nanpos50_60), "Tester: Incorrect NaN value positions in 'dataset_noisy' array."

        self.dataset_noisy = np.copy(dataset_noisy)

        print("Tester: Dataset loading OK")

    def __test_dataset_fill_missing(self, *args):
        '''
        Expected parameters:
            dataset: ndarray(n_samples, n_features) of float32
        '''
        assert len(args) == 1, "Tester error: __test_dataset_fill_missing() expects 1 parameters: dataset. "
        dataset, = args
        assert self.dataset_noisy is not None, "Tester error: Run tester for task 'A' first."

        assert type(dataset) == np.ndarray, "Tester: 'dataset' must be a numpy array."
        assert dataset.dtype == np.float32, "Tester: 'dataset' array must have float32 data type."
        expected_shape = (395, 36)
        assert dataset.shape == expected_shape, f"Tester: 'dataset' array must have a shape of {expected_shape}."
        assert not np.any(np.isnan(dataset)), "Tester: There should be no NaN values in the 'dataset' array."
        assert np.allclose(np.nansum(self.dataset_noisy), np.sum(dataset[:,:-3]), atol=1e-2), "Tester: Non-NaN values from the 'dataset_noisy' array should be unchanged in the 'dataset' array."

        new_vars = dataset[:,-3:]
        assert np.array_equal(np.unique(new_vars), [0, 1]), "Tester: Incorrect values in new variable columns of the 'dataset' array."
        assert np.allclose(np.sum(new_vars), 933., atol=1e-2), "Tester: Incorrect values in new variable columns of the 'dataset' array."

        self.dataset = np.copy(dataset)

        print("Tester: Dataset fill missing OK")

    def __test_dataset_split(self, *args):
        '''
        Expected parameters:
            dataset_split_train, dataset_split_val, dataset_split_test: ndarray(n_split_samples, n_features) of float32
        '''
        assert len(args) == 3, "Tester error: __test_dataset_split() expects 3 parameters: " +\
                                    "dataset_split_train, dataset_split_val, dataset_split_test. "
        assert self.dataset is not None, "Tester error: Run tester for task 'A' & 'B' first."
        dataset_split_train, dataset_split_val, dataset_split_test = args

        dataset_split_arrs = [dataset_split_train, dataset_split_val, dataset_split_test]
        dataset_split_sizes = {'dataset_split_train': 197, 'dataset_split_val': 99, 'dataset_split_test': 99}
        for dataset_split_arr, dataset_split_size_entry in zip(dataset_split_arrs, dataset_split_sizes.items()):
          dataset_split_name, dataset_split_size = dataset_split_size_entry
          assert type(dataset_split_arr) == np.ndarray, f"Tester: '{dataset_split_name}' must be a numpy array."
          assert dataset_split_arr.dtype == np.float32, f"Tester: '{dataset_split_name}' array must have float32 data type."

          assert dataset_split_arr.ndim == 2, f"Tester: '{dataset_split_name}' array has an incorrect shape."
          assert dataset_split_arr.shape[1] == self.dataset.shape[1], f"Tester: '{dataset_split_name}' array has an incorrect shape."
          assert np.fabs(dataset_split_arr.shape[0] - dataset_split_size) < 4, f"Tester: '{dataset_split_name}' array has an incorrect shape."

        dataset_split_concat = np.concatenate([dataset_split_train, dataset_split_val, dataset_split_test], axis=0)
        assert dataset_split_concat.shape == dataset_split_concat.shape, "Tester: concatenated dataset splits do not match original 'dataset' array shape."
        assert not np.array_equal(dataset_split_concat, self.dataset), "Tester: Samples in dataset were not shuffled randomly."
        sample_sums_concat = np.sort(np.sum(dataset_split_concat, axis=1))
        sample_sums_orig = np.sort(np.sum(self.dataset, axis=1))
        assert np.allclose(sample_sums_concat, sample_sums_orig, atol=1e-2), "Tester: Samples might have been shuffled incorrectly (?)."

        print("Tester: Dataset split OK")

    def __test_reg_model_iter(self, *args):
        '''
        Expected parameters:
            dataloader_reg_train, dataloader_reg_val, dataloader_reg_test: ndarray(n_split_samples, n_features) of float32
        '''
        assert len(args) == 3, "Tester error: __test_reg_model_iter() expects 3 parameters: " +\
                                    "dataloader_reg_train, dataloader_reg_val, dataloader_reg_test. "
        dataloader_reg_train, dataloader_reg_val, dataloader_reg_test = args

        dataloader_iters = [dataloader_reg_train, dataloader_reg_val, dataloader_reg_test]
        dataloader_names = ['dataloader_reg_train', 'dataloader_reg_val', 'dataloader_reg_test']
        for dataloader_iter, dataloader_name in zip(dataloader_iters, dataloader_names):
          dataloader_iter = copy_module.copy(dataloader_iter)
          for r in dataloader_iter:   # torch DataLoader implements only __getitem__(), but not __iter__(), so next() does not work
            break
          assert len(r) == 2, f"Tester: '{dataloader_name}' must return two tensors in each iteration."
          r0, r1 = r
          assert type(r0) == type(r1) == torch.Tensor, f"Tester: '{dataloader_name}' must return tensors."
          assert r0.dtype == r1.dtype == torch.float32, f"Tester: '{dataloader_name}' must return tensors with float32 data type."
          assert r0.ndim == r1.ndim == 2, f"Tester: Tensors returned by '{dataloader_name}' have incorrect shape."
          batch_size, n_feat = r0.shape
          assert r1.shape[0] == batch_size, f"Tester: Tensors returned by '{dataloader_name}' have inconsistent shape."
          assert n_feat == 35, f"Tester: Input feature tensor returned by '{dataloader_name}' have incorrect shape."
          assert r1.shape[1] == 1, f"Tester: Label tensor returned by '{dataloader_name}' have incorrect shape."

        print("Tester: Dataset iterators for regression task OK")


    def __test_reg_model_architecture(self, *args):
        '''
        Expected parameters:
            reg_model: PyTorch nn.Module
        '''
        assert len(args) == 1, "Tester error: __test_reg_model_architecture() expects 1 parameter: reg_model "
        reg_model, = args
        
        assert isinstance(reg_model, torch.nn.Module), f"Tester: The 'reg_model' neural network model must be of a subtype of torch.nn.Module."
        n_params = sum([p.numel() for p in reg_model.parameters(recurse=True) if p.requires_grad])
        assert n_params == (35*25 + 25) + (25*25 + 25) + (25*1 + 1), f"Tester: Incorrect structure in 'reg_model' network (incorrect parameter count)."

        print("Tester: Regression model architecture OK")

    def __test_reg_model_learning(self, *args):
        '''
        Expected parameters:
            test_mse: float
        '''
        assert len(args) == 1, "Tester error: __test_reg_model_learning() expects 1 parameter: test_mse. "
        test_mse, = args
        assert 0.6 < test_mse < 3., "Tester: A well trained regression model should provide a test loss (MSE) between 0.6 and 3."

        print("Tester: Regression model learning OK")

    def __test_cl_model_iter(self, *args):
        '''
        Expected parameters:
            y_cat_train, y_cat_val, y_cat_test: ndarray(n_split_samples) of int
        '''
        assert len(args) == 3, "Tester error: __test_cl_model_iter() expects 3 parameters: " +\
                                    "dataloader_cl_train, dataloader_cl_val, dataloader_cl_test. "
        dataloader_cl_train, dataloader_cl_val, dataloader_cl_test = args

        # test shapes
        dataloader_iters = [dataloader_cl_train, dataloader_cl_val, dataloader_cl_test]
        dataloader_names = ['dataloader_cl_train', 'dataloader_cl_val', 'dataloader_cl_test']
        for dataloader_iter, dataloader_name in zip(dataloader_iters, dataloader_names):
          dataloader_iter = copy_module.copy(dataloader_iter)
          for r in dataloader_iter:   # torch DataLoader implements only __getitem__(), but not __iter__(), so next() does not work
            break
          assert len(r) == 2, f"Tester: '{dataloader_name}' must return two tensors in each iteration."
          r0, r1 = r
          assert type(r0) == type(r1) == torch.Tensor, f"Tester: '{dataloader_name}' must return tensors."
          assert r0.dtype == torch.float32, f"Tester: '{dataloader_name}' must return an input feature tensor with float32 data type."
          assert r1.dtype == torch.int64, f"Tester: '{dataloader_name}' must return a target label tensor with int64 data type."
          assert r0.ndim == 2, f"Tester: Input feature tensor returned by '{dataloader_name}' have incorrect shape."
          assert r1.ndim == 1, f"Tester: Target label tensor returned by '{dataloader_name}' have incorrect shape."
          batch_size, n_feat = r0.shape
          assert r1.shape[0] == batch_size, f"Tester: Tensors returned by '{dataloader_name}' have inconsistent shape."
          assert n_feat == 33, f"Tester: Input feature tensor returned by '{dataloader_name}' have incorrect shape."

        print("Tester: Dataset iterators for classification task OK")

    def __test_cl_model_architecture(self, *args):
        '''
        Expected parameters:
            cl_model: keras.models.Sequential() instance
        '''
        assert len(args) == 1, "Tester error: __test_cl_model_architecture() expects 1 parameter: cl_model "
        cl_model, = args

        assert isinstance(cl_model, torch.nn.Module), f"Tester: The 'cl_model' neural network model must be of a subtype of torch.nn.Module."
        n_params = sum([p.numel() for p in cl_model.parameters(recurse=True) if p.requires_grad])
        assert n_params == (33*30 + 30) + (30*20 + 20) + (20*3 + 3), f"Tester: Incorrect structure in 'cl_model' network (incorrect parameter count)."

        print("Tester: Classification model architecture OK")


    def __test_cl_model_learning(self, *args):
        '''
        Expected parameters:
            test_ce, test_acc: float
        '''
        assert len(args) == 2, "Tester error: __test_cl_model_learning() expects 2 parameters:" +\
                                             " test_ce, test_acc"
        test_ce, test_acc = args
        assert 0.5 < test_ce < 1.2, "Tester: A well trained classification model should provide a test loss (CE) between 0.5 and 1.2"
        assert 0.5 < test_acc < 0.8, "Tester: A well trained classification model should provide a test accuracy between 0.5 and 0.8"

        print("Tester: Classification model learning OK")
        self.print_all_tests_successful()

    #