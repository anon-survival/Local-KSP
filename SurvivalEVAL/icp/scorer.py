from __future__ import division
import argparse
import abc
import numpy as np
import sklearn.base
import torch
import torchtuples as tt
import pandas as pd
import os

# from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
# from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
# from model import CoxPH, MTLR, CQRNN, LogNormalNN
# from pycox.models import DeepHitSingle, CoxTime

from SurvivalEVAL.icp.error_functions import AbsErrorErrFunc, RegressionErrFunc
from SurvivalEVAL.utils import pad_tensor
from SurvivalEVAL.utils.util_survival import make_mono_quantiles, format_pred_sksurv, survival_to_quantile
from SurvivalEVAL.Evaluations.util import check_monotonicity


class BaseScorer(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseScorer, self).__init__()

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def score(self, x, y=None):
        pass


class BaseModelNc(BaseScorer):
    """Base class for nonconformity scorers based on an underlying model.

    Parameters
    ----------
    model : ClassifierAdapter or RegressorAdapter
        Underlying classification model used for calculating nonconformity
        scores.

    err_func : ClassificationErrFunc or RegressionErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.
    """

    def __init__(self, model, err_func, normalizer=None, beta=1e-6):
        super(BaseModelNc, self).__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        # If we use sklearn.base.clone (e.g., during cross-validation),
        # object references get jumbled, so we need to make sure that the
        # normalizer has a reference to the proper model adapter, if applicable.
        if (self.normalizer is not None and
                hasattr(self.normalizer, 'base_model')):
            self.normalizer.base_model = self.model

        self.last_x, self.last_y = None, None
        self.last_prediction = None
        self.clean = False

    def fit(self, x, y):
        """Fits the underlying model of the nonconformity scorer.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for fitting the underlying model.

        y : numpy array of shape [n_samples]
            Outputs of examples for fitting the underlying model.

        Returns
        -------
        None
        """
        self.model.fit(x, y)
        if self.normalizer is not None:
            self.normalizer.fit(x, y)
        self.clean = False

    def score(self, x, y=None):
        """Calculates the nonconformity score of a set of samples.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.

        y : numpy array of shape [n_samples]
            Outputs of examples for which to calculate a nonconformity score.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of samples.
        """
        prediction = self.model.predict(x)
        n_test = x.shape[0]
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)
        if prediction.ndim > 1:
            ret_val = self.err_func.apply(prediction, y)
        else:
            ret_val = self.err_func.apply(prediction, y) / norm
        return ret_val


class RegressorNc(BaseModelNc):
    """Nonconformity scorer using an underlying regression model.

    Parameters
    ----------
    model : RegressorAdapter
        Underlying regression model used for calculating nonconformity scores.

    err_func : RegressionErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.

    Attributes
    ----------
    model : RegressorAdapter
        Underlying model object.

    err_func : RegressionErrFunc
        Scorer function used to calculate nonconformity scores.

    See also
    --------
    ProbEstClassifierNc, NormalizedRegressorNc
    """

    def __init__(self,
                 model,
                 err_func=AbsErrorErrFunc(),
                 normalizer=None,
                 beta=1e-6):
        super(RegressorNc, self).__init__(model,
                                          err_func,
                                          normalizer,
                                          beta)

    def predict(self, x, nc, significance=None):
        """Constructs prediction intervals for a set of test examples.

        Predicts the output of each test pattern using the underlying model,
        and applies the (partial) inverse nonconformity function to each
        prediction, resulting in a prediction interval for each test pattern.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        nc : numpy array of shape [n_calibration_samples]
            Nonconformity scores obtained for conformal predictor.

        significance : float
            Significance level (maximum allowed error rate) of predictions.
            Should be a float between 0 and 1. If ``None``, then intervals for
            all significance levels (0.01, 0.02, ..., 0.99) are output in a
            3d-matrix.

        Returns
        -------
        p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
            If significance is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each test pattern, and each significance
            level (0.01, 0.02, ..., 0.99). If significance is a float between
            0 and 1, then p contains the prediction intervals (minimum and
            maximum	boundaries) for the set of test patterns at the chosen
            significance level.
        """
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], 2))
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            if prediction.ndim > 1:  # CQR
                intervals[:, 0] = prediction[:, 0] - err_dist[0, :]
                intervals[:, 1] = prediction[:, -1] + err_dist[1, :]
            else:  # regular conformal prediction
                err_dist *= norm
                intervals[:, 0] = prediction - err_dist[0, :]
                intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals


class SurvivalNC(sklearn.base.BaseEstimator):
    """Nonconformity scorer using an underlying survival model."""

    def __init__(
            self,
            model,
            error_function=RegressionErrFunc(),
            args=argparse.Namespace
    ):
        super(SurvivalNC, self).__init__()
        self.model = model
        self.err_func = error_function
        self.args = args

    def score(
            self,
            feature_df: pd.DataFrame,
            t: np.ndarray,
            e: np.ndarray,
            quantile_levels: np.ndarray,
            method: str):
        """Calculates the nonconformity score of a set of samples.

        Parameters
        ----------
        feature_df: pandas DataFrame of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.
        t: numpy array of shape [n_samples]
            Times of examples.
        e: numpy array of shape [n_samples]
            Event indicators of examples.
        quantile_levels: numpy array of shape [n_significance_levels]
            Significance levels (maximum allowed error rate) of predictions.
        method: str
            Decensoring method to use. See `compute_decensor_times` in `utils/util_survival.py` for details.
        Returns
        -------
        conformal_scores : numpy array of shape [n_samples]
            conformity scores of samples.
        """
        x = feature_df.values
        x_names = feature_df.columns.tolist()
        y = np.stack([t, e], axis=1)

        quantile_predictions = self.predict_nc(x, quantile_levels, x_names)

        if method == 'sampling':
            quantile_predictions = np.repeat(quantile_predictions, 1000, axis=0)

        assert quantile_predictions.shape[0] == y.shape[0], "Sample size does not match."

        conformal_scores = self.err_func.apply(quantile_predictions, y)
        return conformal_scores

    def predict(
            self,
            x: np.ndarray,
            conformal_scores: np.ndarray,
            feature_names: list[str] = None,
            quantile_levels=None
    ):
        quantile_predictions = self.predict_nc(x, quantile_levels, feature_names)

        error_dist = self.err_func.apply_inverse(conformal_scores, quantile_levels)

        quantile_predictions = quantile_predictions - error_dist
        quantile_levels, quantile_predictions = make_mono_quantiles(quantile_levels, quantile_predictions,
                                                                    method=self.args.mono_method, seed=self.args.seed)
        # sanity checks
        assert np.all(quantile_predictions >= 0), "Quantile predictions contain negative."
        assert check_monotonicity(quantile_predictions), "Quantile predictions are not monotonic."

        return quantile_predictions
