__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

# Python core packages:
from math import sqrt

# External packages:
import pytest
import pandas as pd

# TARGET TEST MODULE:
from astropak import stats


def test_class_mixed_model_fit():
    import numpy as np
    import pandas as pd
    # First, construct test data frame:
    points = 80
    np.random.seed(1234)
    d = {'A': np.random.randn(points),
         'B': np.random.randn(points),
         'C': np.random.randn(points),
         'Ran': np.random.randint(0, 3, points),
         'Dep': 0}
    df = pd.DataFrame(d)
    df['Dep'] = 17 + 1*df.A + 2*df.B + 4*df.C + 5*(df.Ran-1) + 1*np.random.randn(len(df))
    categories = ['X', 'Y', 'Z']
    df['Ran'] = [categories[r] for r in df['Ran']]
    df.index = df.index + 200
    # Split test data into model and test blocks:
    df_model = df[0:int(3*points/4)]
    df_test = df[len(df_model):]

    # Construct fit object:
    fit = stats.MixedModelFit(df_model, dep_var='Dep', fixed_vars=['A', 'B', 'C'], group_var='Ran')

    # Test object and scalar attributes:
    assert isinstance(fit, stats.MixedModelFit)
    assert fit.converged is True
    assert fit.nobs == len(df_model)
    assert fit.likelihood == pytest.approx(-95.7673)
    assert fit.dep_var == 'Dep'
    assert fit.fixed_vars == ['A', 'B', 'C']
    assert fit.group_var == 'Ran'
    assert fit.sigma == pytest.approx(1.030723)

    # Test fixed-effect results dataframe:
    assert list(fit.df_fixed_effects.index) == ['Intercept', 'A', 'B', 'C']
    assert list(fit.df_fixed_effects['Name']) == list(fit.df_fixed_effects.index)
    assert list(fit.df_fixed_effects['Value']) == pytest.approx([16.648186, 0.946692,
                                                                 1.959923, 4.069383], abs=0.00001)
    assert list(fit.df_fixed_effects['Stdev']) == pytest.approx([2.844632, 0.142185,
                                                                 0.134386, 0.145358], abs=0.00001)
    assert list(fit.df_fixed_effects['Tvalue']) == pytest.approx([5.8525, 6.65818,
                                                                  14.58429, 27.99568], abs=0.0001)
    assert list(fit.df_fixed_effects['Pvalue'] * 10**9) == pytest.approx([4.8426, 0.027724,
                                                                          0, 0], abs=0.0001)

    # Test random-effect (group) results dataframe:
    assert list(fit.df_random_effects.index) == ['X', 'Y', 'Z']
    assert list(fit.df_random_effects['GroupName']) == list(fit.df_random_effects.index)
    assert list(fit.df_random_effects['Group']) == pytest.approx([-5.164649, 0.543793,
                                                                       4.620857], abs=0.00001)

    # Test observation results dataframe:
    assert list(fit.df_observations['FittedValue'])[0:4] == pytest.approx([24.809899, 10.130408,
                                                                           19.834543, 7.758331],
                                                                          abs=0.00001)
    assert list(fit.df_observations['Residual'])[0:4] == pytest.approx([0.490179, -0.786949,
                                                                        0.58315, -1.23926],
                                                                       abs=0.00001)

    # Verify predictions on model data:
    # Case 1: INCLUDING random effects:
    predictions_with_1 = fit.predict(df_model[0:4], include_random_effect=True)
    predictions_with_2 = fit.predict(df_model[0:4])
    assert list(predictions_with_1) == list(predictions_with_2)  # verify default is inclusion.
    assert list(predictions_with_1) == pytest.approx(list(fit.df_observations['FittedValue'])[0:4])

    # Case 2: OMITTING random effects:
    predictions_without = fit.predict(df_model[0:4], include_random_effect=False)
    random_effect_contributions = pd.Series([fit.df_random_effects.loc[group, 'Group']
                                             for group in df_model.iloc[0:4]['Ran']],
                                            index=predictions_without.index)
    expected_predictions = predictions_with_1 - random_effect_contributions
    assert list(expected_predictions) == pytest.approx(list(predictions_without))


def test_weighted_mean():
    with pytest.raises(ValueError) as e:
        stats.weighted_mean([], [])  # zero-length
    assert 'lengths of values & weights must be equal & non-zero' in str(e)
    with pytest.raises(ValueError) as e:
        stats.weighted_mean([2, 3], [4, 5, 3])  # unequal lengths
    assert 'lengths of values & weights must be equal & non-zero' in str(e)
    with pytest.raises(ValueError) as e:
        stats.weighted_mean([2, 3, 4], [1, 4, -5])  # sum(weights)==0
    assert 'sum of weights must be positive' in str(e)
    assert stats.weighted_mean([1, 3, 8], [0, 3, 9]) == (81 / 12, pytest.approx(3.535533),
                                                        pytest.approx(2.795085))
    assert stats.weighted_mean([1, 3, 8], [0, 3, 9]) == stats.weighted_mean([3, 8], [3, 9])
    value_series = pd.Series([1, 3, 8], index=[4, 2, 999])
    weights_series = pd.Series([0, 3, 9], index=['e', 'XXX', '0-&&'])
    assert stats.weighted_mean(value_series, weights_series) == \
           stats.weighted_mean([1, 3, 8], [0, 3, 9])
    assert stats.weighted_mean([-2, -1, 0, 1, 2], [1, 1, 1, 1, 1]) == \
           (pytest.approx(0, abs=0.000001),
            pytest.approx(sqrt(2.5)), pytest.approx(sqrt(0.5)))
    assert stats.weighted_mean([1, 2, 5, 11], [0, 0, 3, 0]) == (5, 0, 0)  # only 1 nonzero weight