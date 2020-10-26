__author__ = "Eric Dose :: New Mexico Mira Project, Albuquerque"

""" Module 'astrosupport.stats'.
    Numerous statistics and regression classes and functions.
    Fork of (extract of) photrix.util, begun 2020-10-23.
    Intentions: (1) a separate, importable module for use by all EVD astro python projects.
                (2) freely forkable & useful to the astro python global community. 
    See test file test/test_stats.py for usage.
"""

# Python core packages:
from math import sqrt

# External packages:
import pandas as pd
import statsmodels.regression.mixed_linear_model as sm  # NB: only statsmodels version >= 0.8


_____REGRESSION_____________________________________________ = 0


class MixedModelFit:
    """ Object: holds info for one mixed-model (py::statsmodel) fit.  TESTS OK 2020-10-25.
        Generic in nature--NOT tied to astronomical usage.
        Uses formula form, i.e., statsmodel::sm.MixedLM.from_formula()

        Usage: fit = MixedModel(df_input, 'Y', ['X1', 'X2'], 'a_group_type']
               fit = MixedModel(df_input, 'Y', 'X1', 'a_group_type'] (OK if only one indep var)

        Fields available from MixedModelFit object:
            .converged: True iff mixed model regression converged, else False. [boolean]
            .dep_var: name of dependent variable. [string]
            .df_fixed_effects, one row per fixed effect & intercept [pandas Dataframe]
                columns:
                    (index): 'Intercept' or name of fixed effect (independent variable)
                    Name: same as (index)
                    Value: best value (coefficient) of indep variable from fit
                    Stdev: std deviation of Value  from fit
                    Tvalue: Value / Stdev from fit
                    Pvalue: [ignore]
            .df_observations, one row per observation used in fit:
                columns:
                    (index):
                    FittedValue: value predicted from fit for observation
                    Residual: obs value - fitted value for observation
            .df_random_effects: one row per effect group (for photometry, usually one row per image):
                columns:
                    (index): ID of group (for photometry, usually imageID = FITS file name)
                    GroupName: same as index.
                    Group ['GroupValue' previously but now obsolete]: random effect for this group
                        (for photometry, usually 'cirrus effect', the image's average intensity over targets
                        minus the average of these values over all targets)
            .fixed_vars: list of fixed variable names, does not include 'Intercept' [list of strings]
            .likelihood: ?
            .nobs: count of observations [int]
            .sigma: std deviation of residuals over all observations [float]
            .statsmodels_object: embedded MixedMLResults object from statsmodels.
                Needed for making predictions using .predict(), not too directly useful to user.
    """
    # TODO: Replace internal 'formula' API with column-name API, which is much more forgiving of var names.
    def __init__(self, data, dep_var=None, fixed_vars=None, group_var=None):
        """ Executes mixed-model fit & makes data available.
        :param data: input data, one variable per column, one point per row. [pandas Dataframe]
        :param dep_var: one column name as dependent 'Y' variable. [string]
        :param fixed_vars: one or more column names as independent 'X' variable. [string or
                  list of strings]
        :param group_var: one column name as group. (category; random-effect) variable [string]
        """
        if not isinstance(data, pd.DataFrame):
            print('Parameter \'data\' must be a pandas Dataframe of input data.')
            return
        if dep_var is None or fixed_vars is None or group_var is None:
            print('Provide all parameters: dep_var, fixed_vars, and group_var.')
            return
        if not isinstance(dep_var, str) or not isinstance(group_var, str):
            print('Parameters \'dep_var\' and \'group_var\' must both be strings.')
            return
        fixed_vars_valid = False  # default if not validated
        if isinstance(fixed_vars, str):
            fixed_vars = list(fixed_vars)
            fixed_vars_valid = True
        if isinstance(fixed_vars, list):
            if len(fixed_vars) >= 1:
                if all([isinstance(var, str) for var in fixed_vars]):
                    fixed_vars_valid = True
        if not fixed_vars_valid:
            print('Parameter \'fixed_vars\' must be a string or a list of strings.')
            return
        formula = dep_var + ' ~ ' + ' + '.join(fixed_vars)

        model = sm.MixedLM.from_formula(formula, groups=data[group_var], data=data)
        fit = model.fit()

        self.statsmodels_object = fit  # instance of class MixedLMResults (py pkg statsmodels)

        # Scalar and naming attributes:
        self.converged = fit.converged  # bool
        self.nobs = fit.nobs  # number of observations used in fit
        self.likelihood = fit.llf
        self.dep_var = dep_var
        self.fixed_vars = fixed_vars
        self.group_var = group_var
        self.sigma = sqrt(sum(fit.resid**2)/(fit.nobs-len(fixed_vars)-2))

        # Fixed-effects dataframe (joins so we don't count on consistent input ordering):
        df = pd.DataFrame({'Value': fit.fe_params})
        df = df.join(pd.DataFrame({'Stdev': fit.bse_fe}))     # join on index (enforce consistency)
        df = df.join(pd.DataFrame({'Tvalue': fit.tvalues}))   # " & any random effect discarded
        df = df.join(pd.DataFrame({'Pvalue': fit.pvalues}))   # " & "
        df['Name'] = df.index
        self.df_fixed_effects = df.copy()

        # Random-effect dataframe, index=GroupName, cols=GroupName, GroupValue:
        df = pd.DataFrame(fit.random_effects).transpose()  # DataFrame, 1 row/group
        df = df.rename(columns={'groups': 'Group'})  # was 'GroupValue'
        df['GroupName'] = df.index
        self.df_random_effects = df.copy()

        # Observation dataframe (safe to count on consistent input ordering -> easier construction):
        df = pd.DataFrame({'FittedValue': fit.fittedvalues})
        df['Residual'] = fit.resid
        self.df_observations = df.copy()

    def predict(self, df_predict_input, include_random_effect=True):
        """ Takes new_data and renders predicted dependent-variable values.
            Optionally includes effect of groups (random effects), unlike py::statsmodels.
        :param: new_data: new input data used to render predictions.
           Extra (unused) columns OK; model selects only needed columns. [pandas DataFrame]
        :param: include_random_effect: True to include them, False to omit/ignore [bool]
        :return: predictions of dependent-variable values matching rows of new data (pandas Series)
        """

        # Get predicted values on fixed effects only (per statsmodels' weird def. of 'predicted'):
        fixed_effect_inputs = df_predict_input[self.fixed_vars]  # 1 col per fixed effect variable
        predicted_on_fixed_only = self.statsmodels_object.predict(exog=fixed_effect_inputs)

        # If requested, add RE contibs (that were not included in MixedModels object 'fit'):
        if include_random_effect:
            df_random_effect_inputs = pd.DataFrame(df_predict_input[self.group_var])
            df_random_effect_values = self.df_random_effects[['Group']]  # was ['GroupValue']
            predicted_on_random_only = pd.merge(df_random_effect_inputs, df_random_effect_values,
                                                left_on=self.group_var,
                                                right_index=True, how='left',
                                                sort=False)['Group']  # was 'GroupValue'
            total_prediction = predicted_on_fixed_only + predicted_on_random_only
        else:
            total_prediction = predicted_on_fixed_only

        return total_prediction


class LinearFit:
    """ Object: holds info for one ordinary multivariate least squares fit.
    Generic in nature--not tied to astronomical usage.
    Internally uses column-name API to statsmodels OLS.
    """
    def __init__(self, data, dep_var=None, indep_vars=None):
        """ Executes ordinary least-squares multivariate linear fit, makes data available.
        :param data: input data, one variable per column, one point per row. [pandas Dataframe]
        :param dep_var: one column name as dependent 'Y' variable. [string]
        :param indep_vars: one or more column names as independent 'X' variable. [string or
                  list of strings]
        """
        if not isinstance(data, pd.DataFrame):
            print('Parameter \'data\' must be a pandas Dataframe of input data.')
            return
        if dep_var is None or indep_vars is None:
            print('Provide parameters: dep_var and indep_vars.')
            return
        if not isinstance(dep_var, str):
            print('Parameter \'dep_var\' must be a string.')
            return





_____STATISTICAL_FUNCTIONS__________________________________ = 0


def weighted_mean(values, weights):
    """  Returns weighted mean, weighted std deviation of values, and weighted std deviation of the mean.
    TESTS OK 2020-10-25.
    :param values: list (or other iterable) of values to be averaged
    :param weights: list (or other iterable) of weights; length must = length of values
    :return: 3-tuple (weighted mean, weighted std dev (population), weighted std dev of mean)
    """
    if (len(values) != len(weights)) or (len(values) == 0) or (len(weights) == 0):
        raise ValueError('lengths of values & weights must be equal & non-zero.')
    if sum(weights) <= 0:
        raise ValueError('sum of weights must be positive.')
    value_list = list(values)    # py list comprehension often misunderstands pandas Series indices.
    weight_list = list(weights)  # "
    norm_weights = [wt/sum(weights) for wt in weight_list]
    w_mean = sum([nwt * val for (nwt, val) in zip(norm_weights, value_list)])
    n_nonzero_weights = sum([w != 0 for w in weight_list])

    if n_nonzero_weights == 1:
        w_stdev_pop = 0
        w_stdev_w_mean = 0
    else:
        resid2 = [(val-w_mean)**2 for val in value_list]
        nwt2 = sum([nwt**2 for nwt in norm_weights])
        rel_factor = 1.0 / (1.0 - nwt2)  # reliability factor (better than N'/(N'-1))
        w_stdev_pop = sqrt(rel_factor * sum([nwt * r2 for (nwt, r2) in zip(norm_weights, resid2)]))
        w_stdev_w_mean = sqrt(nwt2) * w_stdev_pop
    return w_mean, w_stdev_pop, w_stdev_w_mean