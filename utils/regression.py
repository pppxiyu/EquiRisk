import pandas as pd
import numpy as np
from pysal.model import spreg
from pysal.lib import weights
import math
from scipy import stats
import statsmodels.formula.api as smf
from pysal.explore import esda
import geopandas as gpd



def calculate_morans_i(gdf: gpd.GeoDataFrame, column_name: str, weight_method: str = 'Queen', k: int = None):
    """
    Calculates Moran's I for a given GeoDataFrame column using a specified spatial weighting strategy.

    Args:
        gdf (gpd.GeoDataFrame): The input GeoDataFrame.
        column_name (str): The name of the column in the GeoDataFrame for which to calculate Moran's I.
        weight_method (str, optional): The spatial weights method to use.
                                       Supported methods: 'Queen', 'Rook', 'KNN'. Defaults to 'Queen'.
        k (int, optional): The number of nearest neighbors for 'KNN' weighting method.
                           Required if `weight_method` is 'KNN'. Defaults to None.

    Returns:
        pysal.explore.esda.Moran: The Moran's I object containing the calculated statistics.

    Raises:
        ValueError: If the specified column_name is not found in the GeoDataFrame.
        ValueError: If an unsupported weight_method is provided.
        ValueError: If 'KNN' method is chosen but 'k' is not specified.
    """
    if column_name not in gdf.columns:
        raise ValueError(f"Column '{column_name}' not found in the GeoDataFrame.")

    # Build spatial weights matrix
    w = None
    if weight_method == 'Queen':
        w = weights.Queen.from_dataframe(gdf, silence_warnings=True)
    elif weight_method == 'Rook':
        w = weights.Rook.from_dataframe(gdf, silence_warnings=True)
    elif weight_method == 'KNN':
        if k is None:
            raise ValueError("For 'KNN' weight_method, 'k' (number of neighbors) must be specified.")
        w = weights.KNN.from_dataframe(gdf, k=k, silence_warnings=True)
    else:
        raise ValueError(f"Unsupported weight_method: '{weight_method}'. Choose from 'Queen', 'Rook', 'KNN'.")

    # Calculate Moran's I
    moran = esda.Moran(gdf[column_name], w)

    return moran


def reg_ols(df, x_name, y_name, summary=True, return_residuals=False):
    """
    Runs an Ordinary Least Squares (OLS) linear regression.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        x_name (str): The name of the independent variable column in df.
        y_name (str): The name of the dependent variable column in df.
        summary (bool): Whether to print the regression summary. Defaults to True.
        return_residuals (bool): Whether to return the residual errors. Defaults to False.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: The OLS regression results object.
        pd.DataFrame (optional): The original DataFrame with a new 'residuals' column,
                                 if `return_residuals` is True.

    Raises:
        ValueError: If the specified independent or dependent variable column is not found in the DataFrame.
    """
    # Ensure the columns exist in the DataFrame
    if x_name not in df.columns:
        raise ValueError(f"Independent variable '{x_name}' not found in DataFrame.")
    if y_name not in df.columns:
        raise ValueError(f"Dependent variable '{y_name}' not found in DataFrame.")

    # Drop rows with NaN values in the relevant columns to avoid errors in regression
    df_clean = df[[y_name, x_name]].dropna()

    # Construct the formula string and run the OLS regression
    formula = f"{y_name} ~ {x_name}"
    model = smf.ols(formula=formula, data=df_clean).fit()

    if summary:
        print(model.summary())

    if return_residuals:
        # Create a copy of the original DataFrame to add residuals to
        df_with_residuals = df.copy()
        # Align residuals with the original DataFrame's index
        df_with_residuals.loc[df_clean.index, 'residuals'] = model.resid
        return model, df_with_residuals
    else:
        return model


def reg_spatial_lag(
        df, weight_method='Queen', k=None,
        x='demographic_value', y='diff_travel', w_lag=1, summary=True, spillover=False, method='ord', slx_lags=0
):
    """
    Run a spatial lag regression using PySAL.

    Args:
        df (DataFrame): Input data.
        weight_method (str): Spatial weights method ('Queen', 'Rook', 'KNN').
        k (int, optional): Number of neighbors for KNN.
        x (str): Name of the independent variable.
        y (str): Name of the dependent variable.
        w_lag (int): Number of spatial lags.
        summary (bool): Whether to print the regression summary.
        spillover (bool): Whether to print direct spillover effect.

    Returns:
        Regression results object.
    """
    m = reg_build_matrix(df, weight_method, k=k)
    assert w_lag == 1, 'ML_lag does not support w_lags != 1'
    reg = spreg.ML_Lag(
        df[[y]].values,
        df[[x]].values,
        name_y=y,
        name_x=[x],
        w=m,
        method=method,
        slx_lags=slx_lags,
    )
    if summary:
        print(reg.summary)
    if spillover:
        direct_effect = reg_direct_spillover_effect(reg.rho, reg.betas[1, 0], knn.full()[0], w_lag)
        print(f'\n The direct effect of the demographic feature is {direct_effect}')
    return reg


def reg_direct_spillover_effect(rho, beta, W, w_lag):
    """
    Calculate the direct spillover effect for a spatial lag model.

    Args:
        rho (float): Spatial autoregressive parameter.
        beta (float): Coefficient of interest.
        W (ndarray): Spatial weights matrix.
        w_lag (int): Number of spatial lags.

    Returns:
        float: Direct effect value.
    """
    I = np.identity(W.shape[0])
    term = I
    for i in range(w_lag):
        term -= rho * np.linalg.matrix_power(W, i + 1)
    multiplier = np.linalg.inv(term)
    diagonal = np.diag(multiplier)
    direct_effect = diagonal.mean() * beta
    return direct_effect


def reg_z_score_4_compared_coeff(a1, a2, std1, std2, cov):
    """
    Calculate the z-score for comparing two regression coefficients.

    Args:
        a1 (float): First coefficient.
        a2 (float): Second coefficient.
        std1 (float): Standard error of first coefficient.
        std2 (float): Standard error of second coefficient.
        cov (float): Covariance between coefficients.

    Returns:
        float: z-score.
    """
    return (a1 - a2) / ( math.sqrt(std1 ** 2 + std2 ** 2 - 2 * cov) )


def reg_t_score_4_compared_coeff(a1, a2, std1, std2, cov):
    """
    Calculate the t-score for comparing two regression coefficients.

    Args:
        a1 (float): First coefficient.
        a2 (float): Second coefficient.
        std1 (float): Standard error of first coefficient.
        std2 (float): Standard error of second coefficient.
        cov (float): Covariance between coefficients.

    Returns:
        float: t-score.
    """
    return (a1 - a2) / (math.sqrt(std1 ** 2 + std2 ** 2 - 2 * cov))

def reg_build_matrix(gdf, method, k=None):
    """
    Build a spatial weights matrix from a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): Input data.
        method (str): Spatial weights method ('KNN', 'Queen', 'Rook').
        k (int, optional): Number of neighbors for KNN.

    Returns:
        PySAL weights object: Spatial weights matrix.
    """
    m = None
    if method == 'KNN':
        assert k is not None, 'Specific k'
        m = weights.KNN.from_dataframe(gdf, k=k, use_index=False, silence_warnings=True)
    elif method == 'Queen':
        m = weights.Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    elif method == 'Rook':
        m = weights.Rook.from_dataframe(gdf, use_index=False, silence_warnings=True)
    assert m is not None, 'Calculation failed.'
    return m


def reg_shift_test_bootstrapping(
        gdf1_input, gdf2_input, method,
        x_col='demographic_value', y_col='diff_travel', n_iter=1000, w_lag=1,
        k1=None, k2=None, weight_method='KNN',
        spillover=False,
):
    """
    Perform a bootstrapping test to compare regression coefficients between two groups.

    Args:
        gdf1_input (GeoDataFrame): First group data.
        gdf2_input (GeoDataFrame): Second group data.
        method (str): Regression method.
        x_col (str): Name of the independent variable.
        y_col (str): Name of the dependent variable.
        n_iter (int): Number of bootstrap iterations.
        w_lag (int): Number of spatial lags.
        k1 (int, optional): Number of neighbors for KNN (group 1).
        k2 (int, optional): Number of neighbors for KNN (group 2).
        weight_method (str): Spatial weights method.
        spillover (bool): Whether to compare spillover effects.

    Returns:
        None
    """
    gdf1 = gdf1_input.copy()
    gdf2 = gdf2_input.copy()
    intersection_df = pd.merge(gdf1, gdf2, on='geometry', how='inner')
    gdf1 = gdf1[gdf1['geometry'].isin(intersection_df['geometry'])]
    gdf2 = gdf2[gdf2['geometry'].isin(intersection_df['geometry'])]
    assert (gdf1['geometry'] == gdf2['geometry']).all()
    assert len(gdf1) == len(gdf2)

    diff = []
    c = 0
    for _ in range(n_iter):
        resample_i = np.random.choice(len(gdf1), size=len(gdf1), replace=True)

        gdf1_resample = gdf1.iloc[resample_i].copy()
        gdf2_resample = gdf2.iloc[resample_i].copy()

        knn_1 = reg_build_matrix(gdf1_resample, weight_method, k=k1)
        knn_2 = reg_build_matrix(gdf2_resample, weight_method, k=k2)
        knn_1.silence_warnings = True
        knn_2.silence_warnings = True
        knn_1.transform = 'r'
        knn_2.transform = 'r'
        assert (knn_1.full()[0] == knn_2.full()[0]).all() == True

        try:
            if method == 'GM_Combo_Het':
                reg_1 = spreg.GM_Combo_Het(
                    gdf1_resample[[y_col]].values,
                    gdf1_resample[[x_col]].values,
                    name_y=y_col,
                    name_x=[x_col],
                    w=knn_1,
                    w_lags=w_lag,
                )
                reg_2 = spreg.GM_Combo_Het(
                    gdf2_resample[[y_col]].values,
                    gdf2_resample[[x_col]].values,
                    name_y=y_col,
                    name_x=[x_col],
                    w=knn_2,
                    w_lags=w_lag,
                )
            elif method == 'GM_Combo_Hom':
                reg_1 = spreg.GM_Combo_Hom(
                    gdf1_resample[[y_col]].values,
                    gdf1_resample[[x_col]].values,
                    name_y=y_col,
                    name_x=[x_col],
                    w=knn_1,
                    w_lags=w_lag,
                )
                reg_2 = spreg.GM_Combo_Hom(
                    gdf2_resample[[y_col]].values,
                    gdf2_resample[[x_col]].values,
                    name_y=y_col,
                    name_x=[x_col],
                    w=knn_2,
                    w_lags=w_lag,
                )
            elif method == 'GM':
                reg_1 = spreg.GM_Lag(
                    gdf1_resample[[y_col]].values,
                    gdf1_resample[[x_col]].values,
                    name_y=y_col,
                    name_x=[x_col],
                    w=knn_1,
                    w_lags=w_lag,
                )
                reg_2 = spreg.GM_Lag(
                    gdf2_resample[[y_col]].values,
                    gdf2_resample[[x_col]].values,
                    name_y=y_col,
                    name_x=[x_col],
                    w=knn_2,
                    w_lags=w_lag,
                )
            elif method == 'ML':
                assert w_lag == 1, 'ML_lag does not support w_lags != 1'
                reg_1 = spreg.ML_Lag(
                    gdf1_resample[[y_col]].values,
                    gdf1_resample[[x_col]].values,
                    name_y=y_col,
                    name_x=[x_col],
                    w=knn_1,
                )
                reg_2 = spreg.ML_Lag(
                    gdf2_resample[[y_col]].values,
                    gdf2_resample[[x_col]].values,
                    name_y=y_col,
                    name_x=[x_col],
                    w=knn_2,
                )
        except Exception as e:
            print(e)
            continue
        c += 1

        if spillover:
            e_1 = reg_direct_spillover_effect(reg_1.rho, reg_1.betas[1, 0], knn_1.full()[0], w_lag)
            e_2 = reg_direct_spillover_effect(reg_2.rho, reg_2.betas[1, 0], knn_2.full()[0], w_lag)
            diff.append(e_1 - e_2)
        else:
            b_1 = reg_1.betas[1, 0]
            b_2 = reg_2.betas[1, 0]
            diff.append(b_1 - b_2)

        if c % 100 == 0:
            print(f"p-value at Iteration {c}: {1 - np.mean(np.array(diff) > 0)}.")

    diff_array = np.array(diff)
    p = np.mean(diff_array > 0)
    print(f'{c} interation finished.')
    print(f'Mean value is {diff_array.mean()}, std is {diff_array.std()}')
    print(f'p-value {1 - p}')
    return


def ztest_mean_test(
    series_a: pd.Series,
    series_b: pd.Series,
    pop_std_a: float, # Known population standard deviation for Series A
    pop_std_b: float, # Known population standard deviation for Series B
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> tuple[bool, float, float]:
    """
    Performs a Z-test for two independent means with known population standard deviations.

    Null Hypothesis (H0): mean(Series A) == mean(Series B)
    Alternative Hypothesis (H1):
        - 'two-sided': mean(Series A) != mean(Series B)
        - 'smaller': mean(Series A) < mean(Series B)
        - 'larger': mean(Series A) > mean(Series B)

    Args:
        series_a (pd.Series): The first pandas Series.
        series_b (pd.Series): The second pandas Series.
        pop_std_a (float): The known population standard deviation for Series A.
        pop_std_b (float): The known population standard deviation for Series B.
        alpha (float): The significance level (e.g., 0.05 for 5%).
        alternative (str): The alternative hypothesis. Must be 'two-sided', 'smaller', or 'larger'.

    Returns:
        tuple: A tuple containing:
            - bool: True if the null hypothesis is rejected (i.e., the result is statistically significant),
                    False otherwise.
            - float: The p-value for the test.
            - float: The Z-statistic.

    Raises:
        ValueError: If an invalid 'alternative' string is provided.
    """
    if alternative not in ['two-sided', 'smaller', 'larger']:
        raise ValueError("Alternative must be 'two-sided', 'smaller', or 'larger'.")

    # Calculate sample means
    mean_a = series_a.mean()
    mean_b = series_b.mean()

    # Calculate sample sizes
    n_a = len(series_a)
    n_b = len(series_b)

    # Calculate the standard error of the difference in means
    # Formula: sqrt((sigma_a^2 / n_a) + (sigma_b^2 / n_b))
    se_diff = np.sqrt((pop_std_a**2 / n_a) + (pop_std_b**2 / n_b))

    # Calculate the Z-statistic
    # Formula: (mean_a - mean_b) / se_diff
    z_statistic = (mean_a - mean_b) / se_diff

    # Calculate the p-value based on the alternative hypothesis
    if alternative == 'two-sided':
        # For two-sided, we take the absolute Z-value and multiply the tail probability by 2
        p_value = 2 * stats.norm.cdf(-np.abs(z_statistic))
    elif alternative == 'smaller': # H1: mean(A) < mean(B)
        # We are interested in the left tail probability
        p_value = stats.norm.cdf(z_statistic)
    elif alternative == 'larger': # H1: mean(A) > mean(B)
        # We are interested in the right tail probability (survival function)
        p_value = stats.norm.sf(z_statistic)

    is_significant = p_value < alpha

    return is_significant, p_value, z_statistic


def bootstrap_spatial_inequity(
    df_real, 
    y_real='TravelTime', 
    df_est=None, 
    y_est='Total_Seconds', 
    x_var='demographic_value',
    iterations=500,
    conf_interval=95
):
    """
    Improved spatial bootstrap supporting two DataFrames and dynamic columns.
    
    Args:
        df_real (pd.DataFrame): DF containing the ground-truth target.
        y_real (str): Column name for real-world travel time.
        df_est (pd.DataFrame, optional): DF for estimated target. If None, uses df_real.
        y_est (str): Column name for estimated travel time.
        x_var (str): Independent variable (income/demographics).
        iterations (int): Number of bootstrap samples.
        conf_interval (int): CI level (e.g., 95).
    """
    if df_est is None:
        df_est = df_real

    ratios = []
    valid_inequity_count = 0
    
    # Calculate alpha for percentiles
    lower_p = (100 - conf_interval) / 2
    upper_p = 100 - lower_p

    print(f"Starting bootstrap for {iterations} iterations on n={len(df_real)} observations...")

    for i in range(iterations):
        # 1. Generate bootstrap indices (consistent across both DFs)
        indices = np.random.choice(df_real.index, size=len(df_real), replace=True)
        
        boot_real = df_real.loc[indices].reset_index(drop=True)
        boot_est = df_est.loc[indices].reset_index(drop=True)

        try:
            # 2. Run Real-world model
            res_real = reg_spatial_lag(boot_real, y=y_real, x=x_var, summary=False)
            beta_real = res_real.betas[1][0]
            
            # 3. Run Estimate model
            res_est = reg_spatial_lag(boot_est, y=y_est, x=x_var, summary=False)
            beta_est = res_est.betas[1][0]
            
            # 4. Calculate Ratio
            ratio = beta_est / beta_real
            
            # Directional Filter: Only count if both show 'inequity' (negative coefficient)
            if beta_est < 0 and beta_real < 0:
                ratios.append(ratio)
                valid_inequity_count += 1
            
        except Exception as e:
            # Catch linear algebra errors or singular matrices
            continue
            
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1} complete...")

    # 5. Safety check if no ratios were collected
    if not ratios:
        print("Error: No iterations produced dual-negative coefficients. Check your data.")
        return None, None

    # 6. Calculate Stats
    lower_bound = np.percentile(ratios, lower_p)
    upper_bound = np.percentile(ratios, upper_p)
    mean_ratio = np.mean(ratios)

    print("\n" + "="*30)
    print("      BOOTSTRAP RESULTS      ")
    print("="*30)
    print(f"Valid 'Inequity' Samples: {valid_inequity_count}/{iterations}")
    print(f"Mean Capture Ratio:   {mean_ratio:.4f}")
    print(f"{conf_interval}% CI:              [{lower_bound:.4f}, {upper_bound:.4f}]")
    print("="*30)
    
    return

