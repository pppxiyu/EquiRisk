import pandas as pd
import numpy as np
from pysal.model import spreg
from pysal.lib import weights
import math


def reg_spatial_lag(
        df, weight_method='Queen', k=None,
        x='demographic_value', y='diff_travel', w_lag=1, summary=True, spillover=False,
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
        method='ord',
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
