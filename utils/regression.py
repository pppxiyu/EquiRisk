import pandas as pd
import numpy as np
from pysal.model import spreg
from pysal.lib import weights
import math

def reg_spatial_lag(
        df, k, method,
        x='demographic_value', y='diff_travel', w_lag=1, summary=True, spillover=False,
):
    knn = weights.KNN.from_dataframe(df, k=k)
    if method == 'ML':
        assert w_lag == 1, 'ML_lag does not support w_lags != 1'
        reg = spreg.ML_Lag(
            df[[y]].values,
            df[[x]].values,
            name_y=y,
            name_x=[x],
            w=knn,
            method='ord',
        )
    elif method == 'GM':
        reg = spreg.GM_Lag(
            df[[y]].values,
            df[[x]].values,
            name_y=y,
            name_x=[x],
            w=knn,
            w_lags=w_lag,
        )
    elif method == 'GM_Combo_Het':
        reg = spreg.GM_Combo_Het(
            df[[y]].values,
            df[[x]].values,
            name_y=y,
            name_x=[x],
            w=knn,
            w_lags=w_lag,
        )
    elif method == 'GM_Combo_Hom':
        reg = spreg.GM_Combo_Hom(
            df[[y]].values,
            df[[x]].values,
            name_y=y,
            name_x=[x],
            w=knn,
            w_lags=w_lag,
        )
    if summary:
        print(reg.summary)
    if spillover:
        direct_effect = reg_direct_spillover_effect(reg.rho, reg.betas[1, 0], knn.full()[0], w_lag)
        print(f'\n The direct effect of the demographic feature is {direct_effect}')
    return reg


def reg_direct_spillover_effect(rho, beta, W, w_lag):
    I = np.identity(W.shape[0])
    term = I
    for i in range(w_lag):
        term -= rho * np.linalg.matrix_power(W, i + 1)
    multiplier = np.linalg.inv(term)
    diagonal = np.diag(multiplier)
    direct_effect = diagonal.mean() * beta
    return direct_effect


def reg_z_score_4_compared_coeff(a1, a2, std1, std2, cov):
    return (a1 - a2) / ( math.sqrt(std1 ** 2 + std2 ** 2 - 2 * cov) )


def reg_t_score_4_compared_coeff(a1, a2, std1, std2, cov):
    return (a1 - a2) / ( math.sqrt(std1 ** 2 + std2 ** 2 - 2 * cov))


def reg_SUR(x1, y1, x2, y2,):
    # NOTE: This calculation will lead to wrong results in my context

    from collections import OrderedDict
    import statsmodels.api as sm
    from linearmodels.system import SUR
    pd.options.display.float_format = '{:.8f}'.format
    equations = OrderedDict()
    equations['model1'] = {
        'dependent': y1,
        'exog': sm.add_constant(x1),
    }
    equations['model2'] = {
        'dependent': y2,
        'exog': sm.add_constant(x2),
    }
    mod = SUR(equations)
    res = mod.fit(
        method='gls',
        full_cov=True,
        iterate=False,
        iter_limit=500,
        tol=1e-6,
        cov_type='unadjusted'
    )
    # print(res.cov)
    return res.cov.loc['model1_exog_0.1', 'model2_exog_1.1']


def reg_build_matrix(gdf, method, k=None):
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
        gdf1, gdf2, method,
        x_col='demographic_value', y_col='diff_travel', n_iter=1000, w_lag=1,
        k1=None, k2=None, weight_method='KNN',
        spillover=False,
):
    diff = []
    c = 0
    for _ in range(n_iter):
        resample_i_1 = np.random.choice(len(gdf1), size=len(gdf1), replace=True)
        resample_i_2 = np.random.choice(len(gdf2), size=len(gdf2), replace=True)

        gdf1_resample = gdf1.iloc[resample_i_1].copy()
        gdf2_resample = gdf2.iloc[resample_i_2].copy()

        knn_1 = reg_build_matrix(gdf1_resample, weight_method, k=k1)
        knn_2 = reg_build_matrix(gdf2_resample, weight_method, k=k2)
        knn_1.silence_warnings = True
        knn_2.silence_warnings = True
        knn_1.transform = 'r'
        knn_2.transform = 'r'

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

    diff_array = np.array(diff)
    p = np.mean(diff_array > 0)
    print(f'{c} interation finished.')
    print(f'Mean value is {diff_array.mean()}, std is {diff_array.std()}')
    print(f'p-value {1 - p}')
    return


def reg_shift_test_regime(
        gdf1, gdf2, k, method,
        x_col='demographic_value', y_col="diff_travel", w_lag=1,
):
    gdf1['group'] = [0] * len(gdf1)
    gdf2['group'] = [1] * len(gdf2)

    gdf_combined = pd.concat([gdf1, gdf2], axis=0)
    gdf_combined = gdf_combined.reset_index(drop=True)

    knn = weights.KNN.from_dataframe(gdf_combined, k=k)
    knn_array = knn.full()[0]
    knn_array[:len(gdf1), len(gdf1):] = 0
    knn_array[len(gdf1):, :len(gdf1)] = 0
    knn_updated = weights.KNN.from_array(knn_array, k=k)

    if method == 'GM_Combo_Het':
        reg = spreg.GM_Combo_Het_Regimes(
            y=gdf_combined[[y_col]].values,
            x=gdf_combined[[x_col]].values,
            regimes=gdf_combined['group'].to_list(),
            w=knn_updated,
            constant_regi='many',
            regime_lag_sep=True,
            slx_lags=k,
            name_y=y_col,
            name_x=[x_col],
            name_regimes='group',
            w_lags=w_lag,
        )
    elif method == 'GM_Combo_Hom':
        reg = spreg.GM_Combo_Hom_Regimes(
            y=gdf_combined[[y_col]].values,
            x=gdf_combined[[x_col]].values,
            regimes=gdf_combined['group'].to_list(),
            w=knn_updated,
            constant_regi='many',
            regime_lag_sep=True,
            slx_lags=k,
            name_y=y_col,
            name_x=[x_col],
            name_regimes='group',
            w_lags=w_lag,
        )
    elif method == 'GM':
        reg = spreg.GM_Lag_Regimes(
            y=gdf_combined[[y_col]].values,
            x=gdf_combined[[x_col]].values,
            regimes=gdf_combined['group'].to_list(),
            w=knn_updated,
            constant_regi='many',
            regime_lag_sep=True,
            regime_err_sep=True,
            slx_lags=k,
            name_y=y_col,
            name_x=[x_col],
            name_regimes='group',
            w_lags=w_lag,
        )
    elif method == 'ML':
        assert w_lag == 1, 'ML_lag does not support w_lags != 1'
        reg = spreg.ML_Lag_Regimes(
            y=gdf_combined[[y_col]].values,
            x=gdf_combined[[x_col]].values,
            regimes=gdf_combined['group'].to_list(),
            w=knn_updated,
            constant_regi='many',
            regime_lag_sep=True,
            slx_lags=k,
            name_y=y_col,
            name_x=[x_col],
            name_regimes='group',
        )
    print(reg.summary)
    print(f'covariance is {reg.vm[1, 1 + reg.vm.shape[0] // 2]}')
    return reg
