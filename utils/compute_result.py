from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import numpy as np


def convert_obj_score(ori_obj_score, MOS):
    """
    func:
        fitting the objetive score to the MOS scale.
        nonlinear regression fit
    """

    def logistic_fun(x, a, b, c, d):
        return (a - b) / (1 + np.exp(-(x - c) / abs(d))) + b

    # nolinear fit the MOSp
    param_init = [np.max(MOS), np.min(MOS), np.mean(ori_obj_score), 1]
    popt, pcov = curve_fit(logistic_fun, ori_obj_score, MOS,
                           p0=param_init, ftol=1e-8, maxfev=100000)
    # a, b, c, d = popt[0], popt[1], popt[2], popt[3]

    obj_fit_score = logistic_fun(ori_obj_score, popt[0], popt[1], popt[2], popt[3])

    return obj_fit_score


def compute_metric(y, y_pred):
    index_to_del = []
    for i in range(len(y_pred)):
        if y_pred[i] <= 0:
            print("your prediction seems like not quit good, we reconmand you remove it   ", y_pred[i])
            index_to_del.append(i)
    n=0
    for i in index_to_del:
        y_pred = np.delete(y_pred, i-n)
        y = np.delete(y, i-n)
        n += 1
    print(y_pred.size)
    print(y.size)
    MSE = mean_squared_error
    RMSE = MSE(y_pred, y) ** 0.5
    PLCC = stats.pearsonr(convert_obj_score(y_pred, y), y)[0]
    SROCC = stats.spearmanr(y_pred, y)[0]
    KROCC = stats.kendalltau(y_pred, y)[0]

    return RMSE, PLCC, SROCC, KROCC
