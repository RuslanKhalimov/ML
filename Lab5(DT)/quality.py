eps = 1e-6


def harmonic_mean(precision, recall):
    if precision < eps and recall < eps:
        return 0
    return 2 * precision * recall / (precision + recall)


def weighted_average(v, weights):
    v_sum = 0
    w_sum = 0
    for i in range(len(v)):
        v_sum += v[i] * weights[i]
        w_sum += weights[i]
    if v_sum < eps or w_sum < eps:
        return 0
    return v_sum / w_sum


def f_score(cm, k):
    row_sum = [sum(row) for row in cm]
    column_sum = [sum(column) for column in zip(*cm)]

    macro_precision = [0 if column_sum[i] < eps else cm[i][i] / column_sum[i] for i in range(k)]
    macro_recall = [0 if row_sum[i] < eps else cm[i][i] / row_sum[i] for i in range(k)]

    precision_avg = weighted_average(macro_precision, row_sum)
    recall_avg = weighted_average(macro_recall, row_sum)
    f_score_avg = harmonic_mean(precision_avg, recall_avg)

    return f_score_avg


def test_classifier(classifier, test_X, test_y):
    k = max(classifier.k, max(test_y) + 1)
    cm = [[0] * k for _ in range(k)]
    for i in range(len(test_X)):
        cm[test_y[i]][classifier.predict(test_X[i])] += 1
    return f_score(cm, k)
