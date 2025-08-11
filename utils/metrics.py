import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate


def get_accuracy(y_hat, y, se=False):
    if y.dim() == 2:
        accuracy = ((y_hat.sigmoid() >.5).long() == y).float().mean().item()
    else:
        accuracy = (y_hat.argmax(dim=1) == y).sum().item() * 1.0 / len(y)
    if se:
        se = np.sqrt(accuracy * (1 - accuracy) / len(y))
        return accuracy, se
    return accuracy


def get_f1(y_hat, y):
    pred = (y_hat.sigmoid() >.5).long()if y.dim() == 2 else y_hat.argmax(dim=1)
    return f1_score(pred.detach().cpu(), y.detach().cpu(), average='macro')


def get_auc(y_hat, y):
    pred = (y_hat.sigmoid() >.5).long().detach().cpu().numpy() if y.dim() == 2 else y_hat.softmax(1).detach().cpu().numpy()
    target = y.detach().cpu().numpy()
    try:
        if y_hat.shape[1] > 2:
            auc = roc_auc_score(target, pred, multi_class='ovo', labels=range(pred.shape[1]))
        else:
            auc = roc_auc_score(target, pred[:, 1])
    except:
        auc = torch.tensor(0.0)
    return auc

####################
# My Metrics
####################
def compute_accuracy_with_se(y_true, y_pred):
    """
    Computes accuracy and a standard error based on a binomial approximation.
    """
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    y_pred_np = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)
    acc = accuracy_score(y_true_np, y_pred_np)
    n = len(y_true_np)
    # Standard error for a proportion (sqrt(p*(1-p)/n))
    se = np.sqrt(acc * (1 - acc) / n) if n > 0 else 0.0
    return acc, se

def binary_cm_metrics(y_true, y_pred, positive_class=1):
    """
    Computes binary confusion matrix metrics.
    """
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    y_pred_np = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
        "tnr": tnr,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }

def compute_confusion_metrics(y_true_np, y_pred_np, n_classes):
    """
    Computes a confusion matrix and derives per-class and aggregated metrics
    using a one-vs-rest approach.
    
    Returns a dictionary containing:
      - "cm": the multiclass confusion matrix,
      - Per-class metrics (computed using a binary OVR approach): 
        Accuracy, Precision, Recall, F1 Score, FPR, FNR, TNR, F0.5 and F2 scores.
      - Aggregated (macro and micro) metrics for all the above.
    """
    # Compute the overall confusion matrix for reference.
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(n_classes)))
    total = cm.sum()

    per_class_acc = {}
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    per_class_fpr = {}
    per_class_fnr = {}
    per_class_tnr = {}
    per_class_f0_5 = {}
    per_class_f2 = {}

    TP_total = FP_total = FN_total = TN_total = 0

    # For each class, calculate metrics using a one-vs-rest (OVR) strategy.
    for i in range(n_classes):
        # Create binary arrays: True for class i, False for all other classes.
        y_true_bin = (y_true_np == i)
        y_pred_bin = (y_pred_np == i)

        # Compute confusion matrix elements in the binary case.
        TP = np.sum(y_true_bin & y_pred_bin)
        FP = np.sum(~y_true_bin & y_pred_bin)
        FN = np.sum(y_true_bin & ~y_pred_bin)
        TN = np.sum(~y_true_bin & ~y_pred_bin)

        # Accuracy for class i: (TP + TN) / total samples
        acc = (TP + TN) / total if total > 0 else 0
        per_class_acc[i] = acc

        # Precision: TP / (TP + FP)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        per_class_precision[i] = precision

        # Recall: TP / (TP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        per_class_recall[i] = recall

        # F1 Score: harmonic mean of precision and recall.
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        per_class_f1[i] = f1

        # FPR: FP / (FP + TN)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        per_class_fpr[i] = fpr

        # FNR: FN / (FN + TP)
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        per_class_fnr[i] = fnr

        # TNR: TN / (TN + FP)
        tnr = TN / (TN + FP) if (TN + FP) > 0 else 0
        per_class_tnr[i] = tnr

        # F-beta scores: helper function for F0.5 and F2
        def f_beta(p, r, beta):
            denom = (beta ** 2) * p + r
            return (1 + beta ** 2) * p * r / denom if denom > 0 else 0

        per_class_f0_5[i] = f_beta(precision, recall, 0.5)
        per_class_f2[i]   = f_beta(precision, recall, 2)

        # Accumulate totals for micro-average calculations.
        TP_total += TP
        FP_total += FP
        FN_total += FN
        TN_total += TN

    # Macro averages: average over classes.
    macro_acc = np.mean(list(per_class_acc.values()))
    macro_precision = np.mean(list(per_class_precision.values()))
    macro_recall = np.mean(list(per_class_recall.values()))
    macro_f1 = np.mean(list(per_class_f1.values()))
    macro_fpr = np.mean(list(per_class_fpr.values()))
    macro_fnr = np.mean(list(per_class_fnr.values()))
    macro_tnr = np.mean(list(per_class_tnr.values()))
    macro_f0_5 = np.mean(list(per_class_f0_5.values()))
    macro_f2 = np.mean(list(per_class_f2.values()))

    # Micro averages: computed from overall counts.
    micro_acc = (TP_total) / total # TODO idk how to define micro accuracy
    #micro_acc = (TP_total + TN_total) / (TP_total + TN_total + FP_total + TN_total) if total > 0 else 0 # TODO: IDK if that's correct and if its needed and what it does. Also why we cant use TN_total since we get results over 1
    micro_precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0
    micro_recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0)
    micro_fpr = FP_total / (FP_total + TN_total) if (FP_total + TN_total) > 0 else 0
    micro_fnr = FN_total / (FN_total + TP_total) if (FN_total + TP_total) > 0 else 0
    micro_tnr = TN_total / (TN_total + FP_total) if (TN_total + FP_total) > 0 else 0

    # Micro F-beta scores using aggregated precision and recall.
    if (micro_precision + micro_recall) > 0:
        micro_f0_5 = (1 + 0.5**2) * micro_precision * micro_recall / (0.5**2 * micro_precision + micro_recall)
        micro_f2   = (1 + 2**2) * micro_precision * micro_recall / (2**2 * micro_precision + micro_recall)
    else:
        micro_f0_5 = 0
        micro_f2 = 0

    return {
        "cm": cm,
        "per_class_acc": per_class_acc,
        "macro_acc": macro_acc,
        "micro_acc": micro_acc,
        "per_class_precision": per_class_precision,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "per_class_recall": per_class_recall,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_fpr": per_class_fpr,
        "macro_fpr": macro_fpr,
        "micro_fpr": micro_fpr,
        "per_class_fnr": per_class_fnr,
        "macro_fnr": macro_fnr,
        "micro_fnr": micro_fnr,
        "per_class_tnr": per_class_tnr,
        "macro_tnr": macro_tnr,
        "micro_tnr": micro_tnr,
        "per_class_f0_5": per_class_f0_5,
        "macro_f0_5": macro_f0_5,
        "micro_f0_5": micro_f0_5,
        "per_class_f2": per_class_f2,
        "macro_f2": macro_f2,
        "micro_f2": micro_f2,
    }

def compute_auc_per_class(model_probs, y_true_np, n_classes, func, default=np.nan):
    """
    Compute per-class AUC using the provided function to calculate per class AUC
    in a one-vs-rest manner and returns the results as a dictionary.

    Parameters:
    -----------
    model_probs : np.ndarray
        Array of shape (n_samples, n_classes) containing the predicted probabilities for each class.
    y_true_np : np.ndarray
        Array of true labels (shape: (n_samples,)). Each label should be an integer in [0, n_classes-1].
    n_classes : int
        The total number of classes.
    func : callable
        A function that takes in two parameters: binary true labels and the corresponding predicted probabilities,
        and returns an AUC value (e.g., roc_auc_score or average_precision_score).
    default : float, optional
        The value to assign if a class has less than two unique labels (default is np.nan).

    Returns:
    --------
    dict
        A dictionary mapping each class index to its AUC value (or the default value).
    """
    aucs = {}
    for c in range(n_classes):
        # Create a binary vector for one-vs-rest: 1 if sample belongs to class c, 0 otherwise
        y_true_bin = (y_true_np == c).astype(int)
        # Extract predicted probabilities for class c
        y_scores = model_probs[:, c]
        # Check if there are both positive and negative samples
        if len(np.unique(y_true_bin)) < 2:
            auc = default
        else:
            try:
                auc = func(y_true_bin, y_scores)
            except Exception:
                auc = default
        aucs[c] = auc
    return aucs


def get_derived_scores(metric_name, scores, results, prefix, suffix):
    if scores == {}:
        print(f"No group/class scores found for: {metric_name}.")
        return results

    worst_group = min(scores, key=scores.get)
    worst_score = scores[worst_group]

    best_group = max(scores, key=scores.get)
    best_score = scores[best_group]

    average_score = np.mean(list(scores.values()))

    disparity = best_score - worst_score
    gap = average_score - worst_score

    worst_group_score = {f"{prefix}worst_{metric_name}{suffix}": worst_score}
    best_group_score = {f"{prefix}best_{metric_name}{suffix}": best_score}
    worst_group_name = {f"{prefix}name_worst_{metric_name}{suffix}": worst_group}
    best_group_name = {f"{prefix}name_best_{metric_name}{suffix}": best_group}
    avg_group_score = {f"{prefix}avg_{metric_name}{suffix}": average_score}
    disparity_score = {f"{prefix}disparity_{metric_name}{suffix}": disparity}
    gap_score = {f"{prefix}gap_{metric_name}{suffix}": gap}

    scores_dict = {f"{prefix}{metric_name}_dict{suffix}": scores}

    results = {
        **results,
        **worst_group_score,
        **best_group_score,
        **worst_group_name,
        **best_group_name,
        **avg_group_score,
        **disparity_score,
        **gap_score,
        **scores_dict
    }
    return results

def get_derived_scores_threshold(metric_name, scores, results, prefix, suffix, sample_counts, threshold=0.1):
    if scores == {}:
        print(f"No group/class scores found for: {metric_name}.")
        return results

    filtered_scores = {}
    for key, score in scores.items():
        identifier = key
        if isinstance(identifier, int):
            identifier = str(identifier)
        if prefix and identifier.startswith(prefix):
            identifier = identifier[len(prefix):]
        if suffix and identifier.endswith(suffix):
            identifier = identifier[:-len(suffix)]
        if "_" in identifier:
            identifier = identifier.split("_")[-1]
        
        if identifier.isdigit():
            identifier_lookup = int(identifier)
        else:
            identifier_lookup = identifier
        
        if sample_counts.get(identifier_lookup, 0) >= threshold or sample_counts.get(str(identifier_lookup), "0") >= threshold:
            filtered_scores[key] = score

    if not filtered_scores:
        print(f"No group/class meets the threshold for: {metric_name}.")
        return results

    worst_key = min(filtered_scores, key=filtered_scores.get)
    worst_score = filtered_scores[worst_key]

    best_key = max(filtered_scores, key=filtered_scores.get)
    best_score = filtered_scores[best_key]

    average_score = np.mean(list(filtered_scores.values()))

    disparity = best_score - worst_score
    gap = average_score - worst_score

    derived = {
        f"{prefix}worst_{metric_name}_threshold{threshold}{suffix}": worst_score,
        f"{prefix}best_{metric_name}_threshold{threshold}{suffix}": best_score,
        f"{prefix}name_worst_{metric_name}_threshold{threshold}{suffix}": worst_key,
        f"{prefix}name_best_{metric_name}_threshold{threshold}{suffix}": best_key,
        f"{prefix}avg_{metric_name}_threshold{threshold}{suffix}": average_score,
        f"{prefix}disparity_{metric_name}_threshold{threshold}{suffix}": disparity,
        f"{prefix}gap_{metric_name}_threshold{threshold}{suffix}": gap,
    }

    results.update(derived)
    return results

def get_fairness_metrics(results, y_pred, y_true, classes, groups, prefix, suffix, group_handle, rule_threshold=0.8):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().detach().numpy()
    if isinstance(groups, torch.Tensor):
        groups = groups.cpu().detach().numpy()

    num_classes = len(classes)
    unique_groups = np.unique(groups)

    def safe_mean(x):
        return np.mean(x) if len(x) > 0 else 0.0

    def compute_max_min_ratio(values):
        mx = max(values)
        mn = min(values)
        diff = mx - mn
        ratio = mn/mx if mx > 0 else 0.0
        return diff, ratio

    if num_classes == 2:
        mf = MetricFrame(
            metrics={
                "sr": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=groups
        )

        sr_vals = list(mf.by_group["sr"])
        tpr_vals = list(mf.by_group["tpr"])
        fpr_vals = list(mf.by_group["fpr"])

        dp_diff, dp_ratio = compute_max_min_ratio(sr_vals)
        _, di_min_ratio = compute_max_min_ratio(sr_vals)
        dp_80_pass = (dp_ratio >= rule_threshold)
        di_80_pass = (di_min_ratio >= rule_threshold)

        equal_opportunity_diff, equal_opportunity_ratio = compute_max_min_ratio(tpr_vals)
        fpr_diff, fpr_ratio = compute_max_min_ratio(fpr_vals)
        avg_odds_vals = [0.5*(t+f) for t,f in zip(tpr_vals, fpr_vals)]
        equalized_odds_diff, equalized_odds_ratio = compute_max_min_ratio(avg_odds_vals)

        pred_range = dp_diff
        true_rates = []
        for g in mf.by_group.index:
            mask_g = (groups == g)
            if np.sum(mask_g) == 0:
                true_rates.append(0.0)
            else:
                true_rates.append(safe_mean(y_true[mask_g] == 1))
        true_diff, _ = compute_max_min_ratio(true_rates)
        bias_amp = pred_range - true_diff

        results[f"{prefix}dp_diff{group_handle}{suffix}"] = dp_diff
        results[f"{prefix}dp_ratio{group_handle}{suffix}"] = dp_ratio
        results[f"{prefix}dp_80_percent_rule_pass{group_handle}{suffix}"] = dp_80_pass
        results[f"{prefix}di_min_ratio{group_handle}{suffix}"] = di_min_ratio
        results[f"{prefix}di_80_percent_rule_pass{group_handle}{suffix}"] = di_80_pass

        results[f"{prefix}equal_opportunity_diff{group_handle}{suffix}"] = equal_opportunity_diff
        results[f"{prefix}equal_opportunity_ratio{group_handle}{suffix}"] = equal_opportunity_ratio

        results[f"{prefix}fpr_diff{group_handle}{suffix}"] = fpr_diff
        results[f"{prefix}fpr_ratio{group_handle}{suffix}"] = fpr_ratio

        results[f"{prefix}equalized_odds_diff{group_handle}{suffix}"] = equalized_odds_diff
        results[f"{prefix}equalized_odds_ratio{group_handle}{suffix}"] = equalized_odds_ratio

        results[f"{prefix}bias_amp{group_handle}{suffix}"] = bias_amp
    else:
        dp_diff_list = []
        dp_ratio_list = []
        dp_80_pass_list = []
        di_min_ratio_list = []
        di_80_pass_list = []
        eopp_diff_list = []
        eopp_ratio_list = []
        fpr_diff_list = []
        fpr_ratio_list = []
        eo_odds_diff_list = []
        eo_odds_ratio_list = []
        bias_amp_list = []

        for c_idx, c_name in enumerate(classes):
            y_pred_bin = (y_pred == c_idx).astype(int)
            y_true_bin = (y_true == c_idx).astype(int)

            mf = MetricFrame(
                metrics={
                    "sr": selection_rate,
                    "tpr": true_positive_rate,
                    "fpr": false_positive_rate
                },
                y_true=y_true_bin,
                y_pred=y_pred_bin,
                sensitive_features=groups
            )

            sr_vals = list(mf.by_group["sr"])
            tpr_vals = list(mf.by_group["tpr"])
            fpr_vals = list(mf.by_group["fpr"])

            dp_diff_c, dp_ratio_c = compute_max_min_ratio(sr_vals)
            _, di_min_ratio_c = compute_max_min_ratio(sr_vals)

            dp_80_pass_c = (dp_ratio_c >= rule_threshold)
            di_80_pass_c = (di_min_ratio_c >= rule_threshold)

            eopp_diff_c, eopp_ratio_c = compute_max_min_ratio(tpr_vals)
            fpr_diff_c, fpr_ratio_c = compute_max_min_ratio(fpr_vals)

            avg_odds_vals = [0.5*(t+f) for t,f in zip(tpr_vals, fpr_vals)]
            eo_odds_diff_c, eo_odds_ratio_c = compute_max_min_ratio(avg_odds_vals)

            pred_range_c = dp_diff_c
            true_rates_c = []
            for g in mf.by_group.index:
                mask_g = (groups == g)
                if np.sum(mask_g) == 0:
                    true_rates_c.append(0.0)
                else:
                    true_rates_c.append(safe_mean(y_true[mask_g] == c_idx))
            true_diff_c, _ = compute_max_min_ratio(true_rates_c)
            bias_amp_c = pred_range_c - true_diff_c

            results[f"{prefix}dp_diff_class_{c_name}{group_handle}{suffix}"] = dp_diff_c
            results[f"{prefix}dp_ratio_class_{c_name}{group_handle}{suffix}"] = dp_ratio_c
            results[f"{prefix}dp_80_percent_rule_pass_class_{c_name}{group_handle}{suffix}"] = dp_80_pass_c
            results[f"{prefix}di_min_ratio_class_{c_name}{group_handle}{suffix}"] = di_min_ratio_c
            results[f"{prefix}di_80_percent_rule_pass_class_{c_name}{group_handle}{suffix}"] = di_80_pass_c

            results[f"{prefix}equal_opportunity_diff_class_{c_name}{group_handle}{suffix}"] = eopp_diff_c
            results[f"{prefix}equal_opportunity_ratio_class_{c_name}{group_handle}{suffix}"] = eopp_ratio_c

            results[f"{prefix}fpr_diff_class_{c_name}{group_handle}{suffix}"] = fpr_diff_c
            results[f"{prefix}fpr_ratio_class_{c_name}{group_handle}{suffix}"] = fpr_ratio_c

            results[f"{prefix}equalized_odds_diff_class_{c_name}{group_handle}{suffix}"] = eo_odds_diff_c
            results[f"{prefix}equalized_odds_ratio_class_{c_name}{group_handle}{suffix}"] = eo_odds_ratio_c

            results[f"{prefix}bias_amp_class_{c_name}{group_handle}{suffix}"] = bias_amp_c

            dp_diff_list.append(dp_diff_c)
            dp_ratio_list.append(dp_ratio_c)
            dp_80_pass_list.append(dp_80_pass_c)
            di_min_ratio_list.append(di_min_ratio_c)
            di_80_pass_list.append(di_80_pass_c)
            eopp_diff_list.append(eopp_diff_c)
            eopp_ratio_list.append(eopp_ratio_c)
            fpr_diff_list.append(fpr_diff_c)
            fpr_ratio_list.append(fpr_ratio_c)
            eo_odds_diff_list.append(eo_odds_diff_c)
            eo_odds_ratio_list.append(eo_odds_ratio_c)
            bias_amp_list.append(bias_amp_c)

        def mean_of(lst):
            return float(np.mean(lst)) if len(lst)>0 else 0.0

        dp_80_pass_macro = all(dp_80_pass_list)
        di_80_pass_macro = all(di_80_pass_list)

        results[f"{prefix}dp_diff_macro{group_handle}{suffix}"] = mean_of(dp_diff_list)
        results[f"{prefix}dp_ratio_macro{group_handle}{suffix}"] = mean_of(dp_ratio_list)
        results[f"{prefix}dp_80_percent_rule_pass_macro{group_handle}{suffix}"] = dp_80_pass_macro

        results[f"{prefix}di_min_ratio_macro{group_handle}{suffix}"] = mean_of(di_min_ratio_list)
        results[f"{prefix}di_80_percent_rule_pass_macro{group_handle}{suffix}"] = di_80_pass_macro

        results[f"{prefix}equal_opportunity_diff_macro{group_handle}{suffix}"] = mean_of(eopp_diff_list)
        results[f"{prefix}equal_opportunity_ratio_macro{group_handle}{suffix}"] = mean_of(eopp_ratio_list)

        results[f"{prefix}fpr_diff_macro{group_handle}{suffix}"] = mean_of(fpr_diff_list)
        results[f"{prefix}fpr_ratio_macro{group_handle}{suffix}"] = mean_of(fpr_ratio_list)

        results[f"{prefix}equalized_odds_diff_macro{group_handle}{suffix}"] = mean_of(eo_odds_diff_list)
        results[f"{prefix}equalized_odds_ratio_macro{group_handle}{suffix}"] = mean_of(eo_odds_ratio_list)

        results[f"{prefix}bias_amp_macro{group_handle}{suffix}"] = mean_of(bias_amp_list)
        
    return results