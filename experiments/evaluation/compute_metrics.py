import torch
import tqdm
import numpy as np
from utils.metrics import compute_accuracy_with_se, binary_cm_metrics, compute_confusion_metrics, compute_auc_per_class, get_derived_scores, get_derived_scores_threshold, get_fairness_metrics
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


def compute_model_scores(
        model: torch.nn.Module,
        dl: torch.utils.data.DataLoader,
        device: str,
        limit_batches: int=None):
    model.to(device).eval()
    model_outs = []
    ys = []
    for i, (x_batch, y_batch) in enumerate(tqdm.tqdm(dl)):
        if limit_batches and limit_batches == i:
            break
        model_out = model(x_batch.to(device)).detach().cpu()
        model_outs.append(model_out)
        ys.append(y_batch)

    model_outs = torch.cat(model_outs)
    y_true = torch.cat(ys)

    return model_outs, y_true


def compute_metrics(model_outs, y_true, classes=None, prefix="", suffix="", groups=None, attacked_classes=None):
    results = {}

    # Move outputs to CPU and get predictions
    model_outs = model_outs.cpu()
    y_true = y_true.cpu()
    y_pred = model_outs.argmax(1).cpu()
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    model_outs_np = model_outs.numpy()

    # Others
    binary = (classes is not None) and (len(classes) == 2)
    lower_classes = [c.lower() for c in classes] if classes is not None else None
    unique_groups = groups.unique() if groups is not None else None
    n_classes = len(y_true.unique()) # Excludes the "unk" class

    ################
    # 1. Count Samples
    ################
    # Overall
    results[f"{prefix}n_samples{suffix}"] = len(y_true)
    
    # maybe with Counter
    if classes is not None:
        results[f"{prefix}n_classes{suffix}"] = n_classes
        unique_classes, class_counts = torch.unique(y_true, return_counts=True)
        class_counts_norm = class_counts.float() / class_counts.sum()

        class_counts_dict = {classes[int(name.item())]: int(count.item()) for name, count in zip(unique_classes, class_counts)}
        class_counts_norm_dict = {classes[int(name.item())]: float(count.item()) for name, count in zip(unique_classes, class_counts_norm)}

        results[f"{prefix}n_samples_per_class{suffix}"] = class_counts_dict
        results[f"{prefix}n_samples_per_class_normalized{suffix}"] = class_counts_norm_dict

    # Per Group (Raw and Normalized)
    if groups is not None:
        results[f"{prefix}n_groups{suffix}"] = len(unique_groups)
        unique_groups, group_counts = torch.unique(groups, return_counts=True)
        group_counts_norm = group_counts.float() / group_counts.sum()

        group_counts_dict = {str(gr.item()): int(count.item()) for gr, count in zip(unique_groups, group_counts)}
        group_counts_norm_dict = {str(gr.item()): float(count.item()) for gr, count in zip(unique_groups, group_counts_norm)}

        results[f"{prefix}n_samples_per_group{suffix}"] = group_counts_dict
        results[f"{prefix}n_samples_per_group_normalized{suffix}"] = group_counts_norm_dict

    # Positive and Negative Class
    if binary and classes is not None:
        results[f"{prefix}positive_class{suffix}"] = classes[1]
        results[f"{prefix}negative_class{suffix}"] = classes[0]
        print(f"Positive class: {classes[1]}, Negative class: {classes[0]}")

    ################
    # 2. Overall Model Performance Metrics
    ################

    # 2.1 Accuracy
    acc, se = compute_accuracy_with_se(y_true, y_pred)
    results[f"{prefix}accuracy{suffix}"] = acc
    results[f"{prefix}accuracy{suffix}_sderr"] = se

    ################
    # 3. Aggregated Metrics
    ################

    # 3.1 Confusion Matrix Metrics
    # Binary Case
    if binary:
        binary_cm_results = binary_cm_metrics(y_true, y_pred, positive_class=classes[1])
        # FPR, FNR, TNR, TPR
        results[f"{prefix}tpr_binary{suffix}"] = binary_cm_results["tpr"]
        results[f"{prefix}fpr_binary{suffix}"] = binary_cm_results["fpr"]
        results[f"{prefix}tnr_binary{suffix}"] = binary_cm_results["tnr"]
        results[f"{prefix}fnr_binary{suffix}"] = binary_cm_results["fnr"]

        # Precision, Recall, F1
        results[f"{prefix}precision_binary{suffix}"] = binary_cm_results["precision"]
        results[f"{prefix}recall_binary{suffix}"] = binary_cm_results["recall"]
        results[f"{prefix}f1_binary{suffix}"] = binary_cm_results["f1"]
    
    # Multiclass Case
    cm_metrics = compute_confusion_metrics(y_true_np, y_pred_np, n_classes=n_classes)
    results[f"{prefix}confusion_matrix{suffix}"] = cm_metrics["cm"]
    results[f"{prefix}accuracy_macro{suffix}"] = cm_metrics["macro_acc"]
    results[f"{prefix}accuracy_micro{suffix}"] = cm_metrics["micro_acc"]
    
    # FPR, FNR, TNR, TPR
    results[f"{prefix}tpr_macro{suffix}"] = cm_metrics["macro_recall"]
    results[f"{prefix}tpr_micro{suffix}"] = cm_metrics["micro_recall"]
    results[f"{prefix}fpr_macro{suffix}"] = cm_metrics["macro_fpr"]
    results[f"{prefix}fpr_micro{suffix}"] = cm_metrics["micro_fpr"]
    results[f"{prefix}tnr_macro{suffix}"] = cm_metrics["macro_tnr"]
    results[f"{prefix}tnr_micro{suffix}"] = cm_metrics["micro_tnr"]
    results[f"{prefix}fnr_macro{suffix}"] = cm_metrics["macro_fnr"]
    results[f"{prefix}fnr_micro{suffix}"] = cm_metrics["micro_fnr"]

    # Precision, Recall, F1, F0.5, F2
    results[f"{prefix}precision_macro{suffix}"] = cm_metrics["macro_precision"]
    results[f"{prefix}precision_micro{suffix}"] = cm_metrics["micro_precision"]
    results[f"{prefix}recall_macro{suffix}"] = cm_metrics["macro_recall"]
    results[f"{prefix}recall_micro{suffix}"] = cm_metrics["micro_recall"]
    results[f"{prefix}f1_macro{suffix}"] = cm_metrics["macro_f1"]
    results[f"{prefix}f1_micro{suffix}"] = cm_metrics["micro_f1"]
    results[f"{prefix}f0_5_macro{suffix}"] = cm_metrics["macro_f0_5"]
    results[f"{prefix}f0_5_micro{suffix}"] = cm_metrics["micro_f0_5"]
    results[f"{prefix}f2_macro{suffix}"] = cm_metrics["macro_f2"]
    results[f"{prefix}f2_micro{suffix}"] = cm_metrics["micro_f2"]

    # 3.2 AUC
    model_probs = torch.softmax(model_outs, dim=1).numpy().astype(np.float64)
    y_true_onehot = np.eye(len(classes))[y_true_np].astype(np.float64)
    if binary:
        auc_roc_binary = roc_auc_score(y_true_onehot, model_probs)
        auc_pr_binary = average_precision_score(y_true_onehot, model_probs)
        results[f"{prefix}model_probs_binary{suffix}"] = model_probs
        results[f"{prefix}auc_roc_binary{suffix}"] = auc_roc_binary
        results[f"{prefix}auc_pr_binary{suffix}"] = auc_pr_binary
    else:
        try:
            auc_roc_macro = roc_auc_score(y_true_onehot, model_probs, average='macro', multi_class='ovr')
            auc_roc_micro = roc_auc_score(y_true_onehot, model_probs, average='micro', multi_class='ovr')

            auc_pr_macro = average_precision_score(y_true_onehot, model_probs, average='macro')
            auc_pr_micro = average_precision_score(y_true_onehot, model_probs, average='micro')
            
        except ValueError as ve:
            print(f"Error computing AUC-ROC and AUC-PR: {ve}")
            auc_roc_macro = np.nan
            auc_roc_micro = np.nan
            auc_pr_macro = np.nan
            auc_pr_micro = np.nan
        
        results[f"{prefix}auc_roc_macro{suffix}"] = auc_roc_macro
        results[f"{prefix}auc_roc_micro{suffix}"] = auc_roc_micro
        results[f"{prefix}auc_pr_macro{suffix}"] = auc_pr_macro
        results[f"{prefix}auc_pr_micro{suffix}"] = auc_pr_micro

    # 3.3 Balanced Accuracy (for imbalanced datasets, average of recall per class)
    balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)
    results[f"{prefix}balanced_accuracy{suffix}"] = balanced_acc

    ################
    # 4. Per-Class Metrics
    ################
    if classes is not None:
        auc_roc_per_class = compute_auc_per_class(model_probs, y_true_onehot, n_classes, roc_auc_score)
        auc_pr_per_class = compute_auc_per_class(model_probs, y_true_onehot, n_classes, average_precision_score)

        for i in range(n_classes):
            class_name = classes[i]

            results[f"{prefix}accuracy_class_{class_name}{suffix}"] = cm_metrics['per_class_acc'][i]

            # Precision, Recall, F Beta
            results[f"{prefix}precision_class_{class_name}{suffix}"] = cm_metrics['per_class_precision'][i]
            results[f"{prefix}recall_class_{class_name}{suffix}"] = cm_metrics['per_class_recall'][i]
            results[f"{prefix}f1_class_{class_name}{suffix}"] = cm_metrics['per_class_f1'][i]
            results[f"{prefix}f0_5_class_{class_name}{suffix}"] = cm_metrics['per_class_f0_5'][i]
            results[f"{prefix}f2_class_{class_name}{suffix}"] = cm_metrics['per_class_f2'][i]

            # FPR, FNR, TNR, TPR
            results[f"{prefix}fpr_class_{class_name}{suffix}"] = cm_metrics['per_class_fpr'][i]
            results[f"{prefix}fnr_class_{class_name}{suffix}"] = cm_metrics['per_class_fnr'][i]
            results[f"{prefix}tnr_class_{class_name}{suffix}"] = cm_metrics['per_class_tnr'][i]
            results[f"{prefix}tpr_class_{class_name}{suffix}"] = cm_metrics['per_class_recall'][i]

            # AUC
            results[f"{prefix}auc_roc_class_{class_name}{suffix}"] = auc_roc_per_class[i]
            results[f"{prefix}auc_pr_class_{class_name}{suffix}"] = auc_pr_per_class[i]
        
        # Scores into dicts
        accuracy_per_class_dict = {classes[i]: cm_metrics['per_class_acc'][i] for i in range(n_classes)}
        precision_per_class_dict = {classes[i]: cm_metrics['per_class_precision'][i] for i in range(n_classes)}
        recall_per_class_dict = {classes[i]: cm_metrics['per_class_recall'][i] for i in range(n_classes)}
        f1_per_class_dict = {classes[i]: cm_metrics['per_class_f1'][i] for i in range(n_classes)}
        f0_5_per_class_dict = {classes[i]: cm_metrics['per_class_f0_5'][i] for i in range(n_classes)}
        f2_per_class_dict = {classes[i]: cm_metrics['per_class_f2'][i] for i in range(n_classes)}
        fpr_per_class_dict = {classes[i]: cm_metrics['per_class_fpr'][i] for i in range(n_classes)}
        fnr_per_class_dict = {classes[i]: cm_metrics['per_class_fnr'][i] for i in range(n_classes)}
        tnr_per_class_dict = {classes[i]: cm_metrics['per_class_tnr'][i] for i in range(n_classes)}
        tpr_per_class_dict = {classes[i]: cm_metrics['per_class_recall'][i] for i in range(n_classes)}
        auc_roc_per_class_dict = {classes[i]: auc_roc_per_class[i] for i in range(n_classes)}
        auc_pr_per_class_dict = {classes[i]: auc_pr_per_class[i] for i in range(n_classes)}
        
        # Derived Scores: Worst, Best, Avg, Gap, Disparity
        results = get_derived_scores("class_accuracy", accuracy_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_precision", precision_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_recall", recall_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_f1", f1_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_f0_5", f0_5_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_f2", f2_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_auc_roc", auc_roc_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_auc_pr", auc_pr_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_fpr", fpr_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_fnr", fnr_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_tnr", tnr_per_class_dict, results, prefix, suffix)
        results = get_derived_scores("class_tpr", tpr_per_class_dict, results, prefix, suffix)

        # Derived Scores with Thresholds
        thresholds = [0.0, 0.1, 0.2, 0.25, 0.5]
        for threshold in thresholds:
            results = get_derived_scores_threshold("class_accuracy", accuracy_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_precision", precision_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_recall", recall_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_f1", f1_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_f0_5", f0_5_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_f2", f2_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_auc_roc", auc_roc_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_auc_pr", auc_pr_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_fpr", fpr_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_fnr", fnr_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_tnr", tnr_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("class_tpr", tpr_per_class_dict, results, prefix, suffix, class_counts_norm_dict, threshold)

    ################
    # 5. Per-Group Metrics
    ################
    if groups is not None:
        group_acc = {}
        group_precision = {}
        group_recall = {}
        group_f1 = {}
        group_fpr = {}
        group_fnr = {}
        group_tnr = {}
        group_f0_5 = {}
        group_f2 = {}

        # TODO: In case of ISIC Timestamp, maybe also compute with an aggregated Group for small groups/classes
        for group in unique_groups:
            mask = (groups == group)
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            # Compute group accuracy
            if len(y_true_group) > 0:
                acc_group, _ = compute_accuracy_with_se(y_true_group, y_pred_group)
                group_acc[int(group.item())] = acc_group

                # Compute confusion metrics on the group (using macro averages)
                cm_group = compute_confusion_metrics(y_true_group.numpy(), y_pred_group.numpy(), n_classes=n_classes)
                group_precision[int(group.item())] = cm_group.get("macro_precision", np.nan)
                group_recall[int(group.item())] = cm_group.get("macro_recall", np.nan)
                group_f1[int(group.item())] = cm_group.get("macro_f1", np.nan)
                group_fpr[int(group.item())] = cm_group.get("macro_fpr", np.nan)
                group_fnr[int(group.item())] = cm_group.get("macro_fnr", np.nan)
                group_tnr[int(group.item())] = cm_group.get("macro_tnr", np.nan)
                group_f0_5[int(group.item())] = cm_group.get("macro_f0_5", np.nan)
                group_f2[int(group.item())] = cm_group.get("macro_f2", np.nan)

        results[f"{prefix}accuracy_per_group{suffix}"] = group_acc
        results[f"{prefix}precision_per_group{suffix}"] = group_precision
        results[f"{prefix}recall_per_group{suffix}"] = group_recall
        results[f"{prefix}f1_per_group{suffix}"] = group_f1
        results[f"{prefix}fpr_per_group{suffix}"] = group_fpr
        results[f"{prefix}fnr_per_group{suffix}"] = group_fnr
        results[f"{prefix}tnr_per_group{suffix}"] = group_tnr
        results[f"{prefix}f0_5_per_group{suffix}"] = group_f0_5
        results[f"{prefix}f2_per_group{suffix}"] = group_f2

        # Derived group scores
        results = get_derived_scores("group_accuracy", group_acc, results, prefix, suffix)
        results = get_derived_scores("group_precision", group_precision, results, prefix, suffix)
        results = get_derived_scores("group_recall", group_recall, results, prefix, suffix)
        results = get_derived_scores("group_f1", group_f1, results, prefix, suffix)
        results = get_derived_scores("group_fpr", group_fpr, results, prefix, suffix)
        results = get_derived_scores("group_fnr", group_fnr, results, prefix, suffix)
        results = get_derived_scores("group_tnr", group_tnr, results, prefix, suffix)
        results = get_derived_scores("group_f0_5", group_f0_5, results, prefix, suffix)
        results = get_derived_scores("group_f2", group_f2, results, prefix, suffix)

        # Derived scores with thresholds for groups.
        thresholds = [0.0, 0.02, 0.1, 0.2, 0.25, 0.5]
        for threshold in thresholds:
            results = get_derived_scores_threshold("group_accuracy", group_acc, results, prefix, suffix, group_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("group_precision", group_precision, results, prefix, suffix, group_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("group_recall", group_recall, results, prefix, suffix, group_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("group_f1", group_f1, results, prefix, suffix, group_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("group_fpr", group_fpr, results, prefix, suffix,group_counts_norm_dict,  threshold)
            results = get_derived_scores_threshold("group_fnr", group_fnr, results, prefix, suffix, group_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("group_tnr", group_tnr, results, prefix, suffix, group_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("group_f0_5", group_f0_5, results, prefix, suffix, group_counts_norm_dict, threshold)
            results = get_derived_scores_threshold("group_f2", group_f2, results, prefix, suffix, group_counts_norm_dict, threshold)

        # Bias Aligned and Bias Conflicting Groups
        if 'ch' in suffix or "waterbirds_balanced" in suffix or "isic" in suffix or suffix == "": # TODO: Check if correct
            if "nv" in lower_classes:
                bias_aligned_groups = [0, 3, 4, 6, 8, 10, 12, 14]
                bias_conflicting_groups = [1, 2, 5, 7, 9, 11, 13, 15]
            elif "benign" in lower_classes:
                bias_aligned_groups = [1, 2]
                bias_conflicting_groups = [0, 3]
            elif "waterbird" in lower_classes:
                bias_aligned_groups = [0, 3]
                bias_conflicting_groups = [1, 2]
            elif "frog" in lower_classes and any("frog" == item.lower() for item in attacked_classes):
                bias_aligned_groups = [0, 2, 5, 6, 8, 10, 12, 14]
                bias_conflicting_groups = [1, 3, 4, 7, 9, 11, 13, 15]
            elif "dog" in lower_classes and any("dog" == item.lower() for item in attacked_classes):
                bias_aligned_groups = [1, 2, 4, 6, 8, 10, 12, 14]
                bias_conflicting_groups = [0, 3, 5, 7, 9, 11, 13, 15]
            else:
                print("No Bias Aligned and Bias Conflicting Groups defined for the given classes.")
                bias_aligned_groups = [gr for gr in unique_groups if gr % 2 == 0]
                bias_conflicting_groups = [gr for gr in unique_groups if gr % 2 == 1]
            

            ba_bc_groups = get_new_grouping(groups, [bias_aligned_groups, bias_conflicting_groups])

            for g in np.unique(ba_bc_groups):
                mask = ba_bc_groups == g
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                
                g_name = "bias_aligned" if g in bias_aligned_groups else "bias_conflicting"
                
                accuracy_group, _ = compute_accuracy_with_se(y_true_group, y_pred_group)
                results[f"{prefix}accuracy_group_{g_name}{suffix}"] = accuracy_group
            
            bias_aligned_accuracy = results.get(f"{prefix}accuracy_group_bias_aligned{suffix}", None)
            bias_conflicting_accuracy = results.get(f"{prefix}accuracy_group_bias_conflicting{suffix}", None)
            if bias_aligned_accuracy is not None and bias_conflicting_accuracy is not None:
                results[f"{prefix}group_accuracy_diff_bc_ba{suffix}"] = bias_conflicting_accuracy - bias_aligned_accuracy
            else:
                print("Bias Aligned and Bias Conflicting Group Accuracy not found.")

        if ('ch' in suffix or "isic" in suffix or suffix == "") and 'nv' in lower_classes:
            aligned_wo_nv = [0, 4, 6, 8, 10, 12, 14]
            conflicting_wo_nv = [1, 5, 7, 9, 11, 13, 15]

            aligned_wo_nv_groups = get_new_grouping(groups, [aligned_wo_nv, conflicting_wo_nv])
            for g in np.unique(aligned_wo_nv_groups):
                mask = aligned_wo_nv_groups == g
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]
                g_name = "aligned_wo_nv" if g in aligned_wo_nv else "conflicting_wo_nv"
                accuracy_group, _ = compute_accuracy_with_se(y_true_group, y_pred_group)
                results[f"{prefix}accuracy_group_{g_name}{suffix}"] = accuracy_group
            

        if ('ch' in suffix and 'nv' in lower_classes) or "isic" in suffix:
            ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    
            benign_classes_without_artifact = [2, 8, 10, 12]
            benign_classes_with_artifact = [3, 9, 11, 13]  
            malignant_classes_without_artifact = [0, 4, 6, 14]
            malignant_classes_with_artifact = [1, 5, 7, 15]

            group_list = [benign_classes_without_artifact, malignant_classes_without_artifact, benign_classes_with_artifact, malignant_classes_with_artifact]
            group_names = ["benign_without_artifact", "malignant_without_artifact", "benign_with_artifact", "malignant_with_artifact"]

            new_groups = get_new_grouping(groups, group_list)

            group_acc = {}
            for g in np.unique(new_groups):
                mask = new_groups == g
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]

                g_name = group_names[g]

                accuracy_group, _ = compute_accuracy_with_se(y_true_group, y_pred_group)
                results[f"{prefix}accuracy_group_{g_name}{suffix}"] = accuracy_group
                group_acc[g_name] = accuracy_group
            results = get_derived_scores("group_accuracy_benign_malignant", group_acc, results, prefix, suffix)

            # Get Worst and Best Group Accuracy
            results[f"{prefix}worst_group_accuracy_benign_malignant{suffix}"] = min(group_acc.values())
            results[f"{prefix}best_group_accuracy_benign_malignant{suffix}"] = max(group_acc.values())
            results[f"{prefix}avg_group_accuracy_benign_malignant{suffix}"] = np.mean(list(group_acc.values()))
            
                
        
        # Subset Clean vs Subset Attacked Accuracies
        if 'ch' in suffix or "waterbirds_balanced" in suffix or suffix == "" or "isic" in suffix:
            clean_groups = [gr for gr in unique_groups if gr % 2 == 0]
            attacked_groups = [gr for gr in unique_groups if gr % 2 == 1]
            temp_groups = get_new_grouping(groups, [clean_groups, attacked_groups])

            for g in np.unique(temp_groups):
                mask = temp_groups == g
                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]

                g_name = "clean" if g in clean_groups else "attacked"

                accuracy_group, _ = compute_accuracy_with_se(y_true_group, y_pred_group)
                results[f"{prefix}accuracy_group_{g_name}{suffix}"] = accuracy_group
            
            clean_accuracy = results.get(f"{prefix}accuracy_group_clean{suffix}", None)
            attacked_accuracy = results.get(f"{prefix}accuracy_group_attacked{suffix}", None)
            if clean_accuracy is not None and attacked_accuracy is not None:
                results[f"{prefix}group_accuracy_diff_clean_attacked{suffix}"] = clean_accuracy - attacked_accuracy
            else:
                print("Clean and Attacked Group Accuracy not found.")
            
            if clean_accuracy is not None:
                # Diff between overall and clean accuracy
                acc = results[f"{prefix}accuracy{suffix}"]
                results[f"{prefix}group_accuracy_diff_clean{suffix}"] = acc - clean_accuracy

            # AUC per Group
            for g in np.unique(temp_groups):
                mask = temp_groups == g
                model_probs_group = model_probs[mask]
                y_true_group = y_true[mask]
                y_true_onehot_group = np.eye(len(classes))[y_true_group.numpy()].astype(np.float64)
                g_name = "clean" if g in clean_groups else "attacked"

                if binary:
                    auc_roc_group = roc_auc_score(y_true_onehot_group, model_probs_group)
                    auc_pr_group = average_precision_score(y_true_onehot_group, model_probs_group)

                    results[f"{prefix}auc_roc_binary_group_{g_name}{suffix}"] = auc_roc_group
                    results[f"{prefix}auc_pr_binary_group_{g_name}{suffix}"] = auc_pr_group
                else:
                    auc_roc_group = roc_auc_score(y_true_onehot_group, model_probs_group, average='macro', multi_class='ovr')
                    auc_pr_group = average_precision_score(y_true_onehot_group, model_probs_group, average='macro')

                    results[f"{prefix}auc_roc_macro_group_{g_name}{suffix}"] = auc_roc_group
                    results[f"{prefix}auc_pr_macro_group_{g_name}{suffix}"] = auc_pr_group
                        
        if 'ch' in suffix or "waterbirds_balanced" in suffix or suffix == "" or "isic" in suffix:
            results = get_fairness_metrics(results, y_pred, y_true, classes, groups, prefix, suffix, group_handle="_all_groups")
            binary_group_art_no_art = get_new_grouping(groups, [[gr for gr in unique_groups if gr % 2 == 1], [gr for gr in unique_groups if gr % 2 == 0]]) # Artifact = odd group, No Artifact = even group
            results = get_fairness_metrics(results, y_pred, y_true, classes, binary_group_art_no_art, prefix, suffix, group_handle="_fairness_groups")

            results = get_fairness_metrics(results, y_pred, y_true, classes, ba_bc_groups, prefix, suffix, group_handle="_bias_aligned_conflicting_groups")
        
        # model bias score + thresholds
        if 'ch' in suffix or "waterbirds_balanced" in suffix or suffix == "" or "isic" in suffix:
            bias_labels = get_new_grouping(groups, [[gr for gr in unique_groups if gr % 2 == 1], [gr for gr in unique_groups if gr % 2 == 0]])
            bias_labels = bias_labels.numpy()
            pred = y_pred.numpy()
            
            uniq_classes = np.unique(y_true)

            
            def compute_model_bias(pred, y_true, bias_labels, uniq_classes, class_names, threshold=0):
                per_class_bias = {}
                overall_bias = 0.0
                for t in uniq_classes:
                    # Identify samples where the ground truth is the target class t
                    class_mask = (y_true == t)
                    
                    # if less than threshold samples, skip
                    if np.sum(class_mask) < threshold * len(y_true):
                        per_class_bias[class_names[t]] = 0.0
                        continue

                    # Within class t, split the samples based on the bias label
                    group0 = class_mask & (bias_labels == 0)
                    group1 = class_mask & (bias_labels == 1)
                    
                    # Compute the correct classification rate for each bias group
                    # (if there are no samples for a group, we default the rate to 0)
                    rate0 = np.mean(pred[group0] == t) if np.sum(group0) > 0 else 0.0
                    rate1 = np.mean(pred[group1] == t) if np.sum(group1) > 0 else 0.0
                    
                    # Calculate bias for class t as the absolute difference in rates
                    bias_t = abs(rate0 - rate1)
                    per_class_bias[class_names[t]] = bias_t
                    
                    overall_bias += bias_t
            
                return overall_bias, per_class_bias

            thresholds = [0.0, 0.1, 0.2, 0.25, 0.5]
            for threshold in thresholds:
                overall_bias, per_class_bias = compute_model_bias(pred, y_true.numpy(), bias_labels, uniq_classes, classes, threshold)
                results[f"{prefix}model_bias{suffix}_threshold_{threshold}"] = overall_bias
                results[f"{prefix}per_class_bias{suffix}_threshold_{threshold}"] = per_class_bias
                avg_bias = overall_bias / len(uniq_classes)
                results[f"{prefix}average_bias{suffix}_threshold_{threshold}"] = avg_bias
                print(f"Model Bias (threshold={threshold}): {overall_bias}, Average Bias: {avg_bias}")
                print(f"Per Class Bias (threshold={threshold}): {per_class_bias}")


    return results

def compute_tcav_metrics_batch(grad, cav):
    grad = grad if grad.dim() > 2 else grad[..., None, None]
    metrics = {
        'TCAV_pos': ((grad * cav[..., None, None]).sum(1).flatten() > 0).sum().item(), # how many spatial gradients are aligned with the CAV (Counts each spatial location per img)
        'TCAV_neg': ((grad * cav[..., None, None]).sum(1).flatten() < 0).sum().item(),
        'TCAV_pos_mean': ((grad * cav[..., None, None]).sum(1).mean((1, 2)).flatten() > 0).sum().item(), # how many gradients are aligned with the CAV on average (Counts the avg per images)
        'TCAV_neg_mean': ((grad * cav[..., None, None]).sum(1).mean((1, 2)).flatten() < 0).sum().item(),
        'TCAV_sensitivity': (grad * cav[..., None, None]).sum(1).abs().flatten().numpy()

    }
    return metrics

def aggregate_tcav_metrics(TCAV_pos, TCAV_neg, TCAV_pos_mean, TCAV_neg_mean, TCAV_sens_list):

    eps = 1e-8
    TCAV_sens_list = np.concatenate(TCAV_sens_list)
    tcav_quotient = TCAV_pos / (TCAV_neg + TCAV_pos + eps)
    mean_tcav_quotient = TCAV_pos_mean / (TCAV_neg_mean + TCAV_pos_mean + eps)
    mean_tcav_sensitivity = TCAV_sens_list.mean()
    mean_tcav_sensitivity_sem = np.std(TCAV_sens_list) / np.sqrt(len(TCAV_sens_list))
    quotient_sderr = np.sqrt(tcav_quotient * (1 - tcav_quotient) / (TCAV_neg + TCAV_pos + eps))
    mean_quotient_sderr = np.sqrt(mean_tcav_quotient * (1 - mean_tcav_quotient) / (TCAV_neg_mean + TCAV_pos_mean + eps))

    metrics = {
        "tcav_quotient": tcav_quotient,
        "quotient_sderr": quotient_sderr,
        "mean_tcav_quotient": mean_tcav_quotient,
        "mean_quotient_sderr": mean_quotient_sderr,
        "mean_tcav_sensitivity": mean_tcav_sensitivity,
        "mean_tcav_sensitivity_sem": mean_tcav_sensitivity_sem
    }

    return metrics

def get_new_grouping(groups, new_group_order):
    new_grouping = []
    for group in groups:
        assigned = -1
        for idx, group_list in enumerate(new_group_order):
            if group in group_list:
                assigned = idx
                break
        new_grouping.append(assigned)

    if -1 in new_grouping:
        print("Warning: Some groups were not found in the new group order.")
    
    return torch.tensor(new_grouping)