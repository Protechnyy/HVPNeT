def eval_result(true_labels, pred_result, rel2id, logger, use_name=False):
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
        if name in rel2id:
            if use_name:
                neg = name
            else:
                neg = rel2id[name]
            break
    for i in range(total):
        if use_name:
            golden = true_labels[i]
        else:
            golden = true_labels[i]

        if golden == pred_result[i]:
            correct += 1 # 总预测正确数（TP+TN）
            if golden != neg:
                correct_positive += 1 # 正确预测的关系数(TP)
        if golden != neg:
            gold_positive += 1 # 真实有关系的总数（TP+FN）
        if pred_result[i] != neg:
            pred_positive += 1 # 预测为有关系的总数（TP+FP）
    acc = float(correct) / float(total) # 准确率 = (TP+TN) / (TP+TN+FP+FN)
    try:
        # 精确率= TP / (TP + FP)
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        # 召回率= TP / (TP + FN)
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        # F1值= 2 * 精确率 * 召回率 / (精确率 + 召回率)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    logger.info('Evaluation result: {}.'.format(result))
    return result