import matplotlib.pyplot as ply
import glob
import os
import sys
import math

def load_gt(gt_filename):
	# frame_index loop_id1 loop_id2 ....
	gt = []
	with open(gt_filename, 'r') as f:
		for line in f:
			line = line.strip()
			if line == '':
				continue
			if line.find(' ') != -1:
				objs = line.split(' ')
				gt.append((int(objs[0]), [int(i) for i in objs[1:] if int(i) < int(objs[0]) - 50]))
	return { k:v for k, v in gt if len(v) > 0 }


def load_result_file(result_filename):
    # self_index history_index distance
    result = []
    with open(result_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line.count(',') == 3:
                (a, b, d, loss) = line.split(',')
            else:
                (a, b, d, loss) = line.split(' ')
            result.append((int(a), int(b), float(d), float(loss) < 1.0))
    print(len([p for p in result if p[3]]))
    return result

def is_true(self_id, history_id, gt):
    if self_id not in gt:
        return False
    return history_id in gt[self_id]

def mark_result_for_iteration(result, gt):
    value_category = [] # 0: false positive, 1: true positive, 2: false negative, 3: true negative
    category_counter = [0, 0, 0, 0]

    # using a maximin distance threshold
    for (self_id, history_id, score, b_loss_under) in result:
        if history_id > 0: # positive
            if is_true(self_id, history_id, gt): # true positive
                value_category.append((score, 1)) # true positive
                category_counter[1] += 1
            else: # false positive
                value_category.append((score, 0)) # false positive
                category_counter[0] += 1
        else: # negative
            if self_id in gt: # false negative
                category_counter[2] += 1
            else: # true negative
                category_counter[3] += 1
    
    return value_category, category_counter


def calculate_prvalue(value_category, category_counter):
    TP = category_counter[1]
    FP = category_counter[0]
    FN = category_counter[2]
    TN = category_counter[3]

    sorted_category = sorted(value_category, key=lambda x: x[0], reverse=True)

    pr_values = []

    for (score, category) in sorted_category:
        if category == 0: # false positive to true negative
            FP -= 1
            TN += 1
        elif category == 1: # true positive to false negative
            TP -= 1
            FN += 1
        
        if TP + FP == 0 or TP + FN == 0 or TP == 0:
            continue
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        pr_values.append((precision, recall))
    
    pr_values = sorted(pr_values, key=lambda x: x[1], reverse=True)
    return pr_values


def calculate_PR(result, gt):
    value_category, category_counter = mark_result_for_iteration(result, gt)
    pr_values = calculate_prvalue(value_category, category_counter)
    return pr_values

def downsample_pr_values(pr_values):
    ds_pr = [pr_values[0]]
    (P, R) = pr_values[0]
    for (p, r) in pr_values:
        if math.isclose(r, R, abs_tol=0.1) and math.isclose(p, P, abs_tol=0.1):
            continue
        ds_pr.append((p, r))
        (P, R) = (p, r)
    return ds_pr

def plot_PR_curve(names, pr_values, save_path=None):
    assert len(names) == len(pr_values)

    colors = ['gold','dodgerblue','yellowgreen','purple','tomato','darkorange','purple','red','chocolate']
    markers = ['s','^','d','p','*','o','x','1','*','h']

    plots = []
    used_names = []
    for i in range(len(pr_values)):
        if len(pr_values[i]) == 0:
            continue
        used_names.append(names[i])
        x = [v[1] for v in pr_values[i]]
        y = [v[0] for v in pr_values[i]]
        p, = ply.plot(x, y, color=colors[i], linewidth=1)

        ds_value = downsample_pr_values(pr_values[i])
        x = [v[1] for v in ds_value]
        y = [v[0] for v in ds_value]
        ply.scatter(x, y, color=colors[i], marker=markers[i])
        plots.append(p)
        
    ply.xlabel("recall", fontsize = 20)
    ply.ylabel("precision", fontsize =20)
	#set xrange and yrange
    ply.xlim(0, 1.02)
    ply.ylim(0, 1.02)
    ply.xticks(fontsize=19)
    ply.yticks(fontsize=19)
    ply.legend(plots, used_names, loc="lower left", numpoints=1, fontsize =16)

    ply.tight_layout()
    if save_path is not None:
        ply.savefig(save_path, dpi=330)
    
    ply.show()


def grab_file_lists(argv):
    gt = argv[1]
    input_files = []
    for i in range(2, len(argv)):
        this_files = glob.glob(argv[i])
        for f in this_files:
            if f not in input_files:
                input_files.append(f)
    return gt, input_files

def get_short_name(file_path):
	(filepath,tempfilename) = os.path.split(file_path)
	(filename,extension) = os.path.splitext(tempfilename)
	return filename

def get_f1score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def get_f1score_from_pr_values(pr_values):
    if len(pr_values) == 0:
        return "N/A"
    f1_scores = [get_f1score(p, r) for (p, r) in pr_values]
    return max(f1_scores)

def run(gt, input_files, save_path=None):
    names = [get_short_name(f) for f in input_files]

    gt = load_gt(gt)
    pr_values = [calculate_PR(load_result_file(f), gt) for f in input_files]
    f1 = [get_f1score_from_pr_values(pr) for pr in pr_values]

    for i in range(len(input_files)):
        print(f"{names[i]}: {f1[i]}")
    
    plot_PR_curve(names, pr_values, save_path)


def main():
    return run(*grab_file_lists(sys.argv))
    

if __name__ == "__main__":
    main()
            