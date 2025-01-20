import matplotlib.pyplot as plt
import math

def index_in_range(index, range_tuple):
    if range_tuple is None:
        return True
    for (from_index, to_index) in range_tuple:
        if from_index <= index and index <= to_index:
            return True
    return False

def load_eval(path, range_tuple : list[tuple[int, int]]):
    with open(path) as f:
        data = f.readlines()
    
    x = []
    y = []

    for line in data:

        (self_id, history_id, score, loss) = line.split(" ")
        self_id = int(self_id)
        if self_id < 0 or not index_in_range(self_id, range_tuple):
            continue
        x.append(self_id)

        if float(loss) <= 0.0:
            y.append(math.log(1e-5))
        else:
            y.append(math.log(float(loss)))
    
    return x, y

methods = [
    ("icp", "ICP"),
    ("ndt", "NDT"),
    ("igicp", "IGICP"),
    ("loamv1", "LOAM"),
    ("lightloam", "Light-LOAM"),
    ("loamv2", "ours"), 
]

def plot_dataset(name, range_tuple):

    # plt.yscale("log")
    colors = ['gold','purple','yellowgreen','dodgerblue','darkorange','red','purple','tomato','chocolate']
    color_index = 0
    previous_length = 0
    for (method, display_name) in methods:
        x, y = load_eval(f"eval/{method}_{name}.eval", range_tuple)
        if previous_length == 0:
            previous_length = len(x)
        else:
            assert previous_length == len(x)
        
        x_tick = [i for i in range(previous_length)]
        plt.scatter(x_tick, y, label=display_name, s=1, color=colors[color_index])
        color_index += 1

    # draw a line at y = -1.5

    plt.axhline(y=0, color='r', linestyle='--', label="Threshold")
    plt.legend()

    plt.xlabel("Loop index")


    plt.ylabel("ln(loss)")
    plt.tight_layout()
    # plt.title(name)

    plt.savefig(f"eval/{name}.png", dpi=330)
    plt.show()

def parse_range_tuple(range_string):
    range_list = range_string.split(",")
    range_tuple = []
    for range_str in range_list:
        range_pair = range_str.split("-")
        assert len(range_pair) == 2
        range_tuple.append((int(range_pair[0]), int(range_pair[1])))
    return range_tuple

if __name__ == "__main__":
    import sys

    range_tuple = None
    if len(sys.argv) > 2:
        range_tuple = parse_range_tuple(sys.argv[2])

    plot_dataset(sys.argv[1], range_tuple)
