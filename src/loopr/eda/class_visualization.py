
from IPython.display import Image
from random import sample
from collections import defaultdict
from PIL.Image import open as open_image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
from torch.utils.data import Dataset
from collections import Counter

from loopr.config.training_nn import TrainingNNConfig
colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#4e79a7",  # soft blue
    "#f28e2b",  # soft orange
    "#59a14f",  # soft green
]


def number_of_files_per_class(train_ds: Dataset, val_ds: Dataset):
    all_classes = ["None"]+TrainingNNConfig.kept_classes+TrainingNNConfig.unkept_classes
    data = Counter(train_ds.classes)
    positions = np.arange(1, len(all_classes)+1, 1)
    # ax = plt.subplot(111)
    values = [0 for i in range(len(all_classes))]
    for i,c in enumerate(all_classes):
        values[i] = data[i]
        
    # if hasattr(train_ds, "all_is_file_defect"):
    #     # classification ds
    #     values[0] = len(train_ds.all_is_file_defect) - sum(train_ds.all_is_file_defect)
    # else:
    #     # semantic seg
    # values[0] = len(train_ds) - sum(values)
        
    fig, ax = plt.subplots()
    bar_container = ax.bar(positions, values, color=colors)
    ax.bar_label(bar_container, fmt='{:,.0f}')
    ax.set_xticks(positions)
    ax.set_yticks([0, min(values), max(values)])
    ax.set_xticklabels(["None"]+TrainingNNConfig.kept+TrainingNNConfig.unkept, rotation=90)
    ax.set_title("Train Class Distribution")
    plt.ylabel("Number of Samples")
    data = Counter(val_ds.classes)
    positions = np.arange(1, len(all_classes)+1, 1)
    values = [0 for i in range(len(all_classes))]
    for i,c in enumerate(all_classes):
        values[i] = data[i]
    # values[0] = len(val_ds.all_is_file_defect) - sum(val_ds.all_is_file_defect)
    
    ax = plt.subplot(111)

    
    fig, ax = plt.subplots()
    
    bar_container = ax.bar(positions, values, color=colors)
    ax.bar_label(bar_container, fmt='{:,.0f}')
    ax.set_xticks(positions)
    ax.set_yticks([0, min(values), max(values)])
    ax.set_xticklabels(["None"]+TrainingNNConfig.kept+TrainingNNConfig.unkept, rotation=90)
    ax.set_title("Validation Class Distribution")
    plt.ylabel("Number of Samples")

def files_by_class():
    defect_image_dir = TrainingNNConfig.data_dir / "Defect_images"
    defect_image_files = list(defect_image_dir.glob("*.png"))
    defect_class_to_image_files = defaultdict(list)
    for defect_image_file in defect_image_files:
        defect_class_to_image_files[int(defect_image_file.name.split("_")[1])].append(defect_image_file)
    for class_label in defect_class_to_image_files:
        print(f"{TrainingNNConfig.class_label_to_name[class_label]}({class_label}): {len(defect_class_to_image_files[class_label])} ")
    ax = plt.subplot(111)
    keys = list(defect_class_to_image_files.keys())
    values = [len(defect_class_to_image_files[key]) for key in keys]
    barss = plt.bar(range(len(values)),values,color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_yticks([0, max(values)])
    ax.set_xticklabels([TrainingNNConfig.class_label_to_name[keys[i]] for i in range(len(keys))], rotation = 90)
    ax.set_title("Defects Per Class")
    plt.ylabel("Number of Example Defects")
    plt.show()

def visualize_few_image():
    defect_image_dir = TrainingNNConfig.data_dir / "Defect_images"
    no_defect_image_dir = TrainingNNConfig.data_dir / "NODefect_images"
    mask_image_dir = TrainingNNConfig.data_dir / "Mask_images"
    num_view = 5
    defect_image_files = list(defect_image_dir.glob("*"))
    random_defect_image_files = sample(
        defect_image_files,
        num_view
    )
    print("Fabric  with Defect Images")
    for defect_image_file in random_defect_image_files:
        defect_image = Image(filename=defect_image_file) 
        display(defect_image)
        
        # Find defect mask
        image_id = defect_image_file.name.split("_")[0]
        mask_image_files = list(mask_image_dir.glob(f"{image_id}*"))
        if len(mask_image_files) == 0: continue
        mask_image_file = mask_image_files[0]
        mask_image = Image(filename=mask_image_file) 
        display(mask_image)
        
        
    no_defect_image_files = list(no_defect_image_dir.glob("*"))
    random_no_defect_image_files = sample(
        no_defect_image_files,
        num_view
    )
    print("Fabric  with No Defect Images")
    for no_defect_image_file in random_no_defect_image_files:
        no_defect_image = Image(filename=no_defect_image_file) 
        display(no_defect_image)

def visualize_by_class():
    defect_image_dir = TrainingNNConfig.data_dir / "Defect_images"
    mask_image_dir = TrainingNNConfig.data_dir / "Mask_images"
    num_view = 5
    defect_image_files = list(defect_image_dir.glob("*"))
    defect_class_to_image_files = defaultdict(list)

    for defect_image_file in defect_image_files:
        defect_class_to_image_files[int(defect_image_file.name.split("_")[1])].append(defect_image_file)
    
    for class_label, class_defect_image_files in defect_class_to_image_files.items():
        print(f"Fabric Defect: {TrainingNNConfig.class_label_to_name[class_label]}")
        random_defect_image_files = sample(
            class_defect_image_files,
            num_view
        ) if len(class_defect_image_files)> num_view else class_defect_image_files
        for defect_image_file in random_defect_image_files:
            image_id = defect_image_file.name.split("_")[0]
            print(f"Image ID: {image_id}")
            defect_image = Image(filename=defect_image_file) 
            display(defect_image)
            
            # Find defect mask
            mask_image_files = list(mask_image_dir.glob(f"{image_id}*"))
            if len(mask_image_files) == 0: continue
            mask_image_file = mask_image_files[0]
            mask_image = Image(filename=mask_image_file) 
            display(mask_image)

def image_intensities_by_class():
    defect_image_dir = TrainingNNConfig.data_dir / "Defect_images"
    defect_image_files = list(defect_image_dir.glob("*"))
    defect_class_to_pixel_intensity = defaultdict(list)
    defect_class_to_image_files = defaultdict(list)
    
    for defect_image_file in defect_image_files:
        defect_class_to_image_files[int(defect_image_file.name.split("_")[1])].append(defect_image_file)
    
    for class_label, class_defect_image_files in defect_class_to_image_files.items():
        # if class_label not in TrainingNNConfig.kept_classes:
        #     continue
        for defect_image_file in class_defect_image_files:
            defect_image = open_image(defect_image_file)
            image_data = np.array(defect_image)
            if len(image_data.shape) == 3:
                image_data = image_data[:, :, 0]
            defect_class_to_pixel_intensity[class_label].extend(image_data.flatten().tolist())
    for class_label in defect_class_to_pixel_intensity:
        defect_class_to_pixel_intensity[class_label] = sample(defect_class_to_pixel_intensity[class_label], 50_000)

    # enforcing order for labels
    keys = list(defect_class_to_pixel_intensity.keys())
    values = list([defect_class_to_pixel_intensity[key] for key in keys])
    positions = np.arange(0, len(keys), 1)
    ax = plt.subplot(111)
    
    violins = plt.violinplot(values, positions=positions, showmeans=True)
    
    ax.set_xticks(positions)
    ax.set_yticks([0, 255])
    label_texts = []
    for i, body in enumerate(violins["bodies"]):
        body.set_facecolor(colors[i])
        label_texts.append(TrainingNNConfig.class_label_to_name[keys[i]])
    ax.set_xticklabels(label_texts, rotation = 90)
    plt.title("Intensities by Defect Class")
    plt.show()

def image_intensities_by_fabric_code():
    defect_image_dir = TrainingNNConfig.data_dir / "Defect_images"
    defect_image_files = list(defect_image_dir.glob("*"))
    fabric_code_to_pixel_intensity = defaultdict(list)
    fabric_code_to_image_files = defaultdict(list)
    
    for defect_image_file in defect_image_files:
        fabric_code = defect_image_file.name.split("_")[2].split(".")[0]
        class_label = int(defect_image_file.name.split("_")[1])
        # if class_label not in TrainingNNConfig.kept_classes:
        #     continue
        fabric_code_to_image_files[fabric_code].append(defect_image_file)
    
    for fabric_code, class_defect_image_files in fabric_code_to_image_files.items():

        for defect_image_file in class_defect_image_files:
            defect_image = open_image(defect_image_file)
            image_data = np.array(defect_image)
            if len(image_data.shape) == 3:
                image_data = image_data[:, :, 0]
            fabric_code_to_pixel_intensity[fabric_code].extend(image_data.flatten().tolist())
    for fabric_code in fabric_code_to_pixel_intensity:
        fabric_code_to_pixel_intensity[fabric_code] = sample(
            fabric_code_to_pixel_intensity[fabric_code], 10_000) 


    # enforcing order for labels
    keys = list(fabric_code_to_pixel_intensity.keys())
    values = list([fabric_code_to_pixel_intensity[key] for key in keys])
    positions = np.arange(0, len(keys), 1)
    ax = plt.subplot(111)
    plt.title("Intensities by Fabric Type")
    
    violins = plt.violinplot(values, positions=positions, showmeans=True)

    
    ax.set_xticks(positions)
    ax.set_yticks([0, 255])
    label_texts = []
    for i, body in enumerate(violins["bodies"]):
        body.set_facecolor(colors[i])
        label_texts.append(keys[i])
    ax.set_xticklabels(label_texts, rotation = 90)
    plt.show()

def average_label_size_by_class():
    defect_image_dir = TrainingNNConfig.data_dir / "Defect_images"
    mask_image_dir = TrainingNNConfig.data_dir / "Mask_images"
    
    defect_image_files = list(defect_image_dir.glob("*"))
    defect_class_to_average_label_perc = defaultdict(float)
    defect_class_to_num_viable_files = defaultdict(int)
    defect_class_to_image_files = defaultdict(list)
    
    for defect_image_file in defect_image_files:
        defect_class_to_image_files[int(defect_image_file.name.split("_")[1])].append(defect_image_file)
    
    image_data = None
    
    for class_label, class_defect_image_files in defect_class_to_image_files.items():
        # if class_label not in TrainingNNConfig.kept_classes:
        #     continue
        for defect_image_file in class_defect_image_files:
            # Get mask file
            image_id = defect_image_file.name.split("_")[0]
            
            # Find defect mask
            mask_image_files = list(mask_image_dir.glob(f"{image_id}*"))
            if len(mask_image_files) == 0:
                print(f"no mask { class_label}:{image_id}")
                continue
            mask_image_file = mask_image_files[0]
            mask_image = open_image(mask_image_file)
            image_data = np.array(mask_image)
            if len(image_data.shape) == 3:
                image_data = image_data[:, :, 0]

            defect_class_to_average_label_perc[class_label] += (image_data!=0).sum() / image_data.size
            defect_class_to_num_viable_files[class_label] += 1
    for class_label, unnormalized_label_perc in defect_class_to_average_label_perc.items():
        defect_class_to_average_label_perc[class_label] /= defect_class_to_num_viable_files[class_label]
    # enforcing order for labels
    keys = list(defect_class_to_average_label_perc.keys())
    values = list([defect_class_to_average_label_perc[key] * 100 for key in keys])
    positions = np.arange(0, len(keys), 1)
    ax = plt.subplot(111)
    
    barss = plt.bar(positions,values,color=colors)
    
    ax.set_xticks(positions)
    ax.set_yticks([0, max(values)])
    ax.set_xticklabels( [TrainingNNConfig.class_label_to_name[keys[i]] for i in range(len(keys))], rotation = 90)
    ax.set_title("Average Defect Size per Class")
    plt.ylabel("Average % Of Image That Is Defect")
    plt.show()

def average_label_bb_size_by_class():
    defect_image_dir = TrainingNNConfig.data_dir / "Defect_images"
    mask_image_dir = TrainingNNConfig.data_dir / "Mask_images"
    defect_image_files = list(defect_image_dir.glob("*"))
    defect_class_to_average_label_perc = defaultdict(float)
    defect_class_to_num_viable_files = defaultdict(int)
    defect_class_to_image_files = defaultdict(list)
    
    for defect_image_file in defect_image_files:
        defect_class_to_image_files[int(defect_image_file.name.split("_")[1])].append(defect_image_file)
    
    image_data = None
    
    for class_label, class_defect_image_files in defect_class_to_image_files.items():
        # if class_label not in TrainingNNConfig.kept_classes:
        #     continue
        for defect_image_file in class_defect_image_files:
            # Get mask file
            image_id = defect_image_file.name.split("_")[0]
            
            # Find defect mask
            mask_image_files = list(mask_image_dir.glob(f"{image_id}*"))
            if len(mask_image_files) == 0:
                print(f"no mask { class_label}:{image_id}")
                continue
            mask_image_file = mask_image_files[0]
            mask_image = open_image(mask_image_file)
            image_data = np.array(mask_image)
            if len(image_data.shape) == 3:
                image_data = image_data[:, :, 0]
            y, x = np.where(image_data!=0)  # ~ negates the boolean array
            if y.size == 0: continue
            x_min = x.min()
            x_max = x.max()+1
            y_min = y.min()
            y_max = y.max()+1
            rect_area = (x_max - x_min) * (y_max - y_min)
            defect_class_to_average_label_perc[class_label] += (image_data!=0).sum() / rect_area
            defect_class_to_num_viable_files[class_label] += 1
    for class_label, unnormalized_label_perc in defect_class_to_average_label_perc.items():
        defect_class_to_average_label_perc[class_label] /= defect_class_to_num_viable_files[class_label]
    # enforcing order for labels
    keys = list(defect_class_to_average_label_perc.keys())
    values = list([defect_class_to_average_label_perc[key] * 100 for key in keys])
    positions = np.arange(0, len(keys), 1)
    ax = plt.subplot(111)
    
    barss = plt.bar(positions,values,color=colors)
    
    ax.set_xticks(positions)
    ax.set_yticks([0, 100])
    ax.set_xticklabels( [TrainingNNConfig.class_label_to_name[keys[i]] for i in range(len(keys))], rotation = 90)
    ax.set_title("Defect Class Roundness")
    plt.ylabel("Average % Of Bounding Box That Is Defect")
    plt.show()

def test_image_sizes():
    defect_image_dir = TrainingNNConfig.data_dir / "Defect_images"
    no_defect_image_dir = TrainingNNConfig.data_dir / "NODefect_images"
    all_files = [path for path in defect_image_dir.glob("**/*.png") ]
    all_files += [path for path in no_defect_image_dir.glob("**/*.png")]
    sizes = []
    for file in all_files:
        image = open_image(file)
        sizes.append(image.size)
    print("All Xs=4096:",sum([size[0]==4096 for size in sizes])/len(sizes)*100)
    print("All Ys=256:",sum([size[1]==256 for size in sizes])/len(sizes)*100)