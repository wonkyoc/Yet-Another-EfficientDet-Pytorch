import os
import json



annon_path = "/home/wonkyoc/git/Yet-Another-EfficientDet-Pytorch/datasets/argoverse/annotations/"
TARGET = 17



def main():
    with open(annon_path + "instances_val.json", "r") as f:
        data = json.loads(f.read())

    """
    data.keys()
        : categories, images, annotations, sequences, seq_dirs, coco_subset,
        coco_mapping, n tracks
    """
    new_dict = {}

    for key in data.keys():
        new_dict[key] = []

    print(data.keys())
    new_dict["categories"] = data["categories"]

    for image in data["images"]:
        if image["sid"] == TARGET:
            new_dict["images"].append(image)

    for annon in data["annotations"]:
        if annon["image_id"] >= 11332 and annon["image_id"] <= 11806:
            new_dict["annotations"].append(annon)
            if annon["category_id"] == 0:
                print(annon)

    new_dict["sequences"] = data["sequences"][TARGET]
    new_dict["seq_dirs"] = data["seq_dirs"][TARGET]
    new_dict["coco_subset"] = data["coco_subset"]
    new_dict["coco_mapping"] = data["coco_mapping"]
    new_dict["n_tracks"] = data["n_tracks"]


    #with open(annon_path + "instances_sid_17.json", "w") as f:
    #    json.dump(new_dict, f)



if __name__ == "__main__":
    main()
