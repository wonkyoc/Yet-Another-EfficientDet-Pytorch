# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import cv2
import argparse
import torch
import torchvision
import yaml
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
#from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box, boolean_string


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

# wk
VIS_PATH = ''
preds_dic = {}

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']


color_list = standard_to_bgr(STANDARD_COLORS)

def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.15):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # WK: measure latency
    import time
    start = time.time()
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]

        # only cares about 1d6767 data
        if image_info["sid"] != 17:
            continue
        #image_path = img_path + image_info['file_name']
        image_path = img_path + image_info['name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        preds_dic[compound_coef][image_id] = preds
        #_display([preds], ori_imgs, image_info, imshow=False, imwrite=True)


        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']
        temp = rois.copy()

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'image_name': image_info['name'],
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)
        preds['rois'] = temp

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    end = time.time()
    runtime = end - start
    output_path = f"{project_name}_d{compound_coef}.csv"
    print(runtime)
    with open(output_path, "w") as f:
        f.write(str(runtime))

    # write output
    filepath = f'{set_name}_{compound_coef}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

def _display(preds_list, imgs, image_info, imshow=True, imwrite=False):
    imgs = imgs[0].copy()

    selected = ["person", "bicycle", "car", "bus", "truck", "traffic light",
                "stop sign"]

    color_counter = 0
    for preds in preds_list:
        for j in range(len(preds['rois'])):
            x1, y1, x2, y2 = preds['rois'][j].astype(np.int)
            obj = obj_list[preds['class_ids'][j]]

            if obj not in selected:
                continue

            score = float(preds['scores'][j])
            plot_one_box(imgs, [x1, y1, x2, y2], label=obj,
                         score=score,
                         #color=color_list[get_index_label(obj, obj_list)])
                         color=color_list[color_counter])
        color_counter += 4

    if imshow:
        cv2.imshow('img', imgs)
        cv2.waitKey(0)

    if imwrite:
        output_path = f'{VIS_PATH}/{image_info["name"]}'
        if not cv2.imwrite(output_path, imgs):
            raise Exception("image write error")


if __name__ == '__main__':
    SET_NAME = params['val_set']
    VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/input_{SET_NAME}_3x3_filtered/'
    VIS_PATH = f'datasets/{params["project_name"]}/output_sid_17_d{compound_coef}_3x3_filtered'
    MAX_IMAGES = 16000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    preds_dic[compound_coef] = {}
    if override_prev_results or not os.path.exists(
            f'{project_name}_{compound_coef}_bbox_results.json'):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    # big detector
    #compound_coef = 4
    #weights_path = f'weights/efficientdet-d{compound_coef}.pth'
    #print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

    #preds_dic[compound_coef] = {}
    #if override_prev_results or not os.path.exists(
    #        f'{project_name}_{compound_coef}_bbox_results.json'):
    #    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
    #                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    #    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    #    model.requires_grad_(False)
    #    model.eval()

    #    if use_cuda:
    #        model.cuda(gpu)

    #        if use_float16:
    #            model.half()

    #    evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    # extract inference data and make pics
    img_path = VAL_IMGS
    for image_id in tqdm(image_ids):
        image_info = coco_gt.loadImgs(image_id)[0]

        # only cares about 1d6767 data
        if image_info["sid"] != 17:
            continue
        #image_path = img_path + image_info['file_name']
        image_path = img_path + image_info['name']
        #preds_list = [preds_dic[0][image_id], preds_dic[4][image_id]]
        preds_list = [preds_dic[compound_coef][image_id]]

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
        _display(preds_list, ori_imgs, image_info, imshow=False, imwrite=True)

    #_eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')
