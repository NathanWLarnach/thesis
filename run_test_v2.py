import argparse
import datetime
import random
import time
from pathlib import Path
import csv
import pandas as pd
import math

import scipy.optimize
import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings

warnings.filterwarnings('ignore')

# New
import glob

#
# def areEqual(arr1, arr2):
#     # Sort both arrays
#     arr1.sort()
#     arr2.sort()
#
#     # Linearly compare elements
#     for i in range(0, range(len(arr1.shape))):
#         if (arr1[i] != arr2[i]):
#             return False
#
#     # If all elements were same.
#     return True

def euclidDistance(list1, list2):
    sum = 0
    for x, y in zip(list1, list2):
        sum += (x - y) ** 2
    return (sum) ** (1 / 2)

#Old
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='C:/Users/natha/Documents/P2PNet-Multihead/Trajectories',
                        help='path where to save')
    parser.add_argument('--weight_path',
                        default='C:/Users/natha/Documents/ModelWeights/noosa_person_type/noosa_person_type/best_mae.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser


def main(args, debug=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Use glob library to load in noosa data frame
    # for each frame get some number of x y data points
    # set your image path here

    # OLD CODE
    # img_path = "./vis/29_01_2022_9_56_17_00009.png"
    # # load the images
    # img_raw = Image.open(img_path).convert('RGB')
    # # round the size

    # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    # # pre-proccessing
    # img = transform(img_raw)

    # Load in folder of raw video frame
    im = []
    raw= []
    for image in glob.glob(
            'C:/Users/natha/OneDrive - Queensland University of Technology/SLSQ_StudentProjects/NoosaDataFrames/subset_frames/*.jpg'):
        img_raw = Image.open(image).convert('RGB')
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        raw.append(img_raw)
        img = transform(img_raw)
        im.append(img)
    # Create Nested list of points, video frame, and classification of count
    points_agg = [[]]
    for img in range(len(im)):
        samples = torch.Tensor(im[img]).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        for i in range(1, 4):
            outputs = model(samples)[i]
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1]
            outputs_points = outputs['pred_points']
            # print(outputs_points)
            threshold = 0.5
            # filter the predictions
            points = outputs_points[outputs_scores > threshold, :].detach().cpu().numpy().tolist()
            predict_cnt = int((outputs_scores > threshold).sum())
            points_agg.append(points)
            points_agg.append(img)
            points_agg.append(i)

    # Convert list of lists to dataframe with video frame and class for each point
    df = pd.DataFrame(columns=["X", "Y", "Frame", "Class"])
    for i in range(len(points_agg)):
        if i % 3 == 1:
            # print(points_agg[i])
            for j in range(len(points_agg[i])):
                x = points_agg[i][j][0]
                y = points_agg[i][j][1]
                frame = points_agg[i + 1]
                loc = points_agg[i + 2]
                df = df.append({'X': x,
                                'Y': y,
                                'Frame': frame,
                                'Class': loc}, ignore_index=True)
    # Write to CSV = 'out.csv'
    df.to_csv('out.csv')

    # take all instance in f1 and f2
    # build matrix with euclidean distance

    mat = pd.read_csv('out.csv')
    size = 2
    threshold = 25
    viabletraj = [[]]
    for frame in range(1, len(im)):
        currx = (mat['X'].loc[mat['Frame'] == frame]).to_numpy()
        curry = (mat['Y'].loc[mat['Frame'] == frame]).to_numpy()
        curr = np.stack((currx, curry), axis=1)
        prevx = (mat['X'].loc[mat['Frame'] == frame - 1]).to_numpy()
        prevy = (mat['Y'].loc[mat['Frame'] == frame - 1]).to_numpy()
        prev = np.stack((prevx, prevy), axis=1)
        cost = np.zeros((curr.shape[0], prev.shape[0]))

        for i in range(prev.shape[0]):
            for j in range(curr.shape[0]):
                dist = (euclidDistance(prev[i], curr[j]))
                cost[j][i] = dist
        traj = np.array(scipy.optimize.linear_sum_assignment(cost))

        viabletraj = []
        viabletraj.append([])
        viabletraj.append([])
        for i in range(traj.shape[1]):
            if cost[traj[0][i], traj[1][i]] < threshold:
                viabletraj[0].append(traj[0][i])
                viabletraj[1].append(traj[1][i])

        img_to_draw = cv2.cvtColor(np.array(raw[frame]), cv2.COLOR_RGB2BGR)
        for k in range(np.array(viabletraj).shape[1]):
            px = int(prev[viabletraj[1][k]][0])
            py = int(prev[viabletraj[1][k]][1])
            cx = int(curr[viabletraj[0][k]][0])
            cy = int(curr[viabletraj[0][k]][1])
            # cv2 drawline (Maybe draw arrow)
            # draw the predictions
            img_to_draw = cv2.circle(img_to_draw, (px, py), 4, (255, 0, 0), 1)
            img_to_draw = cv2.circle(img_to_draw, (cx, cy),4, (0, 255, 0), 1)
            img_to_draw = cv2.arrowedLine(img_to_draw, (px, py), (cx, cy), (0, 0, 255), 1, tipLength = 0.01)
        cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(frame)), img_to_draw)

    frameSize = (width, height)
    fps = 2
    #Write images back to video file
    img_array = []
    for filename in glob.glob('C:/Users/natha/Documents/P2PNet-Multihead/Trajectories/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('C:/Users/natha/Documents/P2PNet-Multihead/processedFootage/output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 0.5, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
