import copy
import io
import argparse
import os

from scipy import spatial
import matplotlib.pyplot as plt
from skimage import feature
from skimage import io as skio
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
from PIL import Image

import utils
import cv2
from tqdm import tqdm
import pickle
from timm.models import create_model

import albumentations.pytorch
import albumentations as A
import pydegensac

import xcit_retrieval
from reranking_script.reranking_transformer import RerankingTransformer



def compute_putative_matching_keypoints(test_keypoints,
                                        test_descriptors,
                                        train_keypoints,
                                        train_descriptors,
                                        use_ratio_test,
                                        matching_threshold,
                                        max_distance):
    """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""
    train_descriptor_tree = spatial.cKDTree(train_descriptors)
    if use_ratio_test:
        distances,matches=train_descriptor_tree.query(
            test_descriptors,k=2,n_jobs=-1
        )
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        test_matching_keypoints=np.array([
            test_keypoints[i,]
            for i in range(test_kp_count)
            if distances[i][0] < matching_threshold*distances[i][1]
        ])
        train_matching_keypoints=np.array([
            train_keypoints[matches[i][0],]
            for i in range(test_kp_count)
            if distances[i][0] < matching_threshold*distances[i][1]
        ])

    else:
        dist, matches = train_descriptor_tree.query(
              test_descriptors, distance_upper_bound=max_distance)
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        test_matching_keypoints = np.array([
              test_keypoints[i,]
              for i in range(test_kp_count)
              if matches[i] != train_kp_count
          ])
        train_matching_keypoints = np.array([
              train_keypoints[matches[i],]
              for i in range(test_kp_count)
              if matches[i] != train_kp_count
          ])
    # print(len(test_matching_keypoints))
    return test_matching_keypoints, train_matching_keypoints 


def get_reranking_model_input(sample1_descriptor, sample1_location, sample1_scale, sample1_score,
                             sample2_descriptor, sample2_location, sample2_scale, sample2_score):
    def padding(descriptor,location,scale,score):
        sample_length,sample_dim = descriptor.shape
        max_length = 2000
        mask = torch.ones((max_length))
        if sample_length == max_length:
            score = torch.unsqueeze(score, 1)
            return descriptor,location, scale.type(torch.LongTensor),score,mask
        elif sample_length > max_length:
            score = torch.unsqueeze(score, 1)
            scale = np.int64(scale)
            return descriptor[:max_length,:], location[:max_length,:], scale[:max_length].type(torch.LongTensor), \
                    score[:max_length,:],mask
        elif sample_length < max_length:
            mask[sample_length:] = 0
            pad_descriptor,pad_location,pad_scale,pad_score = torch.zeros((max_length,sample_dim)), \
                        torch.randint(1024,(max_length,2)), torch.randint(7,(max_length,)), torch.zeros((max_length,))
            pad_descriptor[:sample_length,:] = descriptor
            pad_location[:sample_length,:] = location
            pad_scale[:sample_length] = scale
            pad_score[:sample_length] = score
            pad_score = torch.unsqueeze(pad_score, 1)
            return pad_descriptor,pad_location,pad_scale,pad_score,mask
    sample1_descriptor, sample1_location, sample1_scale, sample1_score = torch.tensor(sample1_descriptor), \
     torch.tensor(sample1_location), torch.tensor(sample1_scale), torch.tensor(sample1_score)
    sample2_descriptor, sample2_location, sample2_scale, sample2_score = torch.tensor(sample2_descriptor), \
     torch.tensor(sample2_location), torch.tensor(sample2_scale), torch.tensor(sample2_score)

    sample1_descriptor, sample1_location, sample1_scale, sample1_score, mask1 = padding(sample1_descriptor, 
                                                                        sample1_location, sample1_scale,sample1_score)
    sample2_descriptor, sample2_location, sample2_scale, sample2_score, mask2 = padding(sample2_descriptor, 
                                                                            sample2_location, sample2_scale,sample2_score)
    device = 'cuda'
    sample1_descriptor, sample1_location, sample1_scale, sample1_score, mask1, sample2_descriptor, \
    sample2_location, sample2_scale, sample2_score, mask2 =  sample1_descriptor.to(device, non_blocking=True), \
    sample1_location.to(device, non_blocking=True), sample1_scale.to(device,
    non_blocking=True),sample1_score.to(device, non_blocking=True),mask1.to(device, non_blocking=True), \
    sample2_descriptor.to(device, non_blocking=True), sample2_location.to(device,non_blocking=True), \
    sample2_scale.to(device, non_blocking=True),sample2_score.to(device, non_blocking=True), \
    mask2.to(device, non_blocking=True)
    
    sample1_descriptor, sample1_location, sample1_scale, sample1_score, mask1, sample2_descriptor, \
    sample2_location, sample2_scale, sample2_score, mask2 =  sample1_descriptor.unsqueeze(0), \
    sample1_location.unsqueeze(0), sample1_scale.unsqueeze(0),sample1_score.unsqueeze(0), \
    mask1.unsqueeze(0),sample2_descriptor.unsqueeze(0),sample2_location.unsqueeze(0), \
    sample2_scale.unsqueeze(0),sample2_score.unsqueeze(0),mask2.unsqueeze(0)
    
    return sample1_descriptor, sample1_location, sample1_scale, sample1_score, mask1, sample2_descriptor, \
    sample2_location, sample2_scale, sample2_score, mask2

def compute_num_inliers(test_keypoints, 
                        test_descriptors, 
                        train_keypoints,
                        train_descriptors,
                        max_reprojection_error,
                        homography_confidence,
                        max_ransac_iteration,
                        use_ratio_test,
                        matching_threshold,
                        max_distance,
                        query_im_array=None,
                        index_im_array=None,
                        test_scales=None, test_scores=None, 
                        train_scales=None, train_scores=None,
                        reranking_model=None):
    """Returns the number of RANSAC inliers."""
    if reranking_model != None:
        sample1_descriptor, sample1_location, sample1_scale, sample1_score, mask1, sample2_descriptor, \
        sample2_location, sample2_scale, sample2_score, mask2 = get_reranking_model_input(test_descriptors,test_keypoints,
              test_scales,test_scores,train_descriptors,train_keypoints,train_scales,train_scores)
        score = reranking_model(sample1_descriptor, sample1_location,
            sample1_scale, sample1_score,sample2_descriptor,sample2_location, sample2_scale, sample2_score,mask1, mask2)
        score = score[0].cpu().data.numpy()
        score = 1/(1 + np.exp(-score))
        print(score)
        test_descriptors, train_descriptors = reranking_model.forward_feature(sample1_descriptor, sample1_location,
            sample1_scale, sample1_score,sample2_descriptor,sample2_location, sample2_scale, sample2_score,mask1, mask2)
        test_descriptors,train_descriptors = test_descriptors[0].cpu().data.numpy(),train_descriptors[0].cpu().data.numpy()
        test_descriptors,train_descriptors=test_descriptors[:len(test_keypoints)],train_descriptors[:len(train_keypoints)]
        # test_descriptors,train_descriptors=test_descriptors/np.expand_dims(np.linalg.norm(test_descriptors, axis=1),1),train_descriptors/np.expand_dims(np.linalg.norm(train_descriptors, axis=1),1)
        
    test_match_kp, train_match_kp = \
            compute_putative_matching_keypoints(test_keypoints, 
                                                test_descriptors, 
                                                train_keypoints, 
                                                train_descriptors,
                                                use_ratio_test=use_ratio_test,
                                                matching_threshold = matching_threshold,
                                                max_distance =  max_distance,
                                                )
    if test_match_kp.shape[0] <= 4:  
        return 0, b''

    try:
        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                            max_reprojection_error,
                                            homography_confidence,
                                            max_ransac_iteration)
    except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
        return 0, b''

    inliers = mask if mask is not None else []

    match_viz_bytes = b''
    if isinstance(query_im_array, np.ndarray) and isinstance(index_im_array, np.ndarray) :
        query_im_scale_factors = [1.0, 1.0]
        index_im_scale_factors = [1.0, 1.0]
        inlier_idxs = np.nonzero(inliers)[0]
        _, ax = plt.subplots()
        ax.axis('off')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        feature.plot_matches(
            ax,
            query_im_array,
            index_im_array,
            test_match_kp * query_im_scale_factors,
            train_match_kp * index_im_scale_factors,
            np.column_stack((inlier_idxs, inlier_idxs)),
            only_matches=True)

        match_viz_io = io.BytesIO()
        plt.savefig(match_viz_io, format='jpeg', bbox_inches='tight', pad_inches=0)
        match_viz_bytes = match_viz_io.getvalue()
    
    return int(copy.deepcopy(mask).astype(np.float32).sum()), match_viz_bytes

def extract_local_descriptors(model, input, patch_size, attn_threshold, min_attn_threshold, max_key_point, multiscale=None):
    if multiscale == None:
        multiscale = [1]
    selected_locations = []
    selected_features = []
    selected_scales = []
    all_local_scores = []
    _,_, input_h, input_w = input.shape
    for scale in multiscale:
        resize_h, resize_w = input_h * scale, input_w * scale
        resize_h, resize_w = ((resize_h -1 ) // patch_size + 1) * patch_size, ((resize_w -1 ) // patch_size + 1) * patch_size
        scale_h, scale_w = resize_h/input_h, resize_w/input_w
        input_scale = nn.functional.interpolate(input, scale_factor=(scale_h, scale_w), mode='bilinear', align_corners=False)
        with torch.no_grad():
            _, attentions, embeddings, _ = model.forward_features(input_scale.to(device))
        nh = attentions.shape[1] # number of head
        h_featmap = input_scale.shape[-2] // patch_size
        w_featmap = input_scale.shape[-1] // patch_size
        attentions = attentions[0]
        attentions = attentions/torch.max(attentions,dim=1).values.unsqueeze(1)
        attentions = torch.mean(attentions, dim =0)
        attentions = attentions / torch.sum(attentions)
        attentions = attentions * len(attentions)

        scale_attn_threshold = attn_threshold
        indices = None
        max_kp_scale = max_key_point * (scale_h * scale_w)
        while(indices is None or len(indices) == 0 or len(indices) < max_kp_scale):
            indices = torch.gt(attentions, scale_attn_threshold).nonzero().squeeze()
            try :
                len(indices)
            except:
                indices = None
            scale_attn_threshold = scale_attn_threshold * 0.5   # use lower threshold if no indexes are found.
            if scale_attn_threshold < min_attn_threshold:
                break
        embeddings = embeddings[0]
        embeddings = nn.functional.normalize(embeddings, dim=1, p=2)
        selected_features.extend(list(torch.index_select(embeddings, dim=0, index=indices).cpu().numpy()))
        all_local_scores.extend(list(torch.index_select(attentions, dim= 0, index=indices).cpu().numpy()))
        for indice in indices.cpu().numpy():
            indice_h = indice // w_featmap
            indice_w = indice % w_featmap
            location_w = int((indice_w * patch_size + patch_size // 2) / scale_w)
            location_h = int((indice_h * patch_size + patch_size // 2) / scale_h)
            selected_locations.append([location_h,location_w])
            selected_scales.append(scale)
    top_2k_index = np.argsort(np.array(all_local_scores))[::-1][:2000]
    
    return np.array(selected_locations)[top_2k_index], np.array(selected_features)[top_2k_index], \
            np.array(all_local_scores)[top_2k_index], np.array(selected_scales)[top_2k_index]

def preprocess_image(args, im_path, image_size = None) :
    patch_size = args.patch_size
    if image_size is None :
        image_size = args.image_size
    
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

            
    im_h, im_w, _ = np.array(img).shape
    resize_h, resize_w = image_size * im_h // max(im_h,im_w),  image_size * im_w // max(im_h,im_w)
    resize_h, resize_w = resize_h - resize_h % patch_size, resize_w - resize_w % patch_size # divisible by the patch size
    resize_size = (resize_h, resize_w) if  args.keep_aspec_ratio else (image_size,image_size)
    img_numpy = cv2.resize(np.array(img),resize_size[::-1])

    transform = A.Compose([
                    A.Resize(resize_size[0],resize_size[1]),
                    A.Normalize(),
                    A.pytorch.transforms.ToTensorV2()
                ])    
    img = transform(image=img)['image'].cpu()
    img = img.unsqueeze(0)
    return img_numpy, img
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Computing match pair')
    # model config
    parser.add_argument('--model', default='xcit_retrievalv2_small_12_p16', type=str, metavar='MODEL',help='')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained', default='../model_checkpoint/xcit_small_12_p16_retrieval_5e-5-1e-7_19_tune_aug/checkpoint_model_epoch_50.pth',type=str, help="Path to pretrained weights to load.")
    parser.add_argument("--use_reranking_model", default=False, type=utils.bool_flag)
    parser.add_argument("--reranking_model", 
                default='reranking_script/model_checkpoint/reranking_loftr/checkpoint.pth', type=str)
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--d_original', default=384, type=int)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--layer_names', default=['self','cross'] * 6, type=list)
                        
    
    # io config
    parser.add_argument("--keep_aspec_ratio", default=False, type=utils.bool_flag, help="Keep aspec_ratio.")
    parser.add_argument("--use_multi_scale", default=True, type=utils.bool_flag, help="Keep aspec_ratio.")
    # parser.add_argument("--querry_image_path", default='', type=str, help="Path to image 1") 
    # parser.add_argument("--galery_image_paths", default='', type=str, help="Path to image 2")
    parser.add_argument('--data_path', default='../test_datasets', type=str)
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--image_root_paths', default='../test_datasets/roxford5k/jpg', help='')
    parser.add_argument("--image_size", default= 512, type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='MatchPairVisv2', help='Path where to save visualizations.')
    parser.add_argument('--descriptor_output_dir', default='', help='') #LocalFeaturePath/Descriptions
    parser.add_argument('--location_output_dir', default='', help='') #LocalFeaturePath/Locations

    # ransac config
    parser.add_argument("--max_reprojection_error", default=18, type=float)
    parser.add_argument("--max_ransac_iteration", default=2000, type=int)
    parser.add_argument("--homography_confidence", default=1.0, type=float)
    parser.add_argument("--matching_ratio_threshold", default=0.93, type=float)
    parser.add_argument("--max_distance", default=0.79, type=float)
    parser.add_argument("--use_ratio_test", default=False, type=utils.bool_flag)
    parser.add_argument("--attn_threshold", default = 60, type=float)
    parser.add_argument("--min_attn_threshold", default = 0.6 , type=float)
    parser.add_argument("--max_keypoint", default=400, type=int)
    
    device = torch.device("cuda") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parser.parse_args()
    
#     model = create_model(
#         args.model,
#         pretrained=False,
#         num_classes=1,
#     )

#     checkpoint = torch.load(args.pretrained, map_location='cpu')
#     checkpoint_model = checkpoint['model']
#     state_dict = model.state_dict()
#     for k in ['head.weight', 'local_head.weight','local_head.bias']:
#         if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del checkpoint_model[k]
#     model.load_state_dict(checkpoint_model, strict=False)
#     device = torch.device("cuda")
#     model.to(device)
#     model.eval()
    
#     reranking_model = None
#     if args.use_reranking_model:
#         reranking_model = RerankingTransformer(args)
#         checkpoint = torch.load(args.reranking_model, map_location='cpu')
#         checkpoint_model = checkpoint['model']
#         reranking_model.load_state_dict(checkpoint_model, strict=True)
#         device = torch.device("cuda")
#         reranking_model.to(device)
#         reranking_model.eval()
                        
#     galery_paths = []
#     all_paths = []
#     pair_descriptors = []
#     pair_locations = []
#     pair_scales = []
#     pair_scores = []
#     pair_imgs = []
#     # 0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0
#     multi_scale = [0.25,0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0] if args.use_multi_scale else None
#     # specific query and galery path
#     idx = 0
#     assert args.data_path != '', " not find data path"
#     gnd_fname = os.path.join(args.data_path, args.dataset, 'gnd_{}.pkl'.format(args.dataset))
#     with open(gnd_fname, 'rb') as f:
#         cfg = pickle.load(f)
#     querry_image_path = os.path.join(args.data_path, args.dataset, 'jpg', cfg["qimlist"][idx] + ".jpg")
#     gt_for_querry_image_paths =  np.concatenate([cfg['gnd'][idx]['hard']]) 
#     print('num hard', len(cfg['gnd'][idx]['hard']))
#     all_paths.append(querry_image_path)
#     for im_path in gt_for_querry_image_paths: 
#         all_paths.append(os.path.join(args.data_path, args.dataset, 'jpg', cfg["imlist"][int(im_path)] + ".jpg"))
    all_paths = ['train_8_a_d_8adbe0d19e3183a2' ,'train_5_6_9_569554b444f74917'] ##'
    galery_paths = []
    pair_descriptors = []
    pair_locations = []
    pair_scales = []
    pair_scores = []
    pair_imgs = []
    reranking_model = None
    for im_path in tqdm(all_paths):
        # if args.location_output_dir != '' and args.descriptor_output_dir != '' :
        #     location_path = im_path.split('/')[-1].split('.')[0] + '.npy'
        #     location = np.load(args.location_output_dir + '/' + location_path )
        #     description = np.load(args.descriptor_output_dir + '/' + location_path)
        # else:
        #     # im_path = '../gglandmark-v2-clean/' + im_path.replace('_','/') + '.jpg'
        #     _ ,img_input = preprocess_image(args, im_path)
        #     location, description, scores, scales = extract_local_descriptors(model, img_input, 
        #             args.patch_size,args.attn_threshold,args.min_attn_threshold, args.max_keypoint,multi_scale)
            
        
        ##
        base_path = os.path.join('reranking_script/reranking_data/val_local_descriptor',im_path)
        description = np.load(os.path.join(base_path,'descriptor.npz'))['arr_0']
        location = np.load(os.path.join(base_path,'location.npz'))['arr_0']
        scales = np.load(os.path.join(base_path,'scale.npz'))['arr_0']
        scores = np.load(os.path.join(base_path,'score.npz'))['arr_0']
        im_path = '../gglandmark-v2-clean/' + im_path.replace('_','/') + '.jpg'
        ##
        
        pair_descriptors.append(description)
        pair_locations.append(location)
        pair_imgs.append(im_path)
        # pair_imgs.append('../gglandmark-v2-clean/' + im_path.replace('_','/') + '.jpg') ##
        pair_scales.append(scales)
        pair_scores.append(scores)



    querry_location = pair_locations[0]
    querry_description = pair_descriptors[0]
    querry_scales = pair_scales[0]
    querry_scores = pair_scores[0]
    quer_im_path = pair_imgs[0]
    query_img, _ = preprocess_image(args,quer_im_path) 

    for idx in range(len(pair_descriptors) -1):
        gal_img, _ = preprocess_image(args, pair_imgs[idx + 1]) 
        num_inliers, match_vis_bytes = compute_num_inliers(querry_location, querry_description,
                                                        pair_locations[idx + 1], pair_descriptors[idx + 1],
                                                        max_reprojection_error = args.max_reprojection_error,
                                                        homography_confidence = args.homography_confidence,
                                                        max_ransac_iteration = args.max_ransac_iteration,
                                                        use_ratio_test = args.use_ratio_test,
                                                        matching_threshold = args.matching_ratio_threshold,
                                                        max_distance = args.max_distance,
                                                        query_im_array = query_img,
                                                        index_im_array = gal_img,
                                                        test_scales=querry_scales,test_scores=querry_scores, 
                                                        train_scales=pair_scales[idx + 1],
                                                        train_scores=pair_scores[idx + 1],
                                                        reranking_model=reranking_model)
        print(idx, num_inliers)
        with open(os.path.join(args.output_dir, f'save_match_{idx}.jpg'),"wb") as fout:
            fout.write(match_vis_bytes)