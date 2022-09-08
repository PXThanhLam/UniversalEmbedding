import os
import sys
import pickle
import argparse

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from PIL import Image, ImageFile
import numpy as np

from timm.models import create_model

import albumentations.pytorch
import albumentations as A
import utils
import xcit_retrieval
import patchconv_retrieval
import gc
from tqdm import tqdm
from match_pair import compute_num_inliers, preprocess_image
import cv2
import numpy as np
from multiprocessing import Pool
import copy
from PIL import Image
from clip_model import ClipvitlModel2, ClipvitlModelReduction3, ClipvitlModelReduction2

class OxfordParisDataset(torch.utils.data.Dataset):
    def __init__(self, dir_main, dataset, split, transform=None, imsize=None, keep_ratio = True,
                r1m_path = None):
        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')
        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])
        cfg['im_fname'] = config_imname
        cfg['qim_fname'] = config_qimname
        cfg['dataset'] = dataset
        self.cfg = cfg
        self.split = split

        self.samples = cfg["qimlist"] if split == "query" else cfg["imlist"]
        self.samples = [os.path.join(self.cfg["dir_images"], sample + ".jpg") for sample in self.samples]
        if r1m_path != 'None':
            print('Load r1m distractor')
            for dir in tqdm(os.listdir(r1m_path)):
                for img_folder in os.listdir(r1m_path + '/' + dir):
                    self.samples.append(r1m_path + '/' + dir + '/' + img_folder)

                
        self.transform = transform
        self.imsize = imsize
        self.keep_ratio= keep_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h,w,_ = img.shape
#         ######
#         if self.split == 'query':
#             bbx = np.array(self.cfg['gnd'][index]['bbx'], dtype= np.int32)
#             x1,y1,x2,y2 = bbx
#             cv2.imwrite(f'QUERRY_IMG/querry_box{index}.png',cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#                         (x1,y1), (x2,y2), (0, 0, 255), 2))
#             img = img[y1:y2,x1:x2]
#         #####
#         if self.imsize is not None:
#             if self.keep_ratio:
#                 h,w,_ = img.shape
#                 resize_h = h*self.imsize//max(h,w)
#                 resize_w = w*self.imsize//max(h,w)
#                 img = cv2.resize(img,(resize_w, resize_h))
#             else:
#                 img = cv2.resize(img,(self.imsize, self.imsize))
#         if self.transform is not None:
#             img = self.transform(image=img)['image']
        img = Image.open(path).convert("RGB")
        convert_to_tensor = pth_transforms.Compose([pth_transforms.PILToTensor()])
        img = convert_to_tensor(img)
        if self.split == 'query':
            bbx = np.array(self.cfg['gnd'][index]['bbx'], dtype= np.int32)
            x1,y1,x2,y2 = bbx
#             cv2.imwrite(f'QUERRY_IMG/querry_box{index}.png',cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
#                         (x1,y1), (x2,y2), (0, 0, 255), 2))
            img = img[:,y1:y2,x1:x2]

        img = pth_transforms.functional.resize(img, size = [self.imsize, self.imsize])
        img = img/ 255.0
        img = pth_transforms.functional.normalize(img, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) 

        return img, index

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None

    for samples, index in metric_logger.log_every(data_loader, 5):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model.get_global_feat(samples).clone()
#         feats = nn.AdaptiveAvgPool1d(64)(feats)
        if features == None :
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
        features[index] = feats



    return features
def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])

def localRank(dataset_train,args, tuple_local_features, train_ids, test_ids, ranks_before_gv):
    gnd = dataset_train.cfg['gnd']
    query_name, location_dic,description_dic = tuple_local_features
    print(">> rerank {}".format(query_name))

    i = test_ids.index(query_name)
    locations = location_dic[query_name]
    descriptors = description_dic[query_name]
    inliers_numrerank = np.zeros(args.num_rerank)  
    
    for j in tqdm(range(args.num_rerank)):
        if ranks_before_gv[j, i] in gnd[i]['junk']:
            continue
        index_img = train_ids[ranks_before_gv[j, i]]
        tlocations = location_dic[index_img]
        tdescriptors = description_dic[index_img]
        try:
            num_inliers, match_vis_bytes =  compute_num_inliers(locations, descriptors,
                                                               tlocations, tdescriptors,
                                                        max_reprojection_error = args.max_reprojection_error,
                                                        homography_confidence = args.homography_confidence,
                                                        max_ransac_iteration = args.max_ransac_iteration,
                                                        use_ratio_test = args.use_ratio_test,
                                                        matching_threshold = args.matching_threshold,
                                                        max_distance = args.max_distance,
                                                        query_im_array = '',
                                                        index_im_array = '')
            inliers_numrerank[j] = num_inliers
        except:
            continue
    return i, inliers_numrerank


def rerankGV_mulprocess(args,dataset_train, dataset_query, local_locations, local_descriptions,ranks_before_gv, ranks_after_gv=None):
    print('>> mulprocess reranking ...')
    ranks_after_gv = ranks_before_gv.copy()
    train_ids = dataset_train.samples#[x + dataset_train.cfg['ext'] for x in dataset_train.cfg['imlist']]
    test_ids = dataset_query.samples#[x + dataset_query.cfg['qext'] for x in dataset_query.cfg['qimlist']]

    N_localfeatures = []
    for query_rank in tqdm(range(len(test_ids))):
        query_idx = test_ids[query_rank]
        location_dic = {}
        description_dic = {}
        location_dic[query_idx] = local_locations[query_idx]
        description_dic[query_idx] = local_descriptions[query_idx]
        for k in range(args.num_rerank):
            index_rank = ranks_before_gv[k, query_rank]
            index_idx = train_ids[index_rank]
            location_dic[index_idx] = local_locations[index_idx]
            description_dic[index_idx] = local_descriptions[index_idx]
        N_localfeatures.append((query_idx, location_dic,description_dic))
    
    del local_descriptions
    del local_locations
    # gc.collect()
    num_rerank = len(test_ids)
    input_multi_process = []
    for tuple_fea in N_localfeatures:
        input_multi_process.append((dataset_train, args, tuple_fea, train_ids, test_ids, ranks_before_gv))
    with Pool(12) as p: #paralel 12 process
        reranking_res = p.starmap(localRank, input_multi_process)
    for res in reranking_res:
        
        query_idx, inliers_numrerank = res
        ranks_after_gv[:args.num_rerank, query_idx] = ranks_before_gv[np.argsort(-1 * inliers_numrerank), query_idx]
        
    return ranks_before_gv, ranks_after_gv

def reportMap(dataset_train,args, ranks):
    gnd = dataset_train.cfg['gnd']
        # evaluate ranks
    ks = [1, 5, 10]
    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = utils.compute_map(ranks, gnd_t, ks)
    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = utils.compute_map(ranks, gnd_t, ks)
    print('>> {}: mAP M: {}, H: {}'.format(args.dataset, np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
    print('>> {}: mP@k{} M: {}, H: {}'.format(args.dataset, np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
    if args.save_rank:
        f = open(args.save_rank, "a")
        f.write('>> max_distance : {} \n'.format(args.max_distance*1))
        f.write('>> {}: mAP M: {}, H: {} \n'.format(args.dataset, np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        f.write('>> {}: mP@k{} M: {}, H: {} \n'.format(args.dataset, np.array(ks), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
        
if __name__ == '__main__':
    np.random.seed(8)
    parser = argparse.ArgumentParser('Image Retrieval on revisited Paris and Oxford')
    parser.add_argument('--data_path', default='../../ImageRetrieval/test_datasets', type=str)
    parser.add_argument('--r1m_path', default='../../ImageRetrieval/test_datasets/revisitop1m/jpg', type=str)
    parser.add_argument('--do_reranking', default=False, type=utils.bool_flag)
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--multiscale', default=False, type=utils.bool_flag)
    parser.add_argument('--imsize', default=512, type=int, help='Image size')
    parser.add_argument("--keep_aspec_ratio", default=False, type=utils.bool_flag, help="Keep aspec_ratio.")
    parser.add_argument('--batch_size', default=16, type=int, help='Batch evaluating size.')
    
    parser.add_argument('--from_timm', default=True, type=utils.bool_flag, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model', default='xcit_retrievalv2_small_12_p16', type=str, metavar='MODEL',help='')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--weight_path', 
                        default='model_checkpoint/xcit_small_12_p16_retrieval/checkpoint.pth',
                        type=str, help="Path to pretrained weights to load.")
    
    
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')

    parser.add_argument("--num_rerank", default=100, type=int, help=" number of top global to go rerank")
    parser.add_argument('--save_rank', default=None)

    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--d_original', default=384, type=int)
    parser.add_argument('--nhead', default=4, type=int)
    parser.add_argument('--layer_names', default=['self','cross'] * 6, type=list)
    parser.add_argument('--max_length', default=2000, type=int)

    # ransac config
    parser.add_argument("--max_reprojection_error", default=16, type=float)
    parser.add_argument("--max_ransac_iteration", default=2000, type=int)
    parser.add_argument("--homography_confidence", default=1.0, type=float)
    parser.add_argument("--max_distance", default=0.82, type=float)
    
    parser.add_argument("--matching_threshold", default= 0.99, type=float)
    parser.add_argument("--use_ratio_test", default=False, type=utils.bool_flag)

    args = parser.parse_args()

    # utils.init_distributed_mode2(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = A.Compose([
                    A.Normalize(),
                    A.pytorch.transforms.ToTensorV2()
        ])   
    dataset_train = OxfordParisDataset(args.data_path, args.dataset, split="train", transform=transform, imsize=args.imsize, keep_ratio= args.keep_aspec_ratio, r1m_path = args.r1m_path) #args.r1m_path
    dataset_query = OxfordParisDataset(args.data_path, args.dataset, split="query", transform=transform, imsize=args.imsize, keep_ratio= args.keep_aspec_ratio, r1m_path = 'None')
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    print(f"train: {len(dataset_train)} imgs / query: {len(dataset_query)} imgs")

    # ============ building network ... ============
    if args.from_timm:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=0,
            reduct = None
        )

        state_dict = model.state_dict()
        checkpoint_model = torch.load(args.weight_path, map_location='cpu')
        checkpoint_model = checkpoint_model['model']
        for k in ['head.kernel.weight']:  
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
        model.load_state_dict(checkpoint_model, strict=True)
        device = torch.device("cuda")
        model.to(device)
        model.eval()

    #     model_backbone = create_model(
    #         args.model,
    #         pretrained=False,
    #         num_classes=0,
    #     )
    #     model = create_model(
    #         'xcit_retrievalv2_reduction',
    #         pretrained=False,
    #         num_classes=0,
    #         retrieval_back_bone = model_backbone,
    #         drop_block_rate=None,
    #     )

    #     device = torch.device("cuda")
    #     resume_path= args.weight_path
    #     checkpoint = torch.load(resume_path, map_location='cpu')['model']
    #     for k in ['head.kernel', 'head.t']:
    #         if k in checkpoint.keys():
    #             print(f"Removing key {k} from pretrained model checkpoint")
    #             del checkpoint[k]
    #     model.load_state_dict(checkpoint)
    #     model.to(device)
    #     model.eval()

    else:
#         model = ClipvitlModel2(n_class = 0)
#         checkpoint_model = torch.load(args.weight_path, map_location='cpu')
#         checkpoint_model = checkpoint_model
# #         for k in ['head.kernel.weight']:   #['head.weight']
# #             print(f"Removing key {k} from pretrained checkpoint")
# #             del checkpoint_model[k]
#         model.load_state_dict(checkpoint_model, strict=True)
#         device = torch.device("cuda")
#         model.to(device)
#         model.eval()
        
        model_backbone = ClipvitlModel2(n_class = 0)        
        model = ClipvitlModelReduction2(pretrained_backbone = model_backbone )
        checkpoint_model = torch.load(args.weight_path, map_location='cpu')['model']
        model.load_state_dict(checkpoint_model, strict=True)
        device = torch.device("cuda")
        utils.freeze_weights(model,['retrieval_back_bone'])
        model.to(device)

        


    
    ################
    
    # Step 1: extract features
    query_features = extract_features(model, data_loader_query, args.use_cuda, multiscale=args.multiscale)
    train_features = extract_features(model, data_loader_train, args.use_cuda, multiscale=args.multiscale)

    # normalize features
    train_features_norm = nn.functional.normalize(train_features, dim=1, p=2)
    query_features_norm = nn.functional.normalize(query_features, dim=1, p=2)

    ############################################################################
    # Step 2: global search
    sim = torch.mm(train_features_norm, query_features_norm.T)
    ranks = torch.argsort(-sim, dim=0).cpu().numpy()
        
    print('Global search mAP')
    reportMap(dataset_train,args, ranks)
    
                


# python eval_image_retrieval.py --weight_path  model_checkpoint/vitl_cip_adapsubarc_tune23highd_d3_reduct_ckpt0_nonlineardecode2_normalize/checkpoint_model_epoch_0.pth --imsize 336 --r1m_path None --from_timm False



