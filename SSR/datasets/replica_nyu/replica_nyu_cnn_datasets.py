
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import cv2

class Replica_CNN_NYU(Dataset):
    def __init__(self, data_dir, train_ids, test_ids, nyu_mode, img_h=None, img_w=None,  load_softmax=False):
        
        assert nyu_mode == "nyu13" or nyu_mode == "nyu34" or  nyu_mode == "gt_nyu13"

        traj_file = os.path.join(data_dir, "traj_w_c.txt")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")  # depth is in mm uint
        # self.cnn_semantic_class_dir = os.path.join(data_dir, "CNN_semantic_class_{}".format(nyu_mode))
        if nyu_mode == "nyu13":
            self.cnn_semantic_class_dir = os.path.join(data_dir, "CNN_semantic_class_nyu13")
            self.gt_semantic_class_dir = os.path.join(data_dir, "semantic_class_nyu13_remap")
        elif nyu_mode=="nyu34":
            self.cnn_semantic_class_dir = os.path.join(data_dir, "CNN_semantic_class_nyu34")
            self.gt_semantic_class_dir = os.path.join(data_dir, "semantic_class_nyu40_remap_nyu34")
        elif nyu_mode == "gt_nyu13":
            self.cnn_semantic_class_dir = os.path.join(data_dir, "semantic_class_nyu13_remap")
            self.gt_semantic_class_dir = os.path.join(data_dir, "semantic_class_nyu13_remap")
        
        # self.cnn_softmax_dir = os.path.join(data_dir, "semantic_prob_CNN")
        

        self.nyu_mode = nyu_mode
        self.load_softmax = load_softmax
        
        self.train_ids = train_ids
        self.train_num = len(train_ids)
        self.test_ids = test_ids
        self.test_num = len(test_ids)

        self.img_h = img_h
        self.img_w = img_w

        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.depth_list = sorted(glob.glob(self.depth_dir + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.cnn_semantic_list = sorted(glob.glob(self.cnn_semantic_class_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.gt_semantic_list = sorted(glob.glob(self.gt_semantic_class_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        if load_softmax:
            self.cnn_softmax_list = sorted(glob.glob(self.cnn_softmax_dir + '/softmax_prob_*.npy'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))



        self.train_samples = {'image': [], 'depth': [],
                          'cnn_semantic': [], 
                          'gt_semantic': [],
                          'cnn_softmax': [],
                          'cnn_entropy':[],
                          'T_wc': []}

        self.test_samples = {'image': [], 'depth': [],
                          'cnn_semantic': [], 
                          'gt_semantic': [],
                          'cnn_softmax': [],
                          'cnn_entropy':[],
                          'T_wc': []}
       # training samples
        for idx in train_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
            cnn_semantic = cv2.imread(self.cnn_semantic_list[idx], cv2.IMREAD_UNCHANGED)
            gt_semantic = cv2.imread(self.gt_semantic_list[idx], cv2.IMREAD_UNCHANGED)


            if (self.img_h is not None and self.img_h != image.shape[0]) or \
                    (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                cnn_semantic = cv2.resize(cnn_semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                gt_semantic = cv2.resize(gt_semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
            T_wc = self.Ts_full[idx]

            self.train_samples["image"].append(image)
            self.train_samples["depth"].append(depth)
            self.train_samples["cnn_semantic"].append(cnn_semantic)
            self.train_samples["gt_semantic"].append(gt_semantic)
            self.train_samples["T_wc"].append(T_wc)


        # test samples
        for idx in test_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
            cnn_semantic = cv2.imread(self.cnn_semantic_list[idx], cv2.IMREAD_UNCHANGED)
            gt_semantic = cv2.imread(self.gt_semantic_list[idx], cv2.IMREAD_UNCHANGED)


            if (self.img_h is not None and self.img_h != image.shape[0]) or \
                    (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                cnn_semantic = cv2.resize(cnn_semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                gt_semantic = cv2.resize(gt_semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
            T_wc = self.Ts_full[idx]

            self.test_samples["image"].append(image)
            self.test_samples["depth"].append(depth)
            self.test_samples["cnn_semantic"].append(cnn_semantic)
            self.test_samples["gt_semantic"].append(gt_semantic)
            self.test_samples["T_wc"].append(T_wc)



        if load_softmax is True:
            softmax_2_entropy_np = lambda x, axis: np.sum(-np.log2(x+1e-12)*x, axis=axis, keepdims=False)  # H,W
            # training samples
            cnt = 0
            for idx in train_ids:
                cnn_softmax = np.clip(np.load(self.cnn_softmax_list[idx]), a_min=0, a_max=1.0)
                if (self.img_h is not None and self.img_h != cnn_softmax.shape[0]) or \
                        (self.img_w is not None and self.img_w != cnn_softmax.shape[1]):
                    cnn_softmax = cv2.resize(cnn_softmax, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                    # opencv resize support resize 512 channel at maximum
        
                valid_mask = self.train_samples["gt_semantic"][cnt]>0
                entropy = softmax_2_entropy_np(cnn_softmax, -1)*valid_mask
                cnn_softmax = cnn_softmax*valid_mask[:,:,None]
                self.train_samples["cnn_softmax"].append(cnn_softmax)
                self.train_samples["cnn_entropy"].append(entropy)
                cnt += 1
            assert cnt==len(train_ids)

            # test samples
            cnt = 0
            for idx in test_ids:
                cnn_softmax = np.load(self.cnn_softmax_list[idx])
                assert cnn_softmax.shape[-1]==34
                if (self.img_h is not None and self.img_h != cnn_softmax.shape[0]) or \
                        (self.img_w is not None and self.img_w != cnn_softmax.shape[1]):
                    cnn_softmax = cv2.resize(cnn_softmax, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                # we do not need softmax for testing, can also save memory
                valid_mask = self.test_samples["gt_semantic"][cnt]>0
                entropy = softmax_2_entropy_np(cnn_softmax, -1)*valid_mask
                self.test_samples["cnn_entropy"].append(entropy)
                cnt += 1
            assert cnt==len(test_ids)


        for key in self.test_samples.keys():  # transform list of np array to array with batch dimension
            self.train_samples[key] = np.asarray(self.train_samples[key])
            self.test_samples[key] = np.asarray(self.test_samples[key])

        if nyu_mode == "nyu13" or nyu_mode == "gt_nyu13":
            self.semantic_classes = np.arange(14) # 0-void, 1-13 valid classes
            self.num_semantic_class = 14 # 13 valid class + 1 void class
            from SSR.utils import image_utils
            self.colour_map_np = image_utils.nyu13_colour_code
        elif nyu_mode=="nyu34":
            self.semantic_classes = np.arange(35)  # 0-void, 1-34 valid classes
            self.num_semantic_class = 35 # 34 valid class + 1 void class
            self.colour_map_np = image_utils.nyu34_colour_code

        self.mask_ids = np.ones(self.train_num)  # init self.mask_ids as full ones
        # 1 means the correspinding label map is used for semantic loss during training, while 0 means no semantic loss
        self.train_samples["cnn_semantic_clean"] = self.train_samples["cnn_semantic"].copy()
        
        print()
        print("Training Sample Summary:")
        for key in self.train_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.train_samples[key].shape, self.train_samples[key].dtype))
        print()
        print("Testing Sample Summary:")
        for key in self.test_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.test_samples[key].shape, self.test_samples[key].dtype))