import os, sys
import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
import cv2
import imageio
from imgviz import label_colormap

class ReplicaDatasetCache(Dataset):
    def __init__(self, data_dir, train_ids, test_ids, img_h=None, img_w=None):

        traj_file = os.path.join(data_dir, "traj_w_c.txt")
        self.rgb_dir = os.path.join(data_dir, "rgb")
        self.depth_dir = os.path.join(data_dir, "depth")  # depth is in mm uint
        self.semantic_class_dir = os.path.join(data_dir, "semantic_class")
        self.semantic_instance_dir = os.path.join(data_dir, "semantic_instance")
        if not os.path.exists(self.semantic_instance_dir):
            self.semantic_instance_dir = None


        self.train_ids = train_ids
        self.train_num = len(train_ids)
        self.test_ids = test_ids
        self.test_num = len(test_ids)

        self.img_h = img_h
        self.img_w = img_w

        self.Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.depth_list = sorted(glob.glob(self.depth_dir + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.semantic_list = sorted(glob.glob(self.semantic_class_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        if self.semantic_instance_dir is not None:
            self.instance_list = sorted(glob.glob(self.semantic_instance_dir + '/semantic_instance_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        self.train_samples = {'image': [], 'depth': [],
                          'semantic': [], 'T_wc': [],
                          'instance': []}

        self.test_samples = {'image': [], 'depth': [],
                          'semantic': [], 'T_wc': [],
                          'instance': []}

       # training samples
        for idx in train_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
            semantic = cv2.imread(self.semantic_list[idx], cv2.IMREAD_UNCHANGED)
            if self.semantic_instance_dir is not None:
                instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_UNCHANGED) # uint16

            if (self.img_h is not None and self.img_h != image.shape[0]) or \
                    (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                if self.semantic_instance_dir is not None:
                    instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

            T_wc = self.Ts_full[idx]

            self.train_samples["image"].append(image)
            self.train_samples["depth"].append(depth)
            self.train_samples["semantic"].append(semantic)
            if self.semantic_instance_dir is not None:
                self.train_samples["instance"].append(instance)
            self.train_samples["T_wc"].append(T_wc)


        # test samples
        for idx in test_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
            semantic = cv2.imread(self.semantic_list[idx], cv2.IMREAD_UNCHANGED)
            if self.semantic_instance_dir is not None:
                instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_UNCHANGED) # uint16

            if (self.img_h is not None and self.img_h != image.shape[0]) or \
                    (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                if self.semantic_instance_dir is not None:
                    instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
            T_wc = self.Ts_full[idx]

            self.test_samples["image"].append(image)
            self.test_samples["depth"].append(depth)
            self.test_samples["semantic"].append(semantic)
            if self.semantic_instance_dir is not None:
                self.test_samples["instance"].append(instance)
            self.test_samples["T_wc"].append(T_wc)

        for key in self.test_samples.keys():  # transform list of np array to array with batch dimension
            self.train_samples[key] = np.asarray(self.train_samples[key])
            self.test_samples[key] = np.asarray(self.test_samples[key])

        self.semantic_classes = np.unique(
            np.concatenate(
                (np.unique(self.train_samples["semantic"]), 
            np.unique(self.test_samples["semantic"])))).astype(np.uint8)
        self.num_semantic_class = self.semantic_classes.shape[0]  # number of semantic classes, including the void class of 0

        self.colour_map_np = label_colormap()[self.semantic_classes]
        self.mask_ids = np.ones(self.train_num)  # init self.mask_ids as full ones
        # 1 means the correspinding label map is used for semantic loss during training, while 0 means no semantic loss is applied on this frame

        # remap existing semantic class labels to continuous label ranging from 0 to num_class-1
        self.train_samples["semantic_clean"] = self.train_samples["semantic"].copy()
        self.train_samples["semantic_remap"] = self.train_samples["semantic"].copy()
        self.train_samples["semantic_remap_clean"] = self.train_samples["semantic_clean"].copy()

        self.test_samples["semantic_remap"] = self.test_samples["semantic"].copy()

        for i in range(self.num_semantic_class):
            self.train_samples["semantic_remap"][self.train_samples["semantic"]== self.semantic_classes[i]] = i
            self.train_samples["semantic_remap_clean"][self.train_samples["semantic_clean"]== self.semantic_classes[i]] = i
            self.test_samples["semantic_remap"][self.test_samples["semantic"]== self.semantic_classes[i]] = i


        print()
        print("Training Sample Summary:")
        for key in self.train_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.train_samples[key].shape, self.train_samples[key].dtype))
        print()
        print("Testing Sample Summary:")
        for key in self.test_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.test_samples[key].shape, self.test_samples[key].dtype))


    def sample_label_maps(self, sparse_ratio=0.5, K=0, random_sample=False, load_saved=False):
        """
        sparse_ratio means the ratio of removed training images, e.g., 0.3 means 30% of semantic labels are removed
        Input:
            sparse_ratio: the percentage of semantic label frames to be *removed*
            K: the number of frames to be removed, if this is speficied we override the results computed from sparse_ratio
            random_sample: whether to random sample frames or interleavely/evenly sample, True--random sample; False--interleavely sample
            load_saved: use pre-computed mask_ids from previous experiments
        """
        if load_saved is False:
            if K==0:
                K = int(self.train_num*sparse_ratio)  # number of skipped training frames, mask=0

            N = self.train_num-K  # number of used training frames,  mask=1
            assert np.sum(self.mask_ids) == self.train_num  # sanity check that all masks are avaible before sampling

            if K==0: # in case sparse_ratio==0:
                return 
        
            if random_sample:
                self.mask_ids[:K] = 0
                np.random.shuffle(self.mask_ids)
            else:  # sample interleave
                if sparse_ratio<=0.5: # skip less/equal than half frames
                    assert K <= self.train_num/2
                    q, r = divmod(self.train_num, K)
                    indices = [q*i + min(i, r) for i in range(K)]
                    self.mask_ids[indices] = 0

                else: # skip more than half frames
                    assert K > self.train_num/2
                    self.mask_ids = np.zeros_like(self.mask_ids)  # disable all images and  evenly enable N images in total
                    q, r = divmod(self.train_num, N)
                    indices = [q*i + min(i, r) for i in range(N)]
                    self.mask_ids[indices] = 1 

            print("{} of {} semantic labels are sampled (sparse ratio: {}).".format(sum(self.mask_ids), len(self.mask_ids), sparse_ratio))
            noisy_sem_dir = os.path.join(self.semantic_class_dir, "noisy_pixel_sems_sr{}".format(sparse_ratio))
            if not os.path.exists(noisy_sem_dir):
                os.makedirs(noisy_sem_dir)
            with open(os.path.join(noisy_sem_dir, "mask_ids.npy"), 'wb') as f:
                np.save(f, self.mask_ids)
        elif load_saved is True:
            noisy_sem_dir = os.path.join(self.semantic_class_dir, "noisy_pixel_sems_sr{}".format(sparse_ratio))
            self.mask_ids = np.load(os.path.join(noisy_sem_dir, "mask_ids.npy"))


    
    def sample_specific_labels(self, frame_ids, train_ids):
        """
        Only use dense label maps for specific/selected frames.
        """
        assert np.sum(self.mask_ids) == self.train_num  # sanity check that all masks are avaible before sampling

        self.mask_ids = np.zeros_like(self.mask_ids)

        if len(frame_ids)==1 and frame_ids[0] is None:
            # we do not add any semantic supervision 
            return

        relative_ids = [train_ids.index(x) for x in frame_ids]

        self.mask_ids[relative_ids] = 1


    def add_pixel_wise_noise_label(self, 
        sparse_views=False, sparse_ratio=0.0, random_sample=False, 
        noise_ratio=0.0, visualise_save=False, load_saved=False):
        """
        sparse_views: whether we sample a subset of dense semantic labels for training
        sparse_ratio: the ratio of frames to be removed/skipped if sampling a subset of labels
        random_sample: whether to random sample frames or interleavely/evenly sample, True--random sample; False--interleavely sample
        noise_ratio: the ratio of num pixels per-frame to be randomly perturbed
        visualise_save: whether to save the noisy labels into harddrive for later usage
        load_saved: use trained noisy labels for training to ensure consistency betwwen experiments
        """

        if not load_saved:
            if sparse_views:
                self.sample_label_maps(sparse_ratio=sparse_ratio, random_sample=random_sample)
            num_pixel = self.img_h * self.img_w
            num_pixel_noisy = int(num_pixel*noise_ratio)
            train_sem = self.train_samples["semantic_remap"]

            for i in range(len(self.mask_ids)):
                if self.mask_ids[i] == 1:  # add label noise to unmasked/available labels
                    noisy_index_1d = np.random.permutation(num_pixel)[:num_pixel_noisy]
                    faltten_sem = train_sem[i].flatten()

                    faltten_sem[noisy_index_1d] = np.random.choice(self.num_semantic_class, num_pixel_noisy)
                    # we replace the label of randomly selected num_pixel_noisy pixels to random labels from [1, self.num_semantic_class], 0 class is the none class
                    train_sem[i] = faltten_sem.reshape(self.img_h, self.img_w)

            print("{} of {} semantic labels are added noise {} percent area ratio.".format(sum(self.mask_ids), len(self.mask_ids), noise_ratio))

            if visualise_save:
                noisy_sem_dir = os.path.join(self.semantic_class_dir, "noisy_pixel_sems_sr{}_nr{}".format(sparse_ratio, noise_ratio))
                if not os.path.exists(noisy_sem_dir):
                    os.makedirs(noisy_sem_dir)
                with open(os.path.join(noisy_sem_dir, "mask_ids.npy"), 'wb') as f:
                    np.save(f, self.mask_ids)

                vis_noisy_semantic_list = []
                vis_semantic_clean_list = []

                colour_map_np = self.colour_map_np
                # 101 classes in total from Replica, select the existing class from total colour map

                semantic_remap = self.train_samples["semantic_remap"] # [H, W, 3]
                semantic_remap_clean = self.train_samples["semantic_remap_clean"] # [H, W, 3]

                # save semantic labels
                for i in range(len(self.mask_ids)):
                    if self.mask_ids[i] == 1: 
                        vis_noisy_semantic = colour_map_np[semantic_remap[i]] # [H, W, 3]
                        vis_semantic_clean = colour_map_np[semantic_remap_clean[i]] # [H, W, 3]

                        imageio.imwrite(os.path.join(noisy_sem_dir, "semantic_class_{}.png".format(i)), semantic_remap[i])
                        imageio.imwrite(os.path.join(noisy_sem_dir, "vis_sem_class_{}.png".format(i)), vis_noisy_semantic)

                        vis_noisy_semantic_list.append(vis_noisy_semantic)
                        vis_semantic_clean_list.append(vis_semantic_clean)
                    else:
                        # for mask_ids of 0, we skip these frames during training and do not add noise
                        vis_noisy_semantic = colour_map_np[semantic_remap[i]] # [H, W, 3]
                        vis_semantic_clean = colour_map_np[semantic_remap_clean[i]] # [H, W, 3]
                        assert np.all(vis_noisy_semantic==vis_semantic_clean) # apply this check to skipped frames

                        imageio.imwrite(os.path.join(noisy_sem_dir, "semantic_class_{}.png".format(i)), semantic_remap[i])
                        imageio.imwrite(os.path.join(noisy_sem_dir, "vis_sem_class_{}.png".format(i)), vis_noisy_semantic)

                        vis_noisy_semantic_list.append(vis_noisy_semantic)
                        vis_semantic_clean_list.append(vis_semantic_clean)

                imageio.mimwrite(os.path.join(noisy_sem_dir, 'noisy_sem_ratio_{}.mp4'.format(noise_ratio)), 
                        np.stack(vis_noisy_semantic_list, 0), fps=30, quality=8)
                
                imageio.mimwrite(os.path.join(noisy_sem_dir, 'clean_sem.mp4'), 
                        np.stack(vis_semantic_clean_list, 0), fps=30, quality=8)
        else:
            print("Load saved noisy labels.")
            noisy_sem_dir = os.path.join(self.semantic_class_dir, "noisy_pixel_sems_sr{}_nr{}".format(sparse_ratio, noise_ratio))
            assert os.path.exists(noisy_sem_dir)
            self.mask_ids = np.load(os.path.join(noisy_sem_dir, "mask_ids.npy"))
            semantic_img_list = []
            semantic_path_list = sorted(glob.glob(noisy_sem_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
            assert len(semantic_path_list)>0
            for idx in range(len(self.mask_ids)):
                semantic = imread(semantic_path_list[idx])
                semantic_img_list.append(semantic)
            self.train_samples["semantic_remap"]  = np.asarray(semantic_img_list)


    def add_instance_wise_noise_label(self, sparse_views=False, sparse_ratio=0.0, random_sample=False,
            flip_ratio=0.0, uniform_flip=False, 
            instance_id=[3, 6, 7, 9, 11, 12, 13, 48],  
            load_saved=False,
            visualise_save=False):

        """ In this function, we try to test if semantic-NERF can correct the wrong instance label after fusion (training).
        For selected instances, we randomly pick a portion of frames and change their class labels.
        Input:
            sparse_views: if we use a subset of sampled original training set or not.
            sparse_ratio: the ratio of frames to be dropped.
            random_sample: whether to random sample frames or interleavely/evenly sample, True--random sample; False--interleavely sample
            flip_ratio: for all the frames containing certain instances, the ratio of changing labels
            uniform_flip: True: after sorting the candidate frames by instance area ratio,
                                  we uniform sample frames to flip certain instances' semantic class; 
                          False: we take the frames with least instance area ratio to change color.
            instance_id: the instance id of all 8 chairs in Replica Room_2, used for adding region-wise noise
            load_saved: whether to load the saved self.mask_ids or not
            visualise_save: If true, save processed partial labels into local harddrive/folders for futher usage.


        """
        num_pixel = self.img_w * self.img_h

        if not load_saved:
            if sparse_views:
                self.sample_label_maps(sparse_ratio=sparse_ratio, random_sample=random_sample, load_saved=load_saved)
            assert self.semantic_instance_dir is not None
            # instance_id = [3, 6, 7, 9, 11,12, 13, 48]
            # instance_maps_dict = dict.fromkeys(instance_id, [])  # using this one will make all the keys share the same value due to list [] is mutable
            instance_maps_dict = dict.fromkeys(instance_id)
            for k in instance_maps_dict.keys():
                instance_maps_dict[k] = list()


            # find which training images contrain the instance we want to flip labels
            for img_idx in range(self.train_num):
                instance_label_map = self.train_samples["instance"][img_idx]
                for ins_idx in instance_id:
                    instance_ratio = np.sum(instance_label_map==ins_idx)/num_pixel
                    if instance_ratio > 0 and self.mask_ids[img_idx]==1: # larger than 1% image area and the image is also sampled into training set
                        instance_maps_dict[ins_idx].append([img_idx, instance_ratio])

            num_frame_per_instance_id = np.asarray([len(x) for x in  instance_maps_dict.values()])
            num_flip_frame_per_instance_id =  (num_frame_per_instance_id*flip_ratio).astype(np.int)

            for k, v in instance_maps_dict.items():
                instance_maps_dict[k] = sorted(instance_maps_dict[k], key=lambda x: x[1])  # sorted, default is ascending order
            if not uniform_flip: 
            # we flip the labels with minimum area ratio, 
            # the intuition is that the observation is partial and is likely to be wrong.
                for i in range(len(instance_id)):  # loop over instance id
                    selected_frame_id = [x[0] for x in instance_maps_dict[instance_id[i]][:num_flip_frame_per_instance_id[i]]]
                    for m in selected_frame_id: # loop over image ids having the selected instance
                        self.train_samples["semantic_remap"][m][self.train_samples["instance"][m]==instance_id[i]] = np.random.choice(self.num_semantic_class, 1)
            else:
                if flip_ratio<=0.5: # flip less/equal than half frames
                    for i in range(len(instance_id)):  # loop over instance id
                        K = num_flip_frame_per_instance_id[i]
                        q, r = divmod(num_frame_per_instance_id[i], K)
                        indices_to_flip = [q*i + min(i, r) for i in range(K)]
                        valid_frame_id_list = [x[0] for x in instance_maps_dict[instance_id[i]]]
                        selected_frame_id = [valid_frame_id_list[flip_id] for flip_id in indices_to_flip]
                        for m in selected_frame_id: # loop over image ids having the selected instance
                            self.train_samples["semantic_remap"][m][self.train_samples["instance"][m]==instance_id[i]] = np.random.choice(self.num_semantic_class, 1)
                
                else: # flip more than half frames
                    for i in range(len(instance_id)):  # loop over instance id
                        K = num_flip_frame_per_instance_id[i]
                        N = num_frame_per_instance_id[i] - K
                        q, r = divmod(num_frame_per_instance_id[i], N)
                        indices_NOT_flip = [q*i + min(i, r) for i in range(N)]
                        indices_to_flip = [x for x in range(num_frame_per_instance_id[i]) if x not in indices_NOT_flip]
                        valid_frame_id_list = [x[0] for x in instance_maps_dict[instance_id[i]]]
                        selected_frame_id = [valid_frame_id_list[flip_id] for flip_id in indices_to_flip]
                        for m in selected_frame_id: # loop over image ids having the selected instance
                            self.train_samples["semantic_remap"][m][self.train_samples["instance"][m]==instance_id[i]] = np.random.choice(self.num_semantic_class, 1)
        
            colour_map_np = self.colour_map_np
            vis_flip_semantic = [colour_map_np[sem] for sem in self.train_samples["semantic_remap"]]
            vis_gt_semantic = [colour_map_np[sem] for sem in self.train_samples["semantic_remap_clean"]]

            if visualise_save:
                flip_sem_dir  = os.path.join(self.semantic_class_dir, "flipped_chair_nr_{}".format(flip_ratio))
                if not os.path.exists(flip_sem_dir):
                    os.makedirs(flip_sem_dir)
                
                with open(os.path.join(flip_sem_dir, "mask_ids.npy"), 'wb') as f:
                    np.save(f, self.mask_ids)
                    
                for i in range(len(vis_flip_semantic)):
                    imageio.imwrite(os.path.join(flip_sem_dir, "semantic_class_{}.png".format(i)), self.train_samples["semantic_remap"][i])
                    imageio.imwrite(os.path.join(flip_sem_dir, "vis_sem_class_{}.png".format(i)), vis_flip_semantic[i])
                    imageio.imwrite(os.path.join(flip_sem_dir, "vis_gt_{}.png".format(i)), vis_gt_semantic[i])
        else:
            print("Load saved noisy labels.")
            flip_sem_dir  = os.path.join(self.semantic_class_dir, "flipped_chair_nr_{}".format(flip_ratio))
            assert os.path.exists(flip_sem_dir)
            self.mask_ids = np.load(os.path.join(flip_sem_dir, "mask_ids.npy"))
            semantic_img_list = []
            semantic_path_list = sorted(glob.glob(flip_sem_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
            assert len(semantic_path_list)>0
            for idx in range(len(self.mask_ids)):
                semantic = imread(semantic_path_list[idx])
                semantic_img_list.append(semantic)
            self.train_samples["semantic_remap"]  = np.asarray(semantic_img_list)

    def super_resolve_label(self, down_scale_factor=8, dense_supervision=True):
        """ In super-resolution mode, to create training supervisions, we downscale the ground truth label by certain scaling factor to 
        throw away information. We then upscale the image back to original size.

        Two setups for upscaling: 
            (1) Sparse label: we set the interpolated label pixels to void label==0, so we only have losses on grid of every 8 pixels
            (2) Dense label: we penalise also on interpolated pixel values

        down_scale_factor: the scaling factor for down-sampling and up-sampling
        dense_supervision: dense label mode or not.
        """
        if down_scale_factor==1:
            return 
        if dense_supervision:  # for dense labelling,  we down-scale and up-scale label maps again
            scaled_low_res_train_label = []
            for i in range(self.train_num):
                low_res_label = cv2.resize(self.train_samples["semantic_remap"][i], 
                (self.img_w//down_scale_factor, self.img_h//down_scale_factor),
                interpolation=cv2.INTER_NEAREST)

                scaled_low_res_label = cv2.resize(low_res_label, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                scaled_low_res_train_label.append(scaled_low_res_label)

            scaled_low_res_train_label = np.asarray(scaled_low_res_train_label)

            self.train_samples["semantic_remap"] = scaled_low_res_train_label

        else: # for sparse labelling, we only penalise strictly on valid pixel positions
            valid_low_res_pixel_mask = np.zeros((self.img_h, self.img_w))
            valid_low_res_pixel_mask[::down_scale_factor, ::down_scale_factor]=1
            self.train_samples["semantic_remap"] = (self.train_samples["semantic_remap"]*valid_low_res_pixel_mask[None,...]).astype(np.uint8)
            # we mask all the decimated pixel label to void class==0

    def simulate_user_click_partial(self, perc=0, load_saved=False, visualise_save=True):
        """
        Generate partial label maps for label propagation task.
        perc: the percentage of pixels per class per image to be preserved to simulate partial user clicks
            0: single-clicks
            1: 1% user clicks
            5: 5% user clicks

        load_saved: If true, load saved partial clicks to guarantee reproductability. False, create new partial laels
        visualise_save: If true, save processed partial labels into local harddrive/folders for futher usage like visualisation.
        """
        assert perc<=100 and perc >= 0
        assert self.train_num == self.train_samples["semantic_remap"].shape[0]
        single_click=True if perc==0 else False # single_click: whether to use single click only from each class 
        perc = perc/100.0 # make perc value into percentage
        if not load_saved:

            if single_click:
                click_semantic_map = []
                for i in range(self.train_num):
                    if (i+1)%10==0:
                        print("Generating partial label of ratio {} for frame {}/{}.".format(perc, i, self.train_num))
                    im  = self.train_samples["semantic_remap"][i]
                    void_class = [0]
                    label_class = np.unique(im).tolist()
                    valid_class = [i for i in label_class if i not in void_class]
                    im_ = np.zeros_like(im)
                    for l in valid_class:
                        label_idx = np.transpose(np.nonzero(im == l))
                        sample_ind = np.random.choice(label_idx.shape[0], 1, replace=False)
                        label_idx_ = label_idx[sample_ind]
                        im_[label_idx_[:, 0], label_idx_[:, 1]] = l
                    click_semantic_map.append(im_)
                click_semantic_map = np.asarray(click_semantic_map).astype(np.uint8)
                self.train_samples["semantic_remap"] = click_semantic_map
            
                print('Partial Label images with centroid sampling (extreme) has completed.')

            elif perc>0 and not single_click:
                click_semantic_map = []
                for i in range(self.train_num):
                    if (i+1)%10==0:
                        print("Generating partial label of ratio {} for frame {}/{}.".format(perc, i, self.train_num))
                    im  = self.train_samples["semantic_remap"][i]
                    void_class = [0]
                    label_class = np.unique(im).tolist() # find the unique class-ids in the current training label
                    valid_class = [c for c in label_class if c not in void_class]

                    im_ = np.zeros_like(im)
                    for l in valid_class:
                        label_mask = np.zeros_like(im)
                        label_mask_ = im == l # binary mask of pixels equal to class-l 
                        label_idx = np.transpose(np.nonzero(label_mask_)) # Nx2
                        sample_ind = np.random.choice(label_idx.shape[0], 1, replace=False) # shape [1,]
                        label_idx_ = label_idx[sample_ind] # shape [1, 2]
                        target_num = int(perc * label_mask_.sum()) # find the target and total number of pixels belong to class-l in the current image
                        label_mask[label_idx_[0, 0], label_idx_[0, 1]] = 1 # full-zero mask with only selected pixel to be 1
                        label_mask_true = label_mask
                        # label_mask_true initially has only 1 True pixel, we continuously grow mask until reach expected percentage

                        while label_mask_true.sum() < target_num:
                            num_before_grow = label_mask_true.sum()
                            label_mask = cv2.dilate(label_mask, kernel=np.ones([5, 5]))
                            label_mask_true = label_mask * label_mask_
                            num_after_grow = label_mask_true.sum()
                            if num_after_grow==num_before_grow: 
                                print("Initialise Another Seed for Growing!")
                                # The current region stop growing which means the very local area has been filled,
                                #  so we need to initiate another seed to keep it growing
                                uncovered_region_mask = label_mask_ - label_mask_true # pixels which are equal to 1 are un-sampled regions and belong to current class
                                label_idx = np.transpose(np.nonzero(uncovered_region_mask)) # Nx2
                                sample_ind = np.random.choice(label_idx.shape[0], 1, replace=False) # shape [1,]
                                label_idx_ = label_idx[sample_ind] # shape [1, 2]
                                label_mask[label_idx_[0, 0], label_idx_[0, 1]] = 1 

                        im_[label_mask_true.astype(bool)] = l
                    click_semantic_map.append(im_)

                click_semantic_map = np.asarray(click_semantic_map).astype(np.uint8)
                self.train_samples["semantic_remap"] = click_semantic_map
                print('Partial Label images with centroid sampling has completed.')
            else:
                assert False

            if visualise_save:
                partial_sem_dir = os.path.join(self.semantic_class_dir, "partial_perc_{}".format(perc))
                if not os.path.exists(partial_sem_dir):
                    os.makedirs(partial_sem_dir)
                colour_map_np = self.colour_map_np
                vis_partial_sem = []
                for i in range(self.train_num):
                    vis_partial_semantic = colour_map_np[self.train_samples["semantic_remap"][i]] # [H, W, 3]
                    imageio.imwrite(os.path.join(partial_sem_dir, "semantic_class_{}.png".format(i)), self.train_samples["semantic_remap"][i])
                    imageio.imwrite(os.path.join(partial_sem_dir, "vis_sem_class_{}.png".format(i)), vis_partial_semantic)
                    vis_partial_sem.append(vis_partial_semantic)
            
                imageio.mimwrite(os.path.join(partial_sem_dir, 'partial_sem.mp4'), self.train_samples["semantic_remap"], fps=30, quality=8)
                imageio.mimwrite(os.path.join(partial_sem_dir, 'vis_partial_sem.mp4'), np.stack(vis_partial_sem, 0), fps=30, quality=8)
        
        else: # load saved single-click/partial semantics
            saved_partial_sem_dir = os.path.join(self.semantic_class_dir, "partial_perc_{}".format(perc))
            semantic_img_list = []
            semantic_path_list = sorted(glob.glob(saved_partial_sem_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
            assert len(semantic_path_list)>0
            for idx in range(self.train_num):
                semantic = imread(semantic_path_list[idx])
                semantic_img_list.append(semantic)
            self.train_samples["semantic_remap"]  = np.asarray(semantic_img_list).astype(np.uint8)