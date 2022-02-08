import os
import glob
import numpy as np
from skimage.io import imread
import cv2
import imageio

from SSR.datasets.scannet.scannet_utils import load_scannet_nyu40_mapping, load_scannet_nyu13_mapping
from SSR.utils import image_utils
class ScanNet_Dataset(object):
    def __init__(self, scene_dir, img_h=None, img_w=None, sample_step=1, save_dir=None, mode="nyu40"):
        # we only use rgb+poses from Scannet
        self.img_h = img_h
        self.img_w = img_w

        self.scene_dir = scene_dir # scene_dir is the root directory of each sequence, i.e., xxx/ScanNet/scans/scene0088_00"
        # scene_dir = "/home/shuaifeng/Documents/Datasets/ScanNet/scans/scene0088_00"
        scene_name = os.path.basename(scene_dir)
        data_dir = os.path.dirname(scene_dir)

        instance_filt_dir =  os.path.join(scene_dir, scene_name+'_2d-instance-filt')
        label_filt_dir =  os.path.join(scene_dir, scene_name+'_2d-label-filt')
        self.semantic_class_dir = label_filt_dir

        # (0 corresponds to unannotated or no depth).
        if mode=="nyu40":
            label_mapping_nyu = load_scannet_nyu40_mapping(scene_dir)
            colour_map_np = image_utils.nyu40_colour_code
            assert colour_map_np.shape[0] == 41
        elif mode=="nyu13":
            label_mapping_nyu = load_scannet_nyu13_mapping(scene_dir)
            colour_map_np = image_utils.nyu13_colour_code
            assert colour_map_np.shape[0] == 14
        else:
            assert False

        # get camera intrinsics
        # we use color camera intrinsics and resize depth to match
        with open(os.path.join(scene_dir, "{}.txt".format(scene_name))) as info_f:
            info = [line.rstrip().split(' = ') for line in info_f]
            info = {key:value for key, value in info}
            intrinsics = [
                [float(info['fx_color']), 0, float(info['mx_color'])],
                [0, float(info['fy_color']), float(info['my_color'])],
                [0, 0, 1]]

            original_colour_h = int(info["colorHeight"])
            original_colour_w = int(info["colorWidth"])
            original_depth_h = int(info["depthHeight"])
            original_depth_w = int(info["depthWidth"])
            assert original_colour_h==968 and original_colour_w==1296 and original_depth_h==480 and original_depth_w==640

        # load 2D colour frames and poses

        frame_ids = os.listdir(os.path.join(scene_dir, "renders", 'color'))
        frame_ids = [int(os.path.splitext(frame)[0]) for frame in frame_ids]
        frame_ids =  sorted(frame_ids)

        frames_file_list = []
        for i, frame_id in enumerate(frame_ids):
            if i%25==0:
                print('preparing %s frame %d/%d'%(scene_name, i, len(frame_ids)))

            pose = np.loadtxt(os.path.join(scene_dir, "renders", 'pose', '%d.txt' % frame_id))

            # skip frames with no valid pose
            if not np.all(np.isfinite(pose)):
                continue

            frame = {'file_name_image': 
                        os.path.join(scene_dir, "renders", 'color', '%d.jpg'%frame_id),
                    'file_name_depth': 
                        os.path.join(scene_dir, "renders", 'depth', '%d.png'%frame_id),
                    'file_name_instance': 
                        os.path.join(instance_filt_dir, 'instance-filt', '%d.png'%frame_id),
                    'file_name_label': 
                        os.path.join(label_filt_dir, 'label-filt', '%d.png'%frame_id),
                    'intrinsics': intrinsics,
                    'pose': pose,
                    }

            frames_file_list.append(frame)

        step = sample_step
        valid_data_num = len(frames_file_list)
        self.valid_data_num = valid_data_num
        total_ids = range(valid_data_num)
        train_ids = total_ids[::step]
        test_ids = [x+ (step//2) for x in train_ids]   
        if test_ids[-1]>valid_data_num-1:
            test_ids.pop(-1)
        self.train_ids = train_ids
        self.train_num = len(train_ids)
        self.test_ids = test_ids
        self.test_num = len(test_ids)

        self.train_samples = {'image': [], 'depth': [],
                              'semantic_raw': [],  # raw scannet label id
                              'semantic': [],   # nyu40 id
                              'T_wc': [],
                              'instance': []}

    
        self.test_samples = {'image': [], 'depth': [],
                              'semantic_raw': [], 
                              'semantic': [], 
                              'T_wc': [],
                              'instance': []}

        # training samples
        for idx in train_ids:
            image = cv2.imread(frames_file_list[idx]["file_name_image"])[:,:,::-1] # change from BGR uinit 8 to RGB float
            image = cv2.copyMakeBorder(src=image, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0,0,0]) # pad 4 pixels to height so that images have aspect ratio of 4:3
            assert image.shape[0]/image.shape[1]==3/4 and image.shape[1]==original_colour_w and image.shape[0] == 972
            image = image/255.0

            depth = cv2.imread(frames_file_list[idx]["file_name_depth"], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter

            semantic = cv2.imread(frames_file_list[idx]["file_name_label"], cv2.IMREAD_UNCHANGED)
            semantic = cv2.copyMakeBorder(src=semantic, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)

            instance = cv2.imread(frames_file_list[idx]["file_name_instance"], cv2.IMREAD_UNCHANGED)
            instance = cv2.copyMakeBorder(src=instance, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)

            T_wc = frames_file_list[idx]["pose"].reshape((4, 4))

            if (self.img_h is not None and self.img_h != image.shape[0]) or \
                    (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

            self.train_samples["image"].append(image)
            self.train_samples["depth"].append(depth)
            self.train_samples["semantic_raw"].append(semantic)
            self.train_samples["instance"].append(instance)
            self.train_samples["T_wc"].append(T_wc)


        # test samples
        for idx in test_ids:
            image = cv2.imread(frames_file_list[idx]["file_name_image"])[:,:,::-1] # change from BGR uinit 8 to RGB float
            image = cv2.copyMakeBorder(src=image, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0,0,0]) # pad 4 pixels to height so that images have aspect ratio of 4:3
            assert image.shape[0]/image.shape[1]==3/4 and image.shape[1]==original_colour_w and image.shape[0] == 972
            image = image/255.0

            depth = cv2.imread(frames_file_list[idx]["file_name_depth"], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
            
            semantic = cv2.imread(frames_file_list[idx]["file_name_label"], cv2.IMREAD_UNCHANGED)
            semantic = cv2.copyMakeBorder(src=semantic, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)

            instance = cv2.imread(frames_file_list[idx]["file_name_instance"], cv2.IMREAD_UNCHANGED)
            instance = cv2.copyMakeBorder(src=instance, top=2, bottom=2, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=0)

            T_wc = frames_file_list[idx]["pose"].reshape((4, 4))

            if (self.img_h is not None and self.img_h != image.shape[0]) or \
                    (self.img_w is not None and self.img_w != image.shape[1]):
                image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
                semantic = cv2.resize(semantic, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
            

            self.test_samples["image"].append(image)
            self.test_samples["depth"].append(depth)
            self.test_samples["semantic_raw"].append(semantic)
            self.test_samples["instance"].append(instance)
            self.test_samples["T_wc"].append(T_wc)


        scale_y = image.shape[0]/(original_colour_h+4)
        scale_x = image.shape[1]/original_colour_w
        assert scale_x == scale_y # this requires the desired shape to also has a aspect ratio of 4:3

        # we modify the camera intrinsics considering the padding and scaling
        self.intrinsics = np.asarray(intrinsics)
        self.intrinsics[1,2] += 2 # we add c_y by 2 since we pad the height by 4 pixels
        self.intrinsics[0, 0] = self.intrinsics[0, 0]*scale_x # fx
        self.intrinsics[1, 1] = self.intrinsics[1, 1]*scale_x # fy

        self.intrinsics[0, 2] = self.intrinsics[0, 2]*scale_x  # cx
        self.intrinsics[1, 2] = self.intrinsics[1, 2]*scale_x  # cy


        for key in self.test_samples.keys():  # transform list of np array to array with batch dimension
            self.train_samples[key] = np.asarray(self.train_samples[key])
            self.test_samples[key] = np.asarray(self.test_samples[key])

        # map scannet classes to nyu definition
        train_semantic = self.train_samples["semantic_raw"]
        test_semantic = self.test_samples["semantic_raw"]

        train_semantic_nyu = train_semantic.copy()
        test_semantic_nyu = test_semantic.copy()

        for scan_id, nyu_id in label_mapping_nyu.items():
            train_semantic_nyu[train_semantic==scan_id] = nyu_id
            test_semantic_nyu[test_semantic==scan_id] = nyu_id

        self.train_samples["semantic"] = train_semantic_nyu
        self.test_samples["semantic"] = test_semantic_nyu


        self.semantic_classes = np.unique(
            np.concatenate(
                (np.unique(self.train_samples["semantic"]), 
                np.unique(self.test_samples["semantic"])))
                ).astype(np.uint8)
        # each scene may not contain all 40-classes
        
        self.num_semantic_class = self.semantic_classes.shape[0]  # number of semantic classes

        colour_map_np_remap = colour_map_np.copy()[self.semantic_classes] # take corresponding colour map
        self.colour_map_np = colour_map_np
        self.colour_map_np_remap = colour_map_np_remap
        self.mask_ids = np.ones(self.train_num)  # init self.mask_ids as full ones
        # 1 means the correspinding label map is used for semantic loss during training, while 0 means no semantic loss
    
        # save colourised ground truth label to img folder
        if save_dir is not None:
            # save colourised ground truth label to img folder
            vis_label_save_dir = os.path.join(save_dir, "vis-sampled-label-filt")
            os.makedirs(vis_label_save_dir, exist_ok=True)
            vis_train_label = colour_map_np[self.train_samples["semantic"]]
            vis_test_label = colour_map_np[self.test_samples["semantic"]]
            for i in range(self.train_num):
                label = vis_train_label[i].astype(np.uint8)
                cv2.imwrite(os.path.join(vis_label_save_dir, "train_vis_sem_{}.png".format(i)),label[...,::-1])

            for i in range(self.test_num):
                label = vis_test_label[i].astype(np.uint8)
                cv2.imwrite(os.path.join(vis_label_save_dir, "test_vis_sem_{}.png".format(i)),label[...,::-1])


        # remap existing semantic class labels to continuous label ranging from 0 to num_class-1
        self.train_samples["semantic_clean"] = self.train_samples["semantic"].copy()
        self.train_samples["semantic_remap"] = self.train_samples["semantic"].copy()
        self.train_samples["semantic_remap_clean"] = self.train_samples["semantic_clean"].copy()

        self.test_samples["semantic_remap"] = self.test_samples["semantic"].copy()

        for i in range(self.num_semantic_class):
            self.train_samples["semantic_remap"][self.train_samples["semantic"]== self.semantic_classes[i]] = i
            self.train_samples["semantic_remap_clean"][self.train_samples["semantic_clean"]== self.semantic_classes[i]] = i
            self.test_samples["semantic_remap"][self.test_samples["semantic"]== self.semantic_classes[i]] = i


        self.train_samples["semantic_remap"] = self.train_samples["semantic_remap"].astype(np.uint8)
        self.train_samples["semantic_remap_clean"] = self.train_samples["semantic_remap_clean"].astype(np.uint8)
        self.test_samples["semantic_remap"] = self.test_samples["semantic_remap"].astype(np.uint8)

        print()
        print("Training Sample Summary:")
        for key in self.train_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.train_samples[key].shape, self.train_samples[key].dtype))
        print()
        print("Testing Sample Summary:")
        for key in self.test_samples.keys(): 
            print("{} has shape of {}, type {}.".format(key, self.test_samples[key].shape, self.test_samples[key].dtype))


    def sample_label_maps(self, sparse_ratio=0.5, random_sample=False, load_saved=False):
        if load_saved is False:
            K = int(self.train_num*sparse_ratio)  # number of skipped training frames, mask=0
            N = self.train_num-K  # number of used training frames,  mask=1
            assert np.sum(self.mask_ids) == self.train_num  # sanity check that all masks are avaible before sampling

            if K==0: # incase sparse_ratio==0:
                return 
        
            if random_sample:
                self.mask_ids[:K] = 0
                np.random.shuffle(self.mask_ids)
            else:  # sample evenly
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
            noisy_sem_dir = os.path.join(self.scene_dir, "renders", "noisy_pixel_sems_sr{}".format(sparse_ratio))
            if not os.path.exists(noisy_sem_dir):
                os.makedirs(noisy_sem_dir)
            with open(os.path.join(noisy_sem_dir, "mask_ids.npy"), 'wb') as f:
                np.save(f, self.mask_ids)
        elif load_saved is True:
            noisy_sem_dir = os.path.join(self.scene_dir, "renders", "noisy_pixel_sems_sr{}".format(sparse_ratio))
            self.mask_ids = np.load(os.path.join(noisy_sem_dir, "mask_ids.npy"))


    def add_pixel_wise_noise_label(self, 
        sparse_views=False, sparse_ratio=0.5, random_sample=False, 
        noise_ratio=0.3, visualise_save=False, load_saved=False):
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
                noisy_sem_dir = os.path.join(self.scene_dir, "renders", "noisy_pixel_sems_sr{}_nr{}".format(sparse_ratio, noise_ratio))
                if not os.path.exists(noisy_sem_dir):
                    os.makedirs(noisy_sem_dir)
                with open(os.path.join(noisy_sem_dir, "mask_ids.npy"), 'wb') as f:
                    np.save(f, self.mask_ids)


                vis_noisy_semantic_list = []
                vis_semantic_clean_list = []

                colour_map_np = self.colour_map_np_remap 

                semantic_remap = self.train_samples["semantic_remap"] # [H, W, 3]
                semantic_remap_clean = self.train_samples["semantic_remap_clean"] # [H, W, 3]

                for i in range(len(self.mask_ids)):
                    if self.mask_ids[i] == 1:  # add label noise to unmasked/available labels
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
                        assert np.all(vis_noisy_semantic==vis_semantic_clean)

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
            noisy_sem_dir = os.path.join(self.scene_dir, "renders", "noisy_pixel_sems_sr{}_nr{}".format(sparse_ratio, noise_ratio))
            self.mask_ids = np.load(os.path.join(noisy_sem_dir, "mask_ids.npy"))
            semantic_img_list = []
            semantic_path_list = sorted(glob.glob(noisy_sem_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
            assert len(semantic_path_list)>0
            for idx in range(len(self.mask_ids)):
                semantic = imread(semantic_path_list[idx])
                semantic_img_list.append(semantic)
            self.train_samples["semantic_remap"]  = np.asarray(semantic_img_list)


    def super_resolve_label(self, down_scale_factor=8, dense_supervision=True):
        if down_scale_factor==1:
            return 
        if dense_supervision:  # train down-scale and up-scale again
            scaled_low_res_train_label = []
            for i in range(self.train_num):
                low_res_label = cv2.resize(self.train_samples["semantic_remap"][i], 
                (self.img_w//down_scale_factor, self.img_h//down_scale_factor),
                interpolation=cv2.INTER_NEAREST)

                scaled_low_res_label = cv2.resize(low_res_label, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                scaled_low_res_train_label.append(scaled_low_res_label)

            scaled_low_res_train_label = np.asarray(scaled_low_res_train_label)

            self.train_samples["semantic_remap"] = scaled_low_res_train_label

        else: # we only penalise strictly on valid pixel positions
            valid_low_res_pixel_mask = np.zeros((self.img_h, self.img_w))
            valid_low_res_pixel_mask[::down_scale_factor, ::down_scale_factor]=1
            self.train_samples["semantic_remap"] = (self.train_samples["semantic_remap"]*valid_low_res_pixel_mask[None,...]).astype(np.uint8)
            # we mask all the decimated pixel label to void class==0



    def simulate_user_click_partial(self, perc=0, load_saved=False, visualise_save=True):
        assert perc<=100 and perc >= 0
        assert self.train_num == self.train_samples["semantic_remap"].shape[0]
        single_click=True if perc==0 else False # single_click: whether to use single click only from each class 
        perc = perc/100.0 # make perc
        if not load_saved:

            if single_click:
                click_semantic_map = []
                for i in range(self.train_num):
                    if (i+1)%5==10:
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
                    if (i+1)%5==10:
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
                            # print("Before growth: {}, After growth: {}".format(num_before_grow, num_after_grow))
                            if num_after_grow==num_before_grow: 
                                print("Initialise Another Seed for Growing!")
                                # the region does not grow means the very local has been filled,
                                #  and we need to initiate another seed to keep growing
                                uncovered_region_mask = label_mask_ - label_mask_true # pixel equal to 1 means un-sampled regions belong to current class
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
                colour_map_np = self.colour_map_np_remap
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
