import os
import cv2
import numpy as np
import imgviz
from imgviz import label_colormap
from imgviz import draw as draw_module
import matplotlib.pyplot as plt

def numpy2cv(image):
    """

    :param image: a floating numpy images of shape [H,W,3] within range [0, 1]
    :return:
    """

    image_cv = np.copy(image)
    image_cv = np.astype(np.clip(image_cv, 0, 1)*255, np.uint8)[:, :, ::-1]  # uint8 BGR opencv format
    return image_cv




def plot_semantic_legend(
    label, 
    label_name, 
    colormap=None, 
    font_size=30,
    font_path=None,
    save_path=None,
    img_name=None):


    """Plot Colour Legend for Semantic Classes

    Parameters
    ----------
    label: numpy.ndarray, (N,), int
        One-dimensional array containing the unique labels of exsiting semantic classes
    label_names: list of string
        Label id to label name.
    font_size: int
        Font size (default: 30).
    colormap: numpy.ndarray, (M, 3), numpy.uint8
        Label id to color.
        By default, :func:`~imgviz.label_colormap` is used.
    font_path: str
        Font path.

    Returns
    -------
    res: numpy.ndarray, (H, W, 3), numpy.uint8
    Legend image of visualising semantic labels.

    """

    label = np.unique(label)
    if colormap is None:
        colormap = label_colormap()

    text_sizes = np.array(
            [
                draw_module.text_size(
                    label_name[l], font_size, font_path=font_path
                )
                for l in label
            ]
        )

    text_height, text_width = text_sizes.max(axis=0)
    legend_height = text_height * len(label) + 5
    legend_width = text_width + 20 + (text_height - 10)


    legend = np.zeros((legend_height+50, legend_width+50, 3), dtype=np.uint8)
    aabb1 = np.array([25, 25], dtype=float)
    aabb2 = aabb1 + (legend_height, legend_width)

    legend = draw_module.rectangle(
        legend, aabb1, aabb2, fill=(255, 255, 255)
    )  # fill the legend area by white colour

    y1, x1 = aabb1.round().astype(int)
    y2, x2 = aabb2.round().astype(int)

    for i, l in enumerate(label):
        box_aabb1 = aabb1 + (i * text_height + 5, 5)
        box_aabb2 = box_aabb1 + (text_height - 10, text_height - 10)
        legend = draw_module.rectangle(
            legend, aabb1=box_aabb1, aabb2=box_aabb2, fill=colormap[l]
        )
        legend = draw_module.text(
            legend,
            yx=aabb1 + (i * text_height, 10 + (text_height - 10)),
            text=label_name[l],
            size=font_size,
            font_path=font_path,
            )

    
    plt.figure(1)
    plt.title("Semantic Legend!")
    plt.imshow(legend)
    plt.axis("off")

    img_arr = imgviz.io.pyplot_to_numpy()
    plt.close()
    if save_path is not None:
        import cv2
        if img_name is not None:
            sav_dir = os.path.join(save_path, img_name)
        else:
            sav_dir = os.path.join(save_path, "semantic_class_Legend.png")
        # plt.savefig(sav_dir, bbox_inches='tight', pad_inches=0)
        cv2.imwrite(sav_dir, img_arr[:,:,::-1])
    return img_arr




def image_vis(
    pred_data_dict,
    gt_data_dict,
    # enable_sem = True
    ):
    to8b_np = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    batch_size = pred_data_dict["vis_deps"].shape[0]

    gt_dep_row = np.concatenate(np.split(gt_data_dict["vis_deps"], batch_size, 0), axis=-2)[0]
    gt_raw_dep_row = np.concatenate(np.split(gt_data_dict["deps"], batch_size, 0), axis=-1)[0]

    gt_sem_row = np.concatenate(np.split(gt_data_dict["vis_sems"], batch_size, 0), axis=-2)[0]
    gt_sem_clean_row = np.concatenate(np.split(gt_data_dict["vis_sems_clean"], batch_size, 0), axis=-2)[0]
    gt_rgb_row = np.concatenate(np.split(gt_data_dict["rgbs"], batch_size, 0), axis=-2)[0]
        
    pred_dep_row = np.concatenate(np.split(pred_data_dict["vis_deps"], batch_size, 0), axis=-2)[0]
    pred_raw_dep_row = np.concatenate(np.split(pred_data_dict["deps"], batch_size, 0), axis=-1)[0]

    pred_sem_row = np.concatenate(np.split(pred_data_dict["vis_sems"], batch_size, 0), axis=-2)[0]
    pred_entropy_row = np.concatenate(np.split(pred_data_dict["vis_sem_uncers"], batch_size, 0), axis=-2)[0]
    pred_rgb_row = np.concatenate(np.split(pred_data_dict["rgbs"], batch_size, 0), axis=-2)[0]

    rgb_diff = np.abs(gt_rgb_row - pred_rgb_row)

    dep_diff = np.abs(gt_raw_dep_row - pred_raw_dep_row)
    dep_diff[gt_raw_dep_row== 0] = 0
    dep_diff_vis = imgviz.depth2rgb(dep_diff)

    views = [to8b_np(gt_rgb_row), to8b_np(pred_rgb_row), to8b_np(rgb_diff),
            gt_dep_row, pred_dep_row, dep_diff_vis,
            gt_sem_clean_row, gt_sem_row, pred_sem_row, pred_entropy_row]

    viz = np.vstack(views)
    return viz




nyu13_colour_code = (np.array([[0, 0, 0],
                       [0, 0, 1], # BED
                       [0.9137,0.3490,0.1882], #BOOKS
                       [0, 0.8549, 0], #CEILING
                       [0.5843,0,0.9412], #CHAIR
                       [0.8706,0.9451,0.0941], #FLOOR
                       [1.0000,0.8078,0.8078], #FURNITURE
                       [0,0.8784,0.8980], #OBJECTS
                       [0.4157,0.5333,0.8000], #PAINTING
                       [0.4588,0.1137,0.1608], #SOFA
                       [0.9412,0.1373,0.9216], #TABLE
                       [0,0.6549,0.6118], #TV
                       [0.9765,0.5451,0], #WALL
                       [0.8824,0.8980,0.7608]])*255).astype(np.uint8)


# color palette for nyu34 labels
nyu34_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
    #    (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
    #    (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
    #    (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
    #    (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
    #    (178, 127, 135),       # white board

    #    (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)



# color palette for nyu40 labels
nyu40_colour_code = np.array([
       (0, 0, 0),

       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair

       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf

       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),       # blinds
       (247, 182, 210),		# desk
       (66, 188, 102),      # shelves

       (219, 219, 141),		# curtain
       (140, 57, 197),    # dresser
       (202, 185, 52),      # pillow
       (51, 176, 203),    # mirror
       (200, 54, 131),      # floor

       (92, 193, 61),       # clothes
       (78, 71, 183),       # ceiling
       (172, 114, 82),      # books
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),      # tv

       (153, 98, 156),      # paper
       (140, 153, 101),     # towel
       (158, 218, 229),		# shower curtain
       (100, 125, 154),     # box
       (178, 127, 135),       # white board

       (120, 185, 128),       # person
       (146, 111, 194),     # night stand
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),      # lamp

       (227, 119, 194),		# bathtub
       (213, 92, 176),      # bag
       (94, 106, 211),      # other struct
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)       # other prop
    ]).astype(np.uint8)


if __name__ == "__main__":
        # nyu40_class_name_string = ["void",
        # "wall", "floor", "cabinet", "bed", "chair",
        # "sofa", "table", "door", "window", "book", 
        # "picture", "counter", "blinds", "desk", "shelves",
        # "curtain", "dresser", "pillow", "mirror", "floor",
        # "clothes", "ceiling", "books", "fridge", "tv",
        # "paper", "towel", "shower curtain", "box", "white board",
        # "person", "night stand", "toilet", "sink", "lamp",
        # "bath tub", "bag", "other struct", "other furntr", "other prop"] # NYUv2-40-class

        # legend_img_arr = plot_semantic_legend(np.arange(41), nyu40_class_name_string, 
        # colormap=nyu40_colour_code,
        # save_path="/home/shuaifeng/Documents/PhD_Research/SemanticSceneRepresentations/SSR",
        # img_name="nyu40_legned.png")


        nyu13_class_name_string = ["void",
                    "bed", "books", "ceiling", "chair", "floor",
                    "furniture", "objects", "painting/picture", "sofa", "table",
                    "TV", "wall", "window"] # NYUv2-13-class

        legend_img_arr = plot_semantic_legend(np.arange(14), nyu13_class_name_string, 
        colormap=nyu13_colour_code,
        save_path="/home/shuaifeng/Documents/PhD_Research/SemanticSceneRepresentations/SSR",
        img_name="nyu13_legned.png")