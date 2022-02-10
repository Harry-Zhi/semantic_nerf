import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def batchify_rays(render_fn, rays_flat, chunk=1024 * 32):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_fn(rays_flat[i:i + chunk])
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def lr_poly_decay(base_lr, iter, max_iter, power):
    """ Polynomial learning rate decay
    Polynomial Decay provides a smoother decay using a polynomial function and reaches a learning rate of 0
    after max_update iterations.
    https://kiranscaria.github.io/general/2019/08/16/learning-rate-schedules.html

    max_iter: number of iterations to perform before the learning rate is taken to .
    power: the degree of the polynomial function. Smaller values of power produce slower decay and
        large values of learning rate for longer periods.
    """
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_exp_decay(base_lr, exp_base_lr, current_step, decay_steps):
    """ lr = lr0 * decay_base^(−kt)
    """
    new_lrate = base_lr * (exp_base_lr ** (current_step / decay_steps))
    return new_lrate


def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()


def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, ignore_label):
    if (true_labels == ignore_label).all():
        return [0]*4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    valid_pix_ids = true_labels!=ignore_label
    predicted_labels = predicted_labels[valid_pix_ids] 
    true_labels = true_labels[valid_pix_ids]
    
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious


def calculate_depth_metrics(depth_trgt, depth_pred):
    """ Computes 2d metrics between two depth maps
    
    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth
    Returns:
        Dict of metrics
    """
    mask1 = depth_pred>0 # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt<10) * (depth_trgt>0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred-depth_trgt)
    abs_rel = abs_diff/depth_trgt
    sq_diff = abs_diff**2
    sq_rel = sq_diff/depth_trgt
    sq_log_diff = (np.log(depth_pred)-np.log(depth_trgt))**2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25**2).astype('float')
    r3 = (thresh < 1.25**3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics