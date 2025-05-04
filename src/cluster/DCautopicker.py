import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import cv2
import copy
from ..helper import printf, EXPERIMENTAL

def cal_contour_area(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for contr in contours:
        area += cv2.contourArea(contr) 
    return area


def mode_discriminator(_curves):
    DCcurves = dict()
        
    mode_in_order = np.arange(len(_curves))
    c_dist = np.array([])
    for i in range(len(_curves)) :
        f_mean = np.mean(_curves[i][1])
        v_mean = np.mean(_curves[i][0])
        c_dist = np.append(c_dist, np.sqrt(f_mean**2 + v_mean**2))
    mode_in_order = mode_in_order[np.argsort(c_dist)]
    
    for m, id in enumerate(mode_in_order):
        DCcurves.update({
            m : {
                "velocity":  _curves[id][0],
                "frequency": _curves[id][1],
            }
        })
    return DCcurves
                
                   
            
def Picks(  power_matrix,
            velocity,
            frequency,
            kvelocity,
            threshold_power=0.8,
            DBSCAN_eps=8,
            DBSCAN_samples=20,
            cmp=None
            ):
    """ Finds dispersion curves from dispersion spectrum
    Parameter
    ---------
    power_matrix    : ndarray
                    normalized dispersion spectrum, shape = (len(velocity), len(frequency)) 
    velocity    : ndarray | list
    frequency   : ndarray | list
    threshold_power : float < 1
    DBSCAN_eps  : 
    DBSCAN_samples  : 
    
    Return
    ------
    DCcurves : dict , it's keys represent mode number
              {0:{
                  "velocity": ...,
                   "frequency": ...},
               ...
               M:{
                   "velocity": ...,
                   "frequency": ...}
                }
    
    """
    assert power_matrix.shape == (len(velocity), len(frequency)) , ValueError
    assert 0 <= threshold_power <= 1 , ValueError
    
    # vv, ff = np.meshgrid(velocity, frequency)
    filtered_mat = np.where(power_matrix>threshold_power, power_matrix, 0)
    kvelIP = np.poly1d(np.polyfit(frequency, kvelocity, 5))
    
    # threshold filering
    thresh_above_ids = np.argwhere(power_matrix>threshold_power)
    vfmat = []
    for id in thresh_above_ids:
        vfmat.append([velocity[id[0]], frequency[id[1]]])
    vfmat = np.asarray(vfmat)
    printf(f"cmp={cmp} vfmat shape, {vfmat.shape}, {vfmat.nbytes}\n*********", verbose=EXPERIMENTAL)
    # clustering
    try:
        dbscan = DBSCAN(eps=DBSCAN_eps, 
                        min_samples=DBSCAN_samples, 
                        algorithm="brute")
        clabels = dbscan.fit_predict(vfmat)
    except:
        printf(f"cmp={cmp}\n*********", verbose=EXPERIMENTAL)
        raise 
    # save clusters to dictionaries
    clusters = dict()
    for l in np.unique(clabels):
        clusters.update({
            l : []
        })
    clusters_in_power_spec = copy.deepcopy(clusters)
    for i,j in enumerate(clabels):
        #print(i,j)
        clusters[j].append(vfmat[i])
        clusters_in_power_spec[j].append(thresh_above_ids[i])

    cluster_vf = []
    cluster_po_ids = []
    for k in clusters_in_power_spec.keys():
        cluster_vf.append(np.asarray(clusters[k]))
        cluster_po_ids.append(np.asarray(clusters_in_power_spec[k]))

    # print(cluster_vf)   
    # print(cluster_po_ids)
    
    # distinguising modes and noise
    modes = []

    for l, [powrsids, vfmat] in enumerate(zip(cluster_po_ids, cluster_vf)):
        cluster_matrix = np.zeros_like(filtered_mat, dtype="float")
        for pid in powrsids:
            cluster_matrix[pid[0], pid[1]] = filtered_mat[pid[0], pid[1]]
        
        cpts_id = np.argmax(cluster_matrix, axis=0)
        # print(cpts_id)
        
        non_zero_ids = np.where(cpts_id!=0)[0]
        if len(non_zero_ids) > 0:
            p = np.poly1d(np.polyfit(frequency[non_zero_ids], velocity[cpts_id[non_zero_ids]], 20))
            interpv = p(frequency[non_zero_ids])
            diff = np.linalg.norm(velocity[cpts_id[non_zero_ids]] - interpv)
            if diff < 100:
                bin_img = np.where(cluster_matrix > 0, 255, 0).astype(np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                bin_img = cv2.erode(bin_img, kernel, iterations=1)
                bin_img = cv2.dilate(bin_img, kernel, iterations=1) 
                area = cal_contour_area(bin_img)
                kvel = kvelIP([np.mean(frequency[non_zero_ids])])
                printf(f"cmp={cmp} {np.mean(interpv)} > {kvel[0]}={np.mean(interpv) > kvel[0]}\n****************", verbose=EXPERIMENTAL)
                
                if area > 50 and np.mean(interpv) > kvel[0]:   
                    modes.append([interpv, frequency[non_zero_ids]])
    
    curves = mode_discriminator(modes)
    
    return curves

    