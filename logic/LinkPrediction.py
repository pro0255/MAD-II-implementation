from logic.CONSTANTS import IGNORE_ZERO, VERBOSE
from logic.Performance import Performace
from enum import Enum
import numpy as np
import math


class PrediLink(Enum):
    AdamicAdar = "AdamicAdar"
    ResourceAllocationIndex = "ResourceAllocationIndex"
    CosineSimilarity = 'CosineSimilarity'
    SorensenIndex =  "SorensenIndex"
    CARBasedCommonNeighborIndex = 'CARBasedCommonNeighborIndex' 
    CommonNeighbors = 'CommonNeighbors'
    JaccardCoefficient = 'JaccardCoefficient'
    PreferentialAttachment = 'PreferentialAttachment'  



def create_matrix_deps_on_method(matrix, method):
    """Creates matrix according to predict link method
    """
    result_A = np.zeros(shape=matrix.shape)
    for y in range(result_A.shape[0]):
        y_vertex = matrix[y, :]
        for x in range(result_A.shape[1]):
            if y == x:
                continue
            x_vertex = matrix[x, :]
            res = method(y_vertex, x_vertex, matrix)
            result_A[y, x] = res
    return result_A



def adamic_adar_matrix(matrix):
    return create_matrix_deps_on_method(matrix, adamic_adar_calc)

def resource_allocation_index_matrix(matrix):
    return create_matrix_deps_on_method(matrix, resource_allocation_index_calc)

def cosine_similarity_matrix(matrix):
    return create_matrix_deps_on_method(matrix, cosine_similarity_calc)

def sorensen_index_matrix(matrix):
    return create_matrix_deps_on_method(matrix, sorensen_index_calc)

def car_based_common_neighbor_index_matrix(matrix):
    return create_matrix_deps_on_method(matrix, car_based_common_neighbor_index_calc)


def common_neighbors_matrix(matrix):
    return create_matrix_deps_on_method(matrix, cn_calc)

def preferential_attachment_matrix(matrix):
    return create_matrix_deps_on_method(matrix, prefa_calc)

def jaccard_matrix(matrix):
    return create_matrix_deps_on_method(matrix, jaccard_calc)


def adamic_adar_calc(y, x, matrix):
    """Suma through 1/log(z) intersect of neighbors(set(z)) x, y.
    """
    edges_y = np.where(y > 0)
    edges_x = np.where(x > 0)
    set_y = set(np.array(edges_y).flatten()) 
    set_x = set(np.array(edges_x).flatten())
    intersect = set_y.intersection(set_x)
    suma = 0
    for e_z in intersect:
        vertex_z = matrix[e_z, :]
        k_z = len(np.where(np.array(vertex_z) > 0))
        suma += 1 / (math.log(k_z))
    return suma

def resource_allocation_index_calc(y, x, matrix):
    """It highly penalizes higer degree nodes.
    """
    edges_y = np.where(y > 0)
    edges_x = np.where(x > 0)
    set_y = set(np.array(edges_y).flatten()) 
    set_x = set(np.array(edges_x).flatten())
    intersect = set_y.intersection(set_x)
    suma = 0
    for e_z in intersect:
        vertex_z = matrix[e_z, :]
        k_z = len(np.where(np.array(vertex_z) > 0))
        suma += 1 / k_z
    return suma


def cosine_similarity_calc(y, x, matrix):
    edges_y = np.where(y > 0)
    edges_x = np.where(x > 0)
    set_y = set(np.array(edges_y).flatten()) 
    set_x = set(np.array(edges_x).flatten())
    intersect = set_y.intersection(set_x)
    nom = len(intersect)
    k_x = len(edges_x)
    k_y = len(edges_y)
    den = math.sqrt(k_x*k_y)
    if den == 0:
        return 0
    else:
        return nom / den

def sorensen_index_calc(y, x, matrix):
    edges_y = np.where(y > 0)
    edges_x = np.where(x > 0)
    set_y = set(np.array(edges_y).flatten()) 
    set_x = set(np.array(edges_x).flatten())
    intersect = set_y.intersection(set_x)
    nom = len(intersect)
    k_x = len(edges_x)
    k_y = len(edges_y)  
    nom = 2 * nom
    den = k_x + k_y

    if den == 0:
        return 0
    else:
        return nom / den 

def car_based_common_neighbor_index_calc(y, x, matrix):
    CN_part = cn_calc(y, x)

    edges_y = np.where(y > 0)
    edges_x = np.where(x > 0)
    set_y = set(np.array(edges_y).flatten()) 
    set_x = set(np.array(edges_x).flatten())
    intersect = set_y.intersection(set_x)   
    suma = 0
    for z in intersect:
        vertex_z = matrix[z, :]
        n_z = np.where(np.array(vertex_z) > 0)
        set_n_z = set(n_z)
        y_z = set_n_z.intersection(intersect)
        suma += len(y_z) / 2
    return CN_part * suma



def cn_calc(y, x):
    edges_y = np.where(y > 0)
    edges_x = np.where(x > 0)
    set_y = set(np.array(edges_y).flatten()) 
    set_x = set(np.array(edges_x).flatten())
    intersect = set_y.intersection(set_x)
    same = len(intersect)
    return same

def prefa_calc(y, x):
    k_y_w = np.argwhere(y > 0).flatten()
    k_x_w = np.argwhere(x > 0).flatten()
    k_y = len(k_y_w)
    k_x = len(k_x_w)
    return k_y * k_x

def jaccard_calc(y, x):
    edges_y = np.where(y > 0)
    edges_x = np.where(x > 0)
    set_y = set(np.array(edges_y).flatten()) 
    set_x = set(np.array(edges_x).flatten()) 
    intersect = set_y.intersection(set_x)
    unioned = set_y.union(set_x)
    nom = len(intersect)
    den = len(unioned)
    if den == 0:
        return 0
    return nom/den

def apply_threshold(matrix, threshold):
    thresholded_matrix = np.copy(matrix)
    res = np.where(thresholded_matrix > threshold, 1, 0)
    return res

def create_confusion_tuple(matrix, m_matrix, predicted_matrix, t):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for y in range(matrix.shape[0]):
        for x in range(y+1, matrix.shape[1]):
            available_value = matrix[y, x]
            predicted_value = m_matrix[y, x]

            if IGNORE_ZERO and predicted_matrix is not None and predicted_matrix[y, x] == 0:
                continue

            if available_value == True and predicted_value == True:
                true_positive += 1

            if available_value == False and predicted_value == False:
                true_negative += 1

            if available_value == False and predicted_value == True:
                false_positive += 1

            if available_value == True and predicted_value == False:
                false_negative += 1

    res = (true_positive, true_negative, false_positive, false_negative)

    if VERBOSE:
        print(f'{t}\n\n\n\tTrue positive {true_positive}\n\tTrue negative {true_negative}\n\tFalse positive {false_positive}\n\tFalse negative {false_negative}\n')
    return res



    
#TODO: make all methods
prediction_link_dictionary = {
    PrediLink.AdamicAdar: adamic_adar_matrix, #TODO
    PrediLink.ResourceAllocationIndex: resource_allocation_index_matrix,
    PrediLink.CosineSimilarity: cosine_similarity_matrix,
    PrediLink.SorensenIndex: sorensen_index_matrix,
    PrediLink.CARBasedCommonNeighborIndex: car_based_common_neighbor_index_matrix,
    PrediLink.CommonNeighbors: common_neighbors_matrix,
    PrediLink.JaccardCoefficient: jaccard_matrix,
    PrediLink.PreferentialAttachment: preferential_attachment_matrix,
}