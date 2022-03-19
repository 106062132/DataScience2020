import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=5):
    """
    Applies weight sharing to the given model
    """
    for name, module in model.named_children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        quan_range = 2 ** bits
        if len(shape) == 2:  # fully connected layers
            print(f'{name:20} | {str(module.weight.size()):35} | => quantize to {quan_range} indices')
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            #print("correct",weight.shape)
            #print("correct",weight)
            # weight sharing by kmeans
            space = np.linspace(min(mat.data), max(mat.data), num=quan_range)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(mat.data.reshape(-1, 1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight

            # Insert to model
            module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
        elif len(shape) == 4:  # convolution layers
            #################################
            # TODO:
            #    Suppose the weights of a certain convolution layer are called "W"
            #       1. Get the unpruned (non-zero) weights, "non-zero-W",  from "W"
            #       2. Use KMeans algorithm to cluster "non-zero-W" to (2 ** bits) categories
            #       3. For weights belonging to a certain category, replace their weights with the centroid
            #          value of that category
            #       4. Save the replaced weights in "module.weight.data", and need to make sure their indices
            #          are consistent with the original
            #   Finally, the weights of a certain convolution layer will only be composed of (2 ** bits) float numbers
            #   and zero
            #   --------------------------------------------------------
            #   In addition, there is no need to return in this function ("model" can be considered as call by
            #   reference)
            #################################

            print(f'{name:20} | {str(module.weight.size()):35} | => quantize to {quan_range} indices')

            data = []
            for i in range(shape[0]):
                for j in range(shape[1]):
                    data.append(weight[i][j])
            data = np.concatenate(data)
            #print("wrong",data.shape)
            mat = csr_matrix(data)

            # weight sharing by kmeans
            space = np.linspace(min(mat.data), max(mat.data), num=quan_range)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(mat.data.reshape(-1, 1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            mat = mat.toarray()

            
            new_mat_2 = []
            new_mat_4 = []
            for i in range(shape[0]*shape[1]):
                new_mat_1 = []
                for j in range(shape[2]):
                    new_mat_1.append(mat[i*shape[2]+j])
                new_mat_2.append(new_mat_1)
            for i in range(shape[0]):
                new_mat_3 = []
                for j in range(shape[1]):
                    new_mat_3.append(new_mat_2[i*shape[1]+j])
                new_mat_4.append(new_mat_3)
            new_mat_4 = np.array(new_mat_4)
            #print(new_mat_4.shape)
            # Insert to model
            module.weight.data = torch.from_numpy(new_mat_4).to(dev)
