import numpy as np
import torch
from torch.nn.modules.module import Module


class PruningModule(Module):
    def prune_by_percentile(self, q={'conv1': 30, 'conv2': 96, 'conv3': 99, 'conv4': 99, 'conv5': 99, 'fc1': 99, 'fc2': 99, 'fc3': 97}):
        ########################
        # TODO
        # 	For each layer of weights W (including fc and conv layers) in the model, obtain the qth percentile of W as
        # 	the threshold, and then set the nodes with weight W less than threshold to 0, and the rest remain unchanged.
        ########################

        # Calculate percentile value
        # Prune the weights and mask
        threshold = {}
        for name, p in self.named_parameters():
            if 'bias' in name:
                continue
            tmp = p.data.cpu().numpy()
            tmp = tmp[np.nonzero(tmp)]
            q_name = name.split('.')[0]
            threshold[q_name] = np.percentile(abs(tmp), q[q_name])
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                print(f'Pruning with threshold : {threshold[name]} for layer {name}')
                self.prune(module, threshold[name])

    

    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():
            #################################
            # TODO:
            #    Only fully connected layers were considered, but convolution layers also needed
            #################################
            if name in ['fc1', 'fc2', 'fc3', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                self.prune(module, threshold)

    def prune(self, module, threshold):

        #################################
        # TODO:
        #    1. Use "module.weight.data" to get the weights of a certain layer of the model
        #    2. Set weights whose absolute value is less than threshold to 0, and keep the rest unchanged
        #    3. Save the results of the step 2 back to "module.weight.data"
        #    --------------------------------------------------------
        #    In addition, there is no need to return in this function ("module" can be considered as call by
        #    reference)
        #################################
        tmp = module.weight.data.cpu().numpy()
        #tmp2 = module.mask.data.cpu().numpy()
        
        #newmask = np.where(abs(tmp) < threshold, 0, tmp2)
        tmp = np.where(abs(tmp) < threshold, 0, tmp)
        module.weight.data = torch.from_numpy(tmp).to(module.weight.device)
        #module.mask.data = torch.from_numpy(newmask).to(module.mask.device)




