#  ------------------------------------------------------------------------------------------
#  This code is reconstructed based on loralib (https://github.com/microsoft/LoRA) by Baijiong Lin.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from PIL import Image
import numpy as np

def save_image(data, path):
    """Helper function to save a numpy array as an image."""
    img = Image.fromarray(data)
    img.save(path)


def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        fan_in_fan_out: bool = False,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if self.r > 0:
            self.scaling = self.lora_alpha / self.r
        # Mark the weight as unmerged
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}

    def register_lora_param(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            self.register_parameter(f'{lora_name}_lora_A', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[1])))
                )
            self.register_parameter(f'{lora_name}_lora_B', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.r)))
                )
            eval(f'self.{param_name}').requires_grad = False

    def register_lora_param_svd(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            self.register_parameter(f'{lora_name}_lora_U', 
                # nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[1])))
                nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.r)))
                )
            self.register_parameter(f'{lora_name}_lora_S', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r)))
                )                
            self.register_parameter(f'{lora_name}_lora_V', 
                # nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[0])))
                nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[1], self.r)))
                )
            eval(f'self.{param_name}').requires_grad = False

    def init_lora_param_svd(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_U'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_U'), a=math.sqrt(5))
                # nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_S'), a=math.sqrt(5))
                nn.init.zeros_(eval(f'self.{lora_name}_lora_S'))
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_V'), a=math.sqrt(5))
                # nn.init.zeros_(eval(f'self.{lora_name}_lora_V'))

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
                nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))

    def init_lora_param_vera(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_B'), a=math.sqrt(5))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).view(eval(f'self.{param_name}').shape))

    def merge_AB_new(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        # return self.transpose((eval(f'self.{lora_name}_lora_A').permute(1,0) @ eval(f'self.{lora_name}_lora_B').permute(1,0)).view(eval(f'self.{param_name}').shape))        
        # return self.transpose((eval(f'self.{lora_name}_lora_A.detach()').permute(1,0) @ eval(f'self.{lora_name}_lora_B.detach()').permute(1,0)).view(eval(f'self.{param_name}').shape))        
        # return self.transpose(torch.pinverse(eval(f'self.{lora_name}_lora_B.detach()') @ eval(f'self.{lora_name}_lora_A.detach()')).view(eval(f'self.{param_name}').shape))        
        if torch.all(eval(f'self.{lora_name}_lora_B')) != 0 and torch.all(eval(f'self.{lora_name}_lora_A')) != 0:            
            # inverse w/ scaling
            # inverse_BA = torch.inverse(eval(f'self.{lora_name}_lora_B.detach()/self.{lora_name}_lora_B.detach().min()*10') @ eval(f'self.{lora_name}_lora_A.detach()/self.{lora_name}_lora_A.detach().min()*10'))
            # inverse_BA *= eval(f'self.{lora_name}_lora_B.detach().min()/10*self.{lora_name}_lora_A.detach().min()/10')
            # return self.transpose(inverse_BA.reshape(eval(f'self.{param_name}').shape))
            # inverse w/o scaling
            inverse_BA = torch.inverse(eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A'))
            return self.transpose(inverse_BA).reshape(eval(f'self.{param_name}').shape)                    
        else:
            return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).view(eval(f'self.{param_name}').shape))                    
        
        # return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).view(eval(f'self.{param_name}').shape))                                
    
    def merge_BA_repeat(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        return self.transpose((eval(f'self.{lora_name}_lora_B').repeat(5,5) @ eval(f'self.{lora_name}_lora_A').repeat(5,5)).view(eval(f'self.{param_name}').shape))
    
    def merge_lora_param(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # detach() is very important here
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode='update')

    def merge_lora_param_svd(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        res_BA_list = []
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            
            lora_name = self.params_with_lora[param_name]                
            new_lora_U = eval(f'self.{lora_name}_lora_U')
            new_lora_S = eval(f'self.{lora_name}_lora_S') 
            new_lora_V = eval(f'self.{lora_name}_lora_V')            
            
            # p_new = p.detach() + torch.mm(self.w_lora_V.t(), torch.mm(torch.diag(self.w_lora_S), self.w_lora_U)).view(eval(f'self.{param_name}').shape) * self.scaling            
            p_new = p.detach() + torch.mm(torch.mm(self.w_lora_U, torch.diag(self.w_lora_S)),self.w_lora_V.t()).view(eval(f'self.{param_name}').shape) * self.scaling            
            set_param(self, param_name, param=p_new, mode='update')
            
        return res_BA_list

    def merge_lora_param_res(self, *res_layers):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        res_BA_list = []
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # detach() is very important here
            if self.r > 0 :
                p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            else : 
                p_new = p.detach()
            for i in range(len(res_layers[0])):
                # p_new += prev_layers[0][i].merge_BA(param_name) * prev_layers[0][i].scaling
                # p_new += prev_layers[0][i].merge_AB_new(param_name) * prev_layers[0][i].scaling
                # p_res = set_param(res_layers[0][i], param_name, mode='get')
                # p_res_new = p_res.detach() + res_layers[0][i].merge_BA(param_name).detach() * res_layers[0][i].scaling

                # res_BA = res_layers[0][i].merge_BA(param_name).detach() * res_layers[0][i].scaling
                # res_BA = p_res.detach() + res_layers[0][i].merge_BA(param_name).detach() * res_layers[0][i].scaling
                # res_BA = res_layers[0][i].merge_BA(param_name).detach() * res_layers[0][i].scaling
                res_BA = res_layers[0][i].merge_BA(param_name) * res_layers[0][i].scaling
        
                # BAd = torch.zeros_like(res_BA)
                # if torch.all(res_BA) != 0:      
                    # filters = res_BA.detach().cpu().numpy()
                    # num_filters = filters.shape[0]
                    # for i in range(0, min(num_filters, 64), 20):
                    #     filter_img = filters[i, 0]
                    #     filter_img = ((filter_img-filter_img.min())/filter_img.max() * 255 ).astype(np.uint8)  # Normalize to [0, 255]
                    #     save_image(filter_img, f'filter_{i}.png')

                    # Z_prime_enc = torch.randn_like([16,192,16,16])
                    # Z_prime_enc = torch.randn([16,192,16,16])
                    # Y_prime = res_BA @ Z_prime_enc

                    # max_val = torch.max(torch.abs(res_BA))
                    # scaled_res_BA = res_BA / max_val      
                    # inverse_res_BA = torch.inverse(scaled_res_BA) * max_val

                    # max_val = torch.max(torch.abs(res_BA))
                    # scaled_res_BA = res_BA / max_val      
                    # inverse_res_BA = torch.inverse(scaled_res_BA) * max_val

                    # inverse_res_BA = torch.inverse(res_BA)
                    # BAd = inverse_res_BA
                    # BAd = inverse_res_BA - p.detach() 

                    # Z_prime_dec = (p_new + BAd) @ Y_prime

                    # max_val = torch.max(torch.abs(res_BA))
                    # scaled_res_BA = res_BA / max_val      
                    # res_BA = torch.inverse(scaled_res_BA) * max_val

                    # filters = res_BA.detach().cpu().numpy()
                    # num_filters = filters.shape[0]
                    # for i in range(0, min(num_filters, 64), 20):
                    #     filter_img = filters[i, 0]
                    #     filter_img = ((filter_img-filter_img.min())/filter_img.max() * 255 ).astype(np.uint8)  # Normalize to [0, 255]
                    #     save_image(filter_img, f'filter_{i}_inverted.png')
                    # res_BA = torch.flip(res_BA, [2,3])
                                
                res_BA_list.append(res_BA)
                p_new += res_BA



                # res_BA_zero = torch.zeros_like(res_BA)
                # res_BA_list.append(res_BA_zero)


                # Z_prime_enc = torch.randn_like(p_res_new)
                # Y_prime = p_res_new @ Z_prime_enc

                # inverse_p_res_new = torch.inverse(p_res_new)
                # BAd = inverse_p_res_new - p.detach() 

                # Z_prime_dec = (p_new + BAd) @ Y_prime


                # res_BA = torch.zeros_like(p_res_new)
                # if torch.all(p_res_new) != 0:            
                #     res_BA = torch.inverse(p_res_new) - p.detach() 
                #     res_BA = torch.flip(res_BA, [2,3])
                # res_BA_list.append(res_BA)
                # p_new += res_BA.detach()

                # p_new += prev_layers[0][i].merge_AB_new(param_name) * prev_layers[0][i].scaling

            #     #for test
            #     p_new = p.detach()

            #     prev_lora = prev_layers[0][i].merge_AB_new(param_name)
            #     if prev_lora.mean() != 0:
            #         scale = p_new.mean()/prev_lora.mean()
            #         prev_lora *= scale                
            #         scale_value.append(scale)
            #     p_new += prev_lora
            set_param(self, param_name, param=p_new, mode='update')
        return res_BA_list

    def merge_lora_param_res_concat(self, *res_layers):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        res_BA_list = []
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            
            lora_name = self.params_with_lora[param_name]                
            new_lora_B = eval(f'self.{lora_name}_lora_B')
            new_lora_A = eval(f'self.{lora_name}_lora_A') 

            for i in range(len(res_layers[0])):
                #inverse test                
                # res_lora_A = torch.transpose(eval(f'res_layers[0][{i}].{lora_name}_lora_B'), 0, 1)
                # res_lora_B = torch.transpose(eval(f'res_layers[0][{i}].{lora_name}_lora_A'), 0, 1)
                # new_lora_B = torch.cat((new_lora_B, res_lora_B), dim=1)
                # new_lora_A = torch.cat((new_lora_A, res_lora_A), dim=0)               

                new_lora_B = torch.cat((new_lora_B, eval(f'res_layers[0][{i}].{lora_name}_lora_B')), dim=1)
                new_lora_A = torch.cat((new_lora_A, eval(f'res_layers[0][{i}].{lora_name}_lora_A')), dim=0)

            # detach() is very important here            
            # if self.r > 0 :
            #     p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            # else : 
            #     p_new = p.detach()
                 
            # p_new = p.detach() +  (new_lora_B @ new_lora_A).t().view(eval(f'self.{param_name}').shape) * self.scaling
            p_new = p.detach() + (new_lora_B @ new_lora_A).view(eval(f'self.{param_name}').shape) * self.scaling
            set_param(self, param_name, param=p_new, mode='update')
            
        return res_BA_list        
    
    def merge_lora_param_repeat(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # detach() is very important here
            p_new = p.detach() + self.merge_BA_repeat(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode='update')
            

    def add_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data += self.merge_BA(param_name) * self.scaling

    def add_lora_data_svd(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            new_lora_U = eval(f'self.{lora_name}_lora_U')
            new_lora_S = eval(f'self.{lora_name}_lora_S') 
            new_lora_V = eval(f'self.{lora_name}_lora_V')  
            # eval(f'self.{param_name}').data += torch.mm(self.w_lora_V.t(), torch.mm(torch.diag(self.w_lora_S), self.w_lora_U)).view(eval(f'self.{param_name}').shape) * self.scaling            
            eval(f'self.{param_name}').data += torch.mm(torch.mm(self.w_lora_U, torch.diag(self.w_lora_S)),self.w_lora_V.t()).view(eval(f'self.{param_name}').shape) * self.scaling            

    def add_lora_data_repeat(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data += self.merge_BA_repeat(param_name) * self.scaling

    def sub_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA(param_name) * self.scaling

    def sub_lora_data_res(self, res_BA_list, *res_layers):
        r"""NOT differentiable"""
        # if len(prev_layers[0]) == 0:
        for param_name, lora_name in self.params_with_lora.items():
            if self.r > 0:
                eval(f'self.{param_name}').data -= self.merge_BA(param_name) * self.scaling   
            # p = set_param(self, param_name, mode='get')    #for residual 
            for i in range(len(res_layers[0])):
                    # eval(f'self.{param_name}').data -= prev_layers[0][i].merge_BA(param_name) * prev_layers[0][i].scaling                          
                    # eval(f'self.{param_name}').data -= prev_layers[0][i].merge_AB_new(param_name) * prev_layers[0][i].scaling      
                    # p = set_param(self, param_name, mode='get')
                    # p_new = p.detach()
                    # prev_lora = prev_layers[0][i].merge_AB_new(param_name)
                    # if prev_lora.mean() != 0:
                    #     prev_lora *= scale_value[i]                                      
                    # eval(f'self.{param_name}').data -= prev_lora 
                # p_res = set_param(res_layers[0][i], param_name, mode='get')
                # p_res_new = p_res.detach() + res_layers[0][i].merge_BA(param_name).detach() * res_layers[0][i].scaling

                # res_BA = torch.zeros_like(p_res_new)
                # if torch.all(p_res_new) != 0:            
                #     res_BA = torch.inverse(p_res_new) - p.detach()  
                             
                    
                eval(f'self.{param_name}').data -= res_BA_list[i]
                # eval(f'self.{param_name}').data -= res_BA_list[i].detach()                                              

    def sub_lora_data_res_concat(self, res_BA_list, *res_layers):
        r"""NOT differentiable"""
        # if len(prev_layers[0]) == 0:
        for param_name, lora_name in self.params_with_lora.items():
            new_lora_B = eval(f'self.{lora_name}_lora_B')
            new_lora_A = eval(f'self.{lora_name}_lora_A') 

            for i in range(len(res_layers[0])):
                #inverse test
                # res_lora_A = torch.transpose(eval(f'res_layers[0][{i}].{lora_name}_lora_B'), 0, 1)
                # res_lora_B = torch.transpose(eval(f'res_layers[0][{i}].{lora_name}_lora_A'), 0, 1)
                # new_lora_B = torch.cat((new_lora_B, res_lora_B), dim=1)
                # new_lora_A = torch.cat((new_lora_A, res_lora_A), dim=0)
                                
                new_lora_B = torch.cat((new_lora_B, eval(f'res_layers[0][{i}].{lora_name}_lora_B')), dim=1)
                new_lora_A = torch.cat((new_lora_A, eval(f'res_layers[0][{i}].{lora_name}_lora_A')), dim=0)            

            # eval(f'self.{param_name}').data -=  (new_lora_B @ new_lora_A).t().view(eval(f'self.{param_name}').shape) * self.scaling
            eval(f'self.{param_name}').data -= (new_lora_B @ new_lora_A).view(eval(f'self.{param_name}').shape) * self.scaling

    def sub_lora_data_svd(self):
        r"""NOT differentiable"""
        # if len(prev_layers[0]) == 0:
        for param_name, lora_name in self.params_with_lora.items():
            new_lora_U = eval(f'self.{lora_name}_lora_U')
            new_lora_S = eval(f'self.{lora_name}_lora_S') 
            new_lora_V = eval(f'self.{lora_name}_lora_V')                      

            # eval(f'self.{param_name}').data -= torch.mm(self.w_lora_V.t(), torch.mm(torch.diag(self.w_lora_S), self.w_lora_U)).view(eval(f'self.{param_name}').shape) * self.scaling   
            eval(f'self.{param_name}').data -= torch.mm(torch.mm(self.w_lora_U, torch.diag(self.w_lora_S)), self.w_lora_V.t()).view(eval(f'self.{param_name}').shape) * self.scaling   

    def sub_lora_data_repeat(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA_repeat(param_name) * self.scaling

    def sub_lora_data_vera(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA_vera(param_name) * self.scaling

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
            # Make sure that the weights are not merged
                self.sub_lora_data()
            self.merged = False 

        else:
            if not self.merged and self.r > 0:
            # Merge the weights and mark it
                self.add_lora_data()
            self.merged = True 

    def lora_train_svd(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
            # Make sure that the weights are not merged
                self.sub_lora_data_svd()
            self.merged = False 

        else:
            if not self.merged and self.r > 0:
            # Merge the weights and mark it
                self.add_lora_data_svd()
            self.merged = True 

    def lora_train_repeat(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
            # Make sure that the weights are not merged
                self.sub_lora_data_repeat()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
            # Merge the weights and mark it
                self.add_lora_data_repeat()
            self.merged = True 


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        nn.Embedding.reset_parameters(self)
        self.init_lora_param()

    def init_lora_param(self):
        if hasattr(self, 'w_lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.w_lora_A)
            nn.init.normal_(self.w_lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        self.lora_train(mode)
        
    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Embedding.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Embedding.forward(self, x, **kwargs)

class Linear_svd(nn.Linear, LoRALayer):
    # LoRA implemented in a Linear layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out)

        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param_svd()
        nn.Linear.reset_parameters(self)
        self.init_lora_param_svd()
        self.weight.data = self.transpose(self.weight.data)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)      
        self.lora_train_svd(mode)

    def forward(self, x: torch.Tensor, **kwargs):
        if self.r > 0 and not self.merged:
            self.merge_lora_param_svd()
            result = nn.Linear.forward(self, x, **kwargs)
            self.sub_lora_data_svd()
            return result
        else:
            return nn.Linear.forward(self, x, **kwargs)

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a Linear layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out)

        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        nn.Linear.reset_parameters(self)
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Linear.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Linear.forward(self, x, **kwargs)

class Conv1d(nn.Conv1d, LoRALayer):
    # LoRA implemented in a Conv1d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv1d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv1d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv1d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv1d.forward(self, x, **kwargs)

class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a Conv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):
        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv2d.forward(self, x, **kwargs)
            # self.merge_lora_param()
            # result = nn.Conv2d.forward(self, x, **kwargs)
            # self.sub_lora_data()
            # return result

class Conv2d_res(nn.Conv2d, LoRALayer):
    # LoRA implemented in a Conv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)      
        # self.lora_train(mode)

    def forward(self, x: torch.Tensor, *prev_layers, **kwargs):
        if self.r == 0:
            return nn.Conv2d.forward(self, x, **kwargs)            
        elif self.r > 0 and not self.merged:
            """
            res_BA = self.merge_lora_param_res(prev_layers)
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data_res(res_BA, prev_layers)
            """

            """ residual concat """
            res_BA = self.merge_lora_param_res_concat(prev_layers)
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data_res_concat(res_BA, prev_layers)            
            return result
        else:
            """
            res_BA = self.merge_lora_param_res(prev_layers)
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data_res(res_BA, prev_layers)
            return result
            """

            """ residual concat """
            res_BA = self.merge_lora_param_res_concat(prev_layers)
            result = nn.Conv2d.forward(self, x, **kwargs)
            self.sub_lora_data_res_concat(res_BA, prev_layers)            
            return result            
            # return nn.Conv2d.forward(self, x, **kwargs)
        
class ConvTranspose2d(nn.ConvTranspose2d, LoRALayer):
    # LoRA implemented in a deConv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.ConvTranspose2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.ConvTranspose2d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.ConvTranspose2d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.ConvTranspose2d.forward(self, x, **kwargs)

class ConvTranspose2d_res(nn.ConvTranspose2d, LoRALayer):
    # LoRA implemented in a deConv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.ConvTranspose2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.ConvTranspose2d.train(self, mode)      
        # self.lora_train(mode)

    def forward(self, x: torch.Tensor, *prev_layers, **kwargs):
        if self.r == 0:
            return nn.ConvTranspose2d.forward(self, x, **kwargs)
        elif self.r > 0 and not self.merged:
            """
            res_BA = self.merge_lora_param_res(prev_layers)
            # set_param(prev_layers[0], 'weight', param=res_BA[0], mode='update')
            # result = nn.Conv2d.forward(prev_layers[0], x, **kwargs)      
            result = nn.ConvTranspose2d.forward(self, x, **kwargs)
            self.sub_lora_data_res(res_BA, prev_layers)
            return result
            """
            """ residual concat """
            res_BA = self.merge_lora_param_res_concat(prev_layers)
            result = nn.ConvTranspose2d.forward(self, x, **kwargs)
            self.sub_lora_data_res_concat(res_BA, prev_layers)
            return result            


        else:
            """
            res_BA = self.merge_lora_param_res(prev_layers)
            result = nn.ConvTranspose2d.forward(self, x, **kwargs)
            self.sub_lora_data_res(res_BA, prev_layers)
            return result 
            """
            """ residual concat """            
            res_BA = self.merge_lora_param_res_concat(prev_layers)
            result = nn.ConvTranspose2d.forward(self, x, **kwargs)
            self.sub_lora_data_res_concat(res_BA, prev_layers)
            return result             
            # return nn.ConvTranspose2d.forward(self, x, **kwargs)
        
class ConvTranspose2drepeat(nn.ConvTranspose2d, LoRALayer):
    # LoRA implemented in a deConv2d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_channels))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups, r))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.ConvTranspose2d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.ConvTranspose2d.train(self, mode)      
        self.lora_train_repeat(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param_repeat()
            result = nn.ConvTranspose2d.forward(self, x, **kwargs)
            self.sub_lora_data_repeat()
            return result
        else:
            return nn.ConvTranspose2d.forward(self, x, **kwargs)



class Conv3d(nn.Conv3d, LoRALayer):
    # LoRA implemented in a Conv3d layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert type(kernel_size) is int
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels//self.groups*kernel_size, r*kernel_size))
            )
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        nn.Conv3d.reset_parameters(self)
        self.init_lora_param()

    def train(self, mode: bool = True):
        nn.Conv3d.train(self, mode)      
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Conv3d.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Conv3d.forward(self, x, **kwargs)

class MultiheadAttention(nn.MultiheadAttention, LoRALayer):
    # LoRA implemented in a MultiheadAttention layer
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        enable_lora: list = ['q', 'k', 'v', 'o'],
        r: int = 0, 
        lora_alpha: int = 1, 
        **kwargs
    ):
        nn.MultiheadAttention.__init__(self, embed_dim, num_heads, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        # Actual trainable parameters
        if self.r > 0:
            if 'o' in enable_lora:
                self.params_with_lora.update({'out_proj.weight': 'o'})

            if not self._qkv_same_embed_dim:
                for n in ['q', 'k', 'v']:
                    if n in enable_lora:
                        self.params_with_lora.update({f'{n}_proj_weight': n})
                self.register_lora_param()
                nn.MultiheadAttention._reset_parameters(self)
                self.init_lora_param()
            else:
                lora_name, enable_lora_bool = '', []
                for n in ['q', 'k', 'v']:
                    if n in enable_lora:
                        lora_name += n
                        enable_lora_bool.append(True)
                    else:
                        enable_lora_bool.append(False)
                self.params_with_lora.update({'in_proj_weight': lora_name})
                self.register_lora_param()
                nn.MultiheadAttention._reset_parameters(self)
                self.init_lora_param_qkv(enable_lora_bool)

    def init_lora_param_qkv(self, enable_lora_bool):
        lora_name = self.params_with_lora['in_proj_weight']
        nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))
        dim = int(self.in_proj_weight.size()[1] / 3)
        for idx, enable in zip(range(3), enable_lora_bool):
            if enable:
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A')[:,idx*dim:(idx+1)*dim], a=math.sqrt(5))
            else:
                nn.init.zeros_(eval(f'self.{lora_name}_lora_A')[:,idx*dim:(idx+1)*dim])

    def train(self, mode: bool = True):
        nn.MultiheadAttention.train(self, mode)
        self.lora_train(mode)     

    def forward(self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.MultiheadAttention.forward(self, query, key, value, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.MultiheadAttention.forward(self, query, key, value, **kwargs)

class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha)

        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0 and any(enable_lora):
            self.w_lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.w_lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        nn.Linear.reset_parameters(self)
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        delta_w = F.conv1d(
            eval(f'self.{lora_name}_lora_A').unsqueeze(0), 
            eval(f'self.{lora_name}_lora_B').unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return self.transpose(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_train(mode)        

    def forward(self, x: torch.Tensor, **kwargs):

        if self.r > 0 and not self.merged:
            self.merge_lora_param()
            result = nn.Linear.forward(self, x, **kwargs)
            self.sub_lora_data()
            return result
        else:
            return nn.Linear.forward(self, x, **kwargs)


def printing():
    print(';')