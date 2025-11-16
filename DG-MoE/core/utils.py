import os
import torch
import numpy as np
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed = 123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        random.seed(random_seed)

    def add(self, s, a, r, s2):
        experience = (s, a, r, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            
    def add_batch(self, sars2_list):
        if self.count + len(sars2_list) < self.buffer_size: 
            self.buffer = self.buffer + sars2_list
            self.count += len(sars2_list)
        else:
            del self.buffer[:len(self.buffer)//2]
            self.buffer = self.buffer + sars2_list
            self.count = len(self.buffer)
        

    def size(self):
        return self.count

    def sample_batch(self, batch_size, length, dim):

        batch = []

        if self.count < batch_size:
            ran_num = np.arange(self.count)
            batch = list(self.buffer)
        else:
            ran_num = np.random.choice(self.count, batch_size, replace = False)
            batch = [self.buffer[i] for i in ran_num]

        s_batch = torch.zeros(size = (batch_size, length, dim)).cuda()
        a_batch = torch.zeros(size = (batch_size, length, dim)).cuda()
        r_batch = torch.zeros(size = (batch_size,length, dim//6)).cuda()
        s2_batch = torch.zeros(size = (batch_size, length, dim)).cuda()
        for idx, b in enumerate(batch):
            s_batch[idx] = b[0]
            a_batch[idx] = b[1]
            r_batch[idx] = b[2]
            s2_batch[idx] = b[3]

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
        
def save_model(save_path, epoch, model, optimizer):
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_best_results(results, best_results, epoch, model, optimizer, ckpt_root, seed, save_best_model):
    if epoch == 1:
        for key, value in results.items():
            best_results[key] = value
    else:
        for key, value in results.items():
            if (key == 'Has0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Has0_F1_score'] = results['Has0_F1_score']

                if save_best_model:
                    key_eval = 'Has0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'second_teacher_best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif (key == 'Non0_acc_2') and (value > best_results[key]):
                best_results[key] = value
                best_results['Non0_F1_score'] = results['Non0_F1_score']

                if save_best_model:
                    key_eval = 'Non0_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'second_teacher_best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            elif key == 'MAE' and value < best_results[key]:
                best_results[key] = value

                if save_best_model:
                    key_eval = 'MAE'
                    ckpt_path = os.path.join(ckpt_root, f'second_teacher_best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_2' and (value > best_results[key]):
                best_results[key] = value
                best_results['F1_score'] = results['F1_score']

                if save_best_model:
                    key_eval = 'Mult_acc_2'
                    ckpt_path = os.path.join(ckpt_root, f'second_teacher_best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)

            elif key == 'Mult_acc_3' or key == 'Mult_acc_5' or key == 'Mult_acc_7' or key == 'Corr':
                if value > best_results[key]:
                    best_results[key] = value

                if save_best_model:
                    key_eval = key
                    ckpt_path = os.path.join(ckpt_root, f'second_teacher_best_{key_eval}_{seed}.pth')
                    save_model(ckpt_path, epoch, model, optimizer)
            
            else:
                pass
    
    return best_results