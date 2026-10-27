import numpy as np  
import os  
import torch
import json
from torch.utils.data import Dataset, DataLoader  

  
# 自定义Dataset类  
class SkeletonDataset(Dataset):  
    def __init__(self, root_dir, num_class, mode):  
        self.data = []  
        self.labels = []  
        self.root_dir = root_dir  
        self.num_class = num_class
        self.load_data(mode) 
        # 计算均值和标准差  
        self.mean = self.data.mean(dim=0, keepdim=True)  # 沿着样本维度计算均值，保持维度一致  
        self.std = self.data.std(dim=0, keepdim=True)    # 沿着样本维度计算标准差，保持维度一致  
        # 避免除以零，给标准差一个小的正数底限  
        self.std[self.std == 0] = 1e-6  
        # 标准化数据  
        self.data = (self.data - self.mean) / self.std
    
    def process_and_append_subset(self, subset, start_index, subsets_with_indices, block_size):  
        # 如果子集长度大于block_size，则分块处理  
        if len(subset) > block_size:  
            for i in range(0, len(subset), block_size):  
                block = subset[i:i+block_size]  
                # 如果块不足block_size个元素，则往前补足 
                need_length = 0 
                if len(block) < block_size:  
                    need_length = block_size - len(block)
                    block.extend(subset[i-need_length:i])  
                # 添加块和起始下标到结果列表中  
                subsets_with_indices.append((block, i - need_length))  
        # 如果子集长度小于block_size，则跳过  
        elif len(subset) < block_size:  
            pass  
        # 如果子集长度等于block_size，则直接添加  
        else:  
            subsets_with_indices.append((subset, start_index))
            
    def split_subset(self, samples_label, block_size=50):  
        # 初始化当前子集、起始下标和所有子集列表  
        current_subset = []  
        start_index = 0  
        subsets_with_indices = []  
        
        # 遍历samples_label列表  
        for i, label in enumerate(samples_label):  
            # 如果当前子集为空或者当前标签与当前子集最后一个标签不同  
            if not current_subset or label != current_subset[-1]:  
                # 如果当前子集不为空，处理并重置当前子集  
                if current_subset:  
                    self.process_and_append_subset(current_subset, start_index, subsets_with_indices, block_size)  
                # 开始新的子集，并更新起始下标  
                current_subset = [label]  
                start_index = i  
            else:  
                # 如果当前标签与当前子集最后一个标签相同，则添加到当前子集中  
                current_subset.append(label)  
        
        # 处理并添加最后一个子集（如果它不为空的话）  
        if current_subset:  
            self.process_and_append_subset(current_subset, start_index, subsets_with_indices, block_size)  
        
        return subsets_with_indices
    
    def sliding_window_3d(self, data, window_size, step_size, axis=0):  
        """  
        对3D NumPy数组使用滑动窗口。  
        parameter:
            data (np.ndarray): 3D NumPy数组。  
            window_size (int): 窗口大小。  
            step_size (int): 步长。  
            axis (int): 沿着哪个轴移动窗口(默认为0)。  
            
        return:list of np.ndarray: 包含滑动窗口提取的子数组的列表。  
        """  
        # 确保步长不大于窗口大小且窗口大小不超过数组在该轴上的长度  
        if step_size > window_size or window_size > data.shape[axis]:  
            raise ValueError("Invalid window or step size.")  
        
        # 初始化结果列表  
        windows = []  
        # 初始化起始索引  
        start_idx = 0  
        # 当起始索引加上窗口大小不超过数据长度时，继续循环  
        while start_idx + window_size <= data.shape[axis]:  
            # 使用np.take沿着指定轴提取当前窗口的数据  
            window = np.take(data, range(start_idx, start_idx + window_size), axis=axis)  
            # 将窗口添加到结果列表中    
            skeletons_4d = np.expand_dims(window, axis=0)  # (1, t, v, c) 
            windows.append(skeletons_4d)
            # 更新起始索引以进行下一次迭代  
            start_idx += step_size  
        
        return windows 
    
    def load_data(self, mode=1):  
        block_size = 50
        for root, dirs, files in os.walk(self.root_dir):  
            for file in files:  
                if file.endswith('.npy'):  
                    file_path = os.path.join(root, file) 
                    filename_with_ext = os.path.basename(file_path)  
                    filename_without_ext = os.path.splitext(filename_with_ext)[0]
                    parent_dir = os.path.basename(os.path.dirname(file_path)) 
                    data_list = np.load(file_path, allow_pickle=True).tolist()  
                    samples_skeleton = []  # 用于存储一个文件的所有skeleton数据  
                    samples_label = []  # 用于存储一个文件的所有标签数据  
  
                    for item in data_list:  
                        skeleton_data = item['skleton']  
                        label = [item[key] for key in item if key != 'skleton']  
                        samples_skeleton.append(skeleton_data)  
                        samples_label.append(label)  

                    if len(samples_skeleton) < block_size:  
                        continue  # 跳过长度小于block_size的samples_skeleton
                    
                    if mode == 1:
                        # 将samples_skeleton中的多个二维数组合并成一个三维数组  
                        # 假设每个skeleton_data的形状为(v, c)，且所有skeleton_data的v和c相同  
                        skeletons_3d = np.stack(samples_skeleton, axis=0)  # (t, v, c)  
    
                        # 每block_size个数据作为一个块处理 
                        subsets = self.split_subset(samples_label,block_size) 
                        for subset in subsets: 
                            # 取出当前块的数据  
                            block_skeletons_3d = skeletons_3d[subset[1]:subset[1]+block_size]  
                            # 检查是否达到最后一个不完整的块  
                            if block_skeletons_3d.shape[0] < block_size:  
                                continue  # 跳过不完整的块  
                            # 将三维数组扩展为四维数组，以便后续转换形状  
                            skeletons_4d = np.expand_dims(block_skeletons_3d, axis=0)  # (1, t, v, c)  
                            # 将数据添加到self.data中  
                            self.data.append(skeletons_4d)  # (n, 1, t, v, c)  
                            # 添加所有样本的标签到self.labels 
                            s_label =  subset[0][0]
                            self.labels.append(s_label) 
                    elif mode == 2:
                        if not os.path.exists(f'data/intention_data/{parent_dir}/intention_data.json'):  
                            continue   
                        try:  
                            with open(f'data/intention_data/{parent_dir}/intention_data.json', 'r') as _file:  
                                intentions = json.load(_file) 
                                for _item in intentions:  
                                    # 检查字典中是否包含键  
                                    if filename_without_ext in _item:  
                                        _value = _item[filename_without_ext]  
                                        #self.labels.append(_value) 
                                    else:  
                                        continue
                        except json.JSONDecodeError as e:  
                            continue 
                        except Exception as e:  
                            print(f"读取文件 {f'data/intention_data/{parent_dir}/intention_data.json'} 时发生错误: {e}")
                        skeletons_3d = np.stack(samples_skeleton, axis=0)  # (t, v, c)
                        
                        window_3d = self.sliding_window_3d(skeletons_3d, 50, 10, axis=0)
                        for i in window_3d:
                            self.data.append(i)
                            self.labels.append(_value)
        
        self.labels = np.array(self.labels)         
        if mode == 1:
            labels_first_column = self.labels[:, 1]
            one_hot_labels = np.eye(self.num_class)[labels_first_column]
            self.labels = np.array(one_hot_labels)
            self.labels = torch.from_numpy(self.labels.astype(np.float32)) 
        elif mode == 2:
            # 映射标签值到 [0, num_classes-1] 的范围内 
            _label =  self.labels
            mapped_labels = np.where(_label == -1, 0,   
                                    np.where(_label == 0, 1, 2)) 
            one_hot_labels = np.eye(self.num_class)[mapped_labels]
            self.labels = np.array(one_hot_labels)
            self.labels = torch.from_numpy(self.labels.astype(np.float32)) 
        self.data = np.array(self.data)
        self.data = torch.from_numpy(self.data.astype(np.float32))  
        self.data = self.data.permute(0, 4, 2, 3, 1)  # 得到(n, c, t, v, 1)的形状
  
    def __len__(self):  
        return len(self.data)  
  
    def __getitem__(self, idx):  
        return self.data[idx], self.labels[idx]  
  
class LSTM_SkeletonDataset(Dataset):  
    def __init__(self, root_dir):  
        self.all_inputs = []  
        self.all_outputs = [] 
        self.mean = None  
        self.std = None  
        self.root_dir = root_dir  
        self.load_and_process_data()
        self.to_torch_tensors()
        
    def process_npy_file(self, file_path):  
        samples_skeleton = np.load(file_path, allow_pickle=True).tolist() 
        data_list = [] 
        # 假设每个文件包含足够多的帧以形成多个(50, 10)的窗口  
        all_inputs = []  
        all_outputs = []  
        for item in samples_skeleton:  
            skeleton_data = item['skleton']  
            data_list.append(skeleton_data)  
        
        data_list = [(data - self.mean) / self.std for data in data_list]
        for i in range(0, len(data_list) - 50 - 10 + 1, 10):  # 滑动窗口，步长为10  
            inputs = np.stack(data_list[i:i+50], axis=0)  # 取前50帧  
            outputs = np.stack(data_list[i+50:i+60], axis=0)  # 取后10帧作为目标  
            all_inputs.append(inputs)  
            all_outputs.append(outputs)  
        
        return all_inputs, all_outputs  
    
    # 加载所有.npy文件并处理数据  
    def load_and_process_data(self):  
        if self.mean is None or self.std is None:  
            all_data = []  
            for root, dirs, files in os.walk(self.root_dir):  
                for file in files:  
                    if file.endswith('.npy'):  
                        file_path = os.path.join(root, file)  
                        samples_skeleton = np.load(file_path, allow_pickle=True).tolist()  
                        all_data.extend([item['skleton'] for item in samples_skeleton])  
  
            # 计算均值和标准差  
            self.mean = np.mean(all_data, axis=0)  
            self.std = np.std(all_data, axis=0) + 1e-8  # 加上一个小值以避免除以零
        for root, dirs, files in os.walk(self.root_dir):  
            for file in files:  
                if file.endswith('.npy'):  
                    file_path = os.path.join(root, file)  
                    inputs, outputs = self.process_npy_file(file_path)  
                    self.all_inputs.extend(inputs)  
                    self.all_outputs.extend(outputs)  
    
    # 将数据转换为PyTorch张量  
    def to_torch_tensors(self):  
        self.all_inputs = torch.tensor(self.all_inputs, dtype=torch.float32)  
        self.all_outputs = torch.tensor(self.all_outputs, dtype=torch.float32)  
    
    def __len__(self):  
        return len(self.all_inputs)  
  
    def __getitem__(self, idx): 
        input_data = self.all_inputs[idx]  # 形状为 [50, 17, 3]  
        output_data = self.all_outputs[idx]  # 形状为 [10, 17, 3]  
        # 将后两维合并为一维  
        input_data = input_data.view(input_data.size(0), -1)  # 形状变为 [50, 51]  
        output_data = output_data.view(output_data.size(0), -1)  # 形状变为 [10, 51] 
        return input_data, output_data