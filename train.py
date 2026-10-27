import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import random_split 
from torch.utils.data import DataLoader  
import matplotlib.pyplot as plt
from st_gcn import *
from DataSet import *



def stgcn_train(num_epochs, Num_class, save_path, root_dir='data/final_data'): 
    # 实例化数据集和数据加载器  
    num_class = Num_class
    dataset = SkeletonDataset(root_dir,num_class,2)  
    total_dataset_size = len(dataset) 
    # 设定训练集和测试集的比例  
    train_size = int(0.8 * total_dataset_size)  
    test_size = total_dataset_size - train_size
    # 使用random_split进行划分  
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # 为训练集创建DataLoader  
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)  
    # 为测试集创建DataLoader  
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    graph_args={
        "layout": 'yolov8',
        "strategy": 'spatial'}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型  
    model = Model(3,num_class,graph_args,True).to(device) 
    # 定义损失函数和优化器  
    criterion = nn.CrossEntropyLoss()  # 分类任务
    optimizer = optim.SGD(model.parameters(), lr=0.001) 
    #加载模型训练
    #model_state_dict = torch.load('Models/model_with_skeleton_cpu.pth')  
    #model.load_state_dict(model_state_dict) 
 
    loss_history = [] 
    accuracy_history = [] 
    val_loss_history = [] 
    val_accuracy_history = [] 
    for epoch in range(num_epochs):  
        print('---------------------------')  
        model.train()  # 设置模型为训练模式  
        
        # 在每个epoch开始时重置正确预测的数量  
        correct_predictions = 0  
        total_samples = 0  
        total_loss = 0
        
        for i, (data, labels) in enumerate(train_dataloader):  
            data, labels = data.to(device), labels.to(device)  
            # 前向传播  
            outputs = model(data)  
            # 计算损失  
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # 获取预测类别索引和真实标签索引  
            _, predicted = torch.max(outputs, 1)  
            labels_indices = torch.argmax(labels, dim=1)  
            # 累加正确预测的数量  
            correct_predictions += (predicted == labels_indices).sum().item()  
            # 累加样本总数  
            total_samples += labels_indices.size(0)  
            # 反向传播和优化  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
        
        # 在每个epoch结束时计算准确率  
        accuracy = correct_predictions / total_samples  
        #avg_loss = total_loss / len(train_dataloader) 
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss:.4f}, Train Accuracy: {accuracy:.4f}')   
        loss_history.append(total_loss)  
        accuracy_history.append(accuracy) 

        # 在每个epoch结束时进行验证  
        model.eval()  # 设置模型为评估模式  
        with torch.no_grad():  
            total_val_loss = 0.0  
            total_correct = 0  
            total_val = 0  
            for val_data, val_labels in test_dataloader:  
                val_data, val_labels = val_data.to(device), val_labels.to(device)  
                val_outputs = model(val_data)  
                val_loss = criterion(val_outputs, val_labels)  
                total_val_loss += val_loss.item()  
                _, val_predicted = torch.max(val_outputs, 1)  
                val_labels_indices = torch.argmax(val_labels, dim=1) 
                total_correct += (val_predicted == val_labels_indices).sum().item()  
                total_val += val_labels_indices.size(0)  
            
            #avg_val_loss = total_val_loss / len(test_dataloader) 
            val_loss_history.append(total_val_loss) 
            val_accuracy = total_correct / total_val 
            val_accuracy_history.append(val_accuracy) 
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {total_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}') 
    # 保存模型  
    torch.save(model.state_dict(), save_path)
    return loss_history,accuracy_history,val_loss_history,val_accuracy_history

def show(loss_history, accuracy_history, val_loss_history, val_accuracy_history): 

    # 绘制损失历史图  
    plt.plot(loss_history)  
    plt.xlabel('Epoch')  # 设置x轴标签  
    plt.ylabel('Loss')  # 设置y轴标签  
    plt.title('Loss History')  # 设置图表标题  
    plt.show()  
    
    # 绘制准确率历史图  
    plt.plot(accuracy_history)  
    plt.xlabel('Epoch')  # 设置x轴标签  
    plt.ylabel('Accuracy')  # 设置y轴标签  
    plt.title('Accuracy History')  # 设置图表标题  
    plt.show()
    
    # 绘制损失历史图  
    plt.plot(val_loss_history)  
    plt.xlabel('Epoch')  # 设置x轴标签  
    plt.ylabel('Loss')  # 设置y轴标签  
    plt.title('Validation Loss History')  # 设置图表标题  
    plt.show()  
    
    # 绘制损失准确率历史图  
    plt.plot(val_accuracy_history)  
    plt.xlabel('Epoch')  # 设置x轴标签  
    plt.ylabel('Accuracy')  # 设置y轴标签  
    plt.title('Validation Accuracy History')  # 设置图表标题  
    plt.show()
    
def LSTM_train(num_epochs=100,Root_dir='data/final_data',model_path='Models/skeleton_lstm_model.pth'):
    root_dir = Root_dir
    dataset = LSTM_SkeletonDataset(root_dir)  
    total_dataset_size = len(dataset)
    train_size = int(0.8 * total_dataset_size)  
    test_size = total_dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)  
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 定义超参数  
    Input_size = 51  # 骨架点的特征数  
    Hidden_size = 256  # LSTM隐藏层大小  
    Num_layers = 10  # LSTM层数  
    Output_size = 51  # 骨架点的特征数  
    Output_frames = 10  # 预测帧数，应与标签中的帧数一致 
    
    # 实例化模型  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkeletonLSTM(input_size=Input_size, hidden_size=Hidden_size, num_layers=Num_layers,  
                        output_size=Output_size, output_frames=Output_frames).to(device)
    #state_dict = torch.load('Models/skeleton_lstm_model.pth')  
    #model.load_state_dict(state_dict)
    
    criterion = nn.MSELoss()  
    weight_decay = 1e-4 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    loss_history = [] 
    val_loss_history = [] 
    for epoch in range(num_epochs):  
        print("----------------")
        model.train()  # 设置模型为训练模式  
        running_loss = 0.0  
        for i, (inputs, targets) in enumerate(train_dataloader):  
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), targets.size(1), 17, 3)
            optimizer.zero_grad()  # 清零梯度缓存  
            # 将数据输入模型，得到输出  
            outputs = model(inputs) 
            # 计算损失  
            loss = criterion(outputs, targets) 
            # 反向传播，计算当前梯度  
            loss.backward()  
            # 使用优化器更新权重  
            optimizer.step()  
            # 累加损失值  
            running_loss += loss.item()  
        loss_history.append(running_loss) 
        # 打印每个epoch的损失  
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}') 
        
        model.eval()  # 设置模型为评估模式  
        val_running_loss = 0.0  
        with torch.no_grad():  # 不计算梯度，节省内存和计算资源  
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)  
                targets = targets.view(targets.size(0), targets.size(1), 17, 3)
                outputs = model(inputs)  
                loss = criterion(outputs, targets)  
                val_running_loss += loss.item()  
        
        # 计算验证集上的平均损失   
        val_loss_history.append(val_running_loss) 
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_running_loss:.4f}')  
    torch.save(model.state_dict(), model_path)
    return loss_history,val_loss_history

def run():
    l = LSTM_train()

