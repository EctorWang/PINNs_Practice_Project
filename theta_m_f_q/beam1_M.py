import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class BeamDeflection:
    def __init__(self, M_e=1000.0, E=2e11, I=8.33e-9, L=0.1):
        """
        初始化梁的参数
        参数:
        M_e: 力矩，单位：N·m
        E: 弹性模量，单位：Pa
        I: 截面惯性矩，单位：m^4
        L: 梁的长度，单位：m
        """
        self.M_e = M_e
        self.E = E
        self.I = I
        self.L = L
        
    def calculate_w(self, x):
        """
        计算 w = -M_e * x^2 / (2 * E * I)
        参数:
        x: 位置
        返回:
        w: 挠度值
        """
        w = -self.M_e * x**2 / (2 * self.E * self.I)
        return w
    
    def generate_data(self, points=1001):
        """
        生成x数据点和对应的w值
        参数:
        points: 数据点数量
        返回:
        x, w: 位置和对应的挠度值
        """
        self.x = torch.linspace(0, self.L, points)
        self.w = self.calculate_w(self.x)
        return self.x, self.w

class NeuralNet(nn.Module):
    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)
        return x

class BeamDeflectionPINN:
    def __init__(self, M_e=1000.0, E=2e11, I=8.33e-9, L=0.1):
        """
        初始化PINN模型
        参数:
        M_e: 力矩，单位：N·m
        E: 弹性模量，单位：Pa
        I: 截面惯性矩，单位：m^4
        L: 梁的长度，单位：m
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M_e = M_e
        self.E = E
        self.I = I
        self.L = L
        
        # 定义神经网络层
        layers = [1] + 4 * [20] + [1]  # 输入层1个节点，3个隐藏层各20个节点，输出层1个节点
        self.net = NeuralNet(layers).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
        
        # 损失记录
        self.loss_log = []
        
    def generate_training_data(self, n_points=100):
        """生成训练数据点并进行归一化"""
        # 内部点
        x_collocation = np.linspace(0, self.L, n_points)
        self.x_collocation = torch.tensor(x_collocation, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        # 归一化
        self.x_collocation_normalized = self.x_collocation / self.L
        
        # 生成对应的w值
        self.w_true = -self.M_e * (self.x_collocation ** 2) / (2 * self.E * self.I)
        
        # 边界点
        self.x_boundary = torch.tensor([[0.0], [self.L]], dtype=torch.float32, requires_grad=True).to(self.device)
        
    def compute_loss(self):
        """计算损失函数"""
        # 使用归一化的内部点
        x_c = self.x_collocation_normalized.clone().requires_grad_(True).to(self.device)
        w_pred_normalized = self.net(x_c)
        
        # 反归一化预测值
        w_pred = w_pred_normalized * self.L
        
        # 计算一阶导数（转角）
        dw_dx = torch.autograd.grad(w_pred, x_c, 
                                  grad_outputs=torch.ones_like(w_pred),
                                  create_graph=True)[0]
        
        # 计算二阶导数
        d2w_dx2 = torch.autograd.grad(dw_dx, x_c,
                                    grad_outputs=torch.ones_like(dw_dx),
                                    create_graph=True)[0]
        
        # 物理方程损失: EI * d2w/dx2 + M = 0
        f = self.E * self.I * d2w_dx2 + self.M_e
        physics_loss = torch.mean(f ** 2)
        
        # 边界条件损失
        x_b = self.x_boundary.clone().requires_grad_(True).to(self.device)
        w_boundary = self.net(x_b / self.L) * self.L  # 归一化输入，反归一化输出
        dw_dx_boundary = torch.autograd.grad(w_boundary, x_b, 
                                             grad_outputs=torch.ones_like(w_boundary),
                                             create_graph=True)[0]
        
        # w(0) = 0 和 dw/dx(0) = 0
        bc_loss = torch.mean(w_boundary ** 2) + torch.mean(dw_dx_boundary[0] ** 2)  # w(0) = 0, dw/dx(0) = 0
        
        # 位移损失
        displacement_loss = torch.mean((w_pred - self.w_true) ** 2)
        
        # 总损失
        total_loss = physics_loss + bc_loss + displacement_loss
        return total_loss, physics_loss, displacement_loss
        
    def train(self, epochs=10000):
        """训练模型"""
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss, physics_loss, displacement_loss = self.compute_loss()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            self.loss_log.append(total_loss.item())
            
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Total Loss: {total_loss.item():.3e}, Physics Loss: {physics_loss.item():.3e}, Displacement Loss: {displacement_loss.item():.3e}')
                
    def predict(self, x):
        """预测给定位置的挠度"""
        x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(self.device)
        x_normalized = x_tensor / self.L
        with torch.no_grad():
            w_pred_normalized = self.net(x_normalized)
            return (w_pred_normalized * self.L).cpu().numpy()

def main():
    # 创建解析解和PINN实例
    beam_analytical = BeamDeflection(L=0.2)
    beam_pinn = BeamDeflectionPINN(L=0.2)
    
    # 生成训练数据
    beam_pinn.generate_training_data(n_points=100)
    
    # 训练模型
    beam_pinn.train(epochs=30000)  # 增加训练轮次
    
    # 生成预测点
    x_plot = np.linspace(0, 0.2, 1001)
    
    # 获取解析解
    x_analytical, w_analytical = beam_analytical.generate_data()
    
    # 获取PINN预测值
    w_pinn = beam_pinn.predict(x_plot)
    
    # 绘制结果对比
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x_analytical.numpy(), w_analytical.numpy(), 'b-', label='解析解')
    plt.plot(x_plot, w_pinn, 'r--', label='PINN预测')
    plt.xlabel('x (m)')
    plt.ylabel('w (m)')
    plt.title('梁的挠度曲线对比')
    plt.grid(True)
    plt.legend()
    
    # 绘制损失曲线
    plt.subplot(2, 1, 2)
    plt.plot(beam_pinn.loss_log)
    plt.yscale('log')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
