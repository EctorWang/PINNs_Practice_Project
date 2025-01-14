import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import xlrd
from matplotlib import rcParams

# PyTorch Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)  # Last layer without activation 
        return x


# PINN Model Class
class PINN_model:
    def __init__(self, layers, X, q, X_uc, u_c, X_ac, a_c):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.q = torch.tensor(q, dtype=torch.float32).to(self.device)
        self.X_uc = torch.tensor(X_uc, dtype=torch.float32).to(self.device)
        self.u_c = torch.tensor(u_c, dtype=torch.float32).to(self.device)
        self.X_ac = torch.tensor(X_ac, dtype=torch.float32).to(self.device)
        self.a_c = torch.tensor(a_c, dtype=torch.float32).to(self.device)
        # self.X_kc = torch.tensor(X_kc, dtype=torch.float32).to(self.device)
        # self.k_c = torch.tensor(k_c, dtype=torch.float32).to(self.device)

        # Initialize networks
        self.net_u = NeuralNet(layers[0]).to(self.device)
        self.net_a = NeuralNet(layers[1]).to(self.device)
        # self.net_k = NeuralNet(layers[2]).to(self.device)
        # self.net_Q = NeuralNet(layers[3]).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.get_params(), lr=1e-3)
        self.EI = 1.0  # Material property

        # Loss Logs
        self.loss_log = []
        self.loss_c_log = []
        self.loss_f_log = []

    def get_params(self):
        return list(self.net_u.parameters()) + list(self.net_a.parameters()) 
                    # + list(self.net_k.parameters()) + list(self.net_Q.parameters())

    def net_f(self, x):
        x.requires_grad = True
        # Q = self.net_Q(x)  # 剪力
        # k = self.net_k(x)  # 曲率，是为了后面计算弯矩
        a = self.net_a(x)  # 角度
        u = self.net_u(x)  # 位移
        
        # 计算弯矩
        # M = self.EI * k

        # Gradients
        # Q_x = torch.autograd.grad(Q, x, grad_outputs=torch.ones_like(Q), create_graph=True, retain_graph=True)[0]
        # M_x = torch.autograd.grad(M, x, grad_outputs=torch.ones_like(M), create_graph=True, retain_graph=True)[0]
        a_x = torch.autograd.grad(a, x, grad_outputs=torch.ones_like(a), create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        # f_Q_q = Q_x + self.q
        # f_M_Q = M_x - Q
        # f_a_k = a_x + k
        f_u_k = u_x - a

        return f_u_k

    def train(self, nIter=10000):
        for it in range(nIter):
            self.optimizer.zero_grad()
            
            # Compute losses
            u_c_pred = self.net_u(self.X_uc)
            a_c_pred = self.net_a(self.X_ac)
            # k_c_pred = self.net_k(self.X_kc)

            f_u_k = self.net_f(self.X)

            loss_c = torch.mean((u_c_pred - self.u_c) ** 2) # + torch.mean((k_c_pred - self.k_c) ** 2)
            loss_f = torch.mean(f_u_k ** 2) # + torch.mean(f_Q_q ** 2) + torch.mean(f_M_Q ** 2) + torch.mean(f_a_k ** 2)
            loss = loss_c + loss_f

            loss.backward()
            self.optimizer.step()

            # Logging
            self.loss_log.append(loss.item())
            self.loss_c_log.append(loss_c.item())
            self.loss_f_log.append(loss_f.item())

            if it % 100 == 0:
                print(f"Iter {it}, Loss_c: {loss_c:.3e}, Loss_f: {loss_f:.3e}, Total Loss: {loss:.3e}")

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            u = self.net_u(x).cpu().numpy()
            a = self.net_a(x).cpu().numpy()
            # k = self.net_k(x).cpu().numpy()
            # Q = self.net_Q(x).cpu().numpy()
        return a, u


if __name__ == '__main__':
    layers = [[1] + 3 * [10] + [1], [1] + 3 * [10] + [1], [1] + 3 * [10] + [1], [1] + 3 * [10] + [1]]

    # Generate data using the formula EI*d^2v1/dx^2 = -P(v0 + v1) + (M1 + M2)/L*x - M1
    EI = 2.1e11  # Elastic modulus * moment of inertia (N*m^2)
    P = 1000.0   # Axial force (N) 
    M1 = 1000.0  # Moment at x=0 (N*m)
    M2 = 1000.0  # Moment at x=L (N*m)
    L = 1.0      # Beam length (m)
    v0 = 0.001   # Initial imperfection (m)

    X_star = np.linspace(0, L, 1001).reshape(-1, 1)

    # result v1
    k = np.sqrt(P / EI)
    q = P * L**2 / (np.pi**2 * EI)

    # 计算 v1, theta 和 k 的向量化版本
    x = X_star[:, 0]  # 提取 x 值
    sin_kL = np.sin(k * L)
    sin_kx = np.sin(k * x)
    sin_pi_x = np.sin(np.pi * x)
    
    v1 = M1 / P * (np.sin(k * (L - x)) / sin_kL - (L - x) / L) - \
           M2 / P * (sin_kx / sin_kL - x / L) + \
           q / (1 - q) * sin_pi_x * v0
    
        # 计算 v1 的一阶导数 (theta)
    theta = np.gradient(v1, x)  # v1 关于 x 的一阶导数

    X_train = x.reshape(-1, 1)

    u_fem = v1  # Displacement
    a_fem = theta  # Angle
    # k_fem = k  # Curvature

    # Plot training points
    plt.figure(figsize=(10, 1), dpi=300)
    plt.plot([0, 1], [0, 0], '^', markersize=10, label='Boundary')
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.7, 2.0)
    plt.plot(X_train, np.zeros(X_train.shape), 'o', markersize=8, alpha=0.5, label='Random data')
    plt.yticks([])
    plt.xlabel('Location x')
    plt.legend(fontsize=12, ncol=2)
    plt.savefig('beam1_model1_training_points_high_res.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

    q = np.full_like(X_train, 0)
    X_uc = np.array([[0.0], [1.0]])
    u_c = np.array([[u_fem[0]], [u_fem[-1]]])
    X_ac = np.array([[0.0], [1.0]])
    a_c = np.array([[a_fem[0]], [a_fem[-1]]])

    model = PINN_model(layers, X_train, q, X_train, u_fem, X_train, a_fem)
    model.train(1000)

    a_pred, u_pred = model.predict(X_star)

    plt.figure(figsize=(10, 2))
    plt.plot(X_star, u_fem, label='Generated Data', linewidth=2)
    plt.plot(X_star, u_pred, '--', label='PINN', linewidth=2)
    plt.legend()
    plt.savefig('beam1_model1_generated_data_vs_PINN_high_res.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

    # Plot the loss values during training
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_log, label='Total Loss', linewidth=2, color='blue')
    plt.plot(model.loss_c_log, label='Constraint Loss', linewidth=2, color='lightsalmon', alpha=0.5)
    plt.plot(model.loss_f_log, label='Physics Loss', linewidth=2, color='lightgray', alpha=0.5)
    plt.yscale('log')  # Log scale for better visualization of loss reduction
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('beam1_model1_training_losses_high_res.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()
