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
    def __init__(self, layers, X, q, X_uc, u_c, X_ac, a_c, X_kc, k_c, X_u, u, X_a, a, X_k, k):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.q = torch.tensor(q, dtype=torch.float32).to(self.device)
        self.X_uc = torch.tensor(X_uc, dtype=torch.float32).to(self.device)
        self.u_c = torch.tensor(u_c, dtype=torch.float32).to(self.device)
        self.X_ac = torch.tensor(X_ac, dtype=torch.float32).to(self.device)
        self.a_c = torch.tensor(a_c, dtype=torch.float32).to(self.device)
        self.X_kc = torch.tensor(X_kc, dtype=torch.float32).to(self.device)
        self.k_c = torch.tensor(k_c, dtype=torch.float32).to(self.device)

        self.X_u = torch.tensor(X_u, dtype=torch.float32).to(self.device)
        self.u = torch.tensor(u, dtype=torch.float32).to(self.device)
        self.X_a = torch.tensor(X_a, dtype=torch.float32).to(self.device)
        self.a = torch.tensor(a, dtype=torch.float32).to(self.device)
        self.X_k = torch.tensor(X_k, dtype=torch.float32).to(self.device)
        self.k = torch.tensor(k, dtype=torch.float32).to(self.device)

        # Initialize networks
        self.net_u = NeuralNet(layers[0]).to(self.device)
        self.net_a = NeuralNet(layers[1]).to(self.device)
        self.net_k = NeuralNet(layers[2]).to(self.device)
        # self.net_Q = NeuralNet(layers[3]).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.get_params(), lr=1e-3)
        self.EI = 1.0  # Material property

        # Loss Logs
        self.loss_log = []
        self.loss_c_log = []
        self.loss_f_log = []

    def get_params(self):
        return list(self.net_u.parameters()) + list(self.net_a.parameters()) + list(self.net_k.parameters())
               #  + list(self.net_Q.parameters())

    def net_f(self, x):
        x.requires_grad = True
        # Q = self.net_Q(x)  # 剪力
        k = self.net_k(x)  # 曲率，是为了后面计算弯矩
        a = self.net_a(x)  # 角度
        u = self.net_u(x)  # 位移
        
        # 计算弯矩
        M = self.EI * k

        # Gradients
        # Q_x = torch.autograd.grad(Q, x, grad_outputs=torch.ones_like(Q), create_graph=True, retain_graph=True)[0]
        # M_x = torch.autograd.grad(M, x, grad_outputs=torch.ones_like(M), create_graph=True, retain_graph=True)[0]
        a_x = torch.autograd.grad(a, x, grad_outputs=torch.ones_like(a), create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        # f_Q_q = Q_x + self.q
        # f_M_Q = M_x - Q
        f_a_k = a_x - k
        f_u_k = u_x - a

        uc1 = u[0] - self.u[0]
        uc2 = u[-1] - self.u[-1]

        fc_a_k1 = a_x[0] - self.k[0]
        fc_a_k2 = a_x[-1] - self.k[-1]

        fc_u_k1 = u_x[0] - self.a[0]
        fc_u_k2 = u_x[-1] - self.a[-1]

        return uc1, uc2, fc_a_k1, fc_a_k2, fc_u_k1, fc_u_k2, f_a_k, f_u_k

    def train(self, nIter=10000):
        for it in range(nIter):
            self.optimizer.zero_grad()
            
            # Compute losses
            u_c_pred = self.net_u(self.X_uc)
            a_c_pred = self.net_a(self.X_ac)
            k_c_pred = self.net_k(self.X_kc)

            uc1, uc2, fc_a_k1, fc_a_k2, fc_u_k1, fc_u_k2, f_a_k, f_u_k = self.net_f(self.X)

            loss_c = torch.mean((u_c_pred - self.u_c) ** 2) + torch.mean((a_c_pred - self.a_c) ** 2) + torch.mean((k_c_pred - self.k_c) ** 2)
            loss_f = torch.mean(f_u_k ** 2) + torch.mean(f_a_k ** 2) + \
                     1 * torch.mean(uc1 ** 2) + torch.mean(uc2 ** 2) + \
                     1 * torch.mean(fc_u_k1 ** 2) + torch.mean(fc_a_k1 ** 2) + \
                     1 * torch.mean(fc_u_k2 ** 2) + torch.mean(fc_a_k2 ** 2)
                    #  torch.mean(uc1 ** 2) + torch.mean(uc2 ** 2)
                    # 0.1 * torch.mean(fc_u_k1 ** 2) + torch.mean(fc_a_k1 ** 2) + \
                    # 0.1 * torch.mean(fc_u_k2 ** 2) + torch.mean(fc_a_k2 ** 2)
            loss = loss_c + loss_f

            loss.backward()
            self.optimizer.step()

            # Logging
            self.loss_log.append(loss.item())
            self.loss_c_log.append(loss_c.item())
            self.loss_f_log.append(loss_f.item())

            if it % 100 == 0:
                print(f"Iter {it}, Loss_c: {loss_c:.3e}, Loss_f: {loss_f:.3e}, Total Loss: {loss:.3e}")

        # Print final loss after training
        print(f"Final Loss: {loss:.3e}")

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            u = self.net_u(x).cpu().numpy()
            a = self.net_a(x).cpu().numpy()
            # k = self.net_k(x).cpu().numpy()
            # Q = self.net_Q(x).cpu().numpy()
        return a, u


if __name__ == '__main__':
    layers = [[1] + 3 * [30] + [1], [1] + 3 * [30] + [1], [1] + 3 * [30] + [1], [1] + 3 * [30] + [1]]

    # Generate data using the formula EI*d^2v1/dx^2 = -P(v0 + v1) + (M1 + M2)/L*x - M1
    # EI = 2.1e11  # Elastic modulus * moment of inertia (N*m^2)
    EI = 2.1
    P = 1.0   # Axial force (N)
    M1 = 1.0  # Moment at x=0 (N*m)
    M2 = 1.0  # Moment at x=L (N*m)
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
    
    v1 = M1 / P * (np.sin(k * (L - x)) / np.sin(k * L) - (L - x) / L) - \
           M2 / P * (np.sin(k * x) / np.sin(k * L) - x / L) + \
           q / (1 - q) * np.sin(np.pi * x) * v0

    # Update theta to be the first derivative of v1 with respect to x
    # theta = M1 / P * (1 / L - np.cos(k * (L - x)) / sin_kL) + \
    #         M2 / P * (1 / L - k * np.cos(k * x) / sin_kL) + \
    #         np.pi * q / (L - L * q) * np.cos(np.pi * x / L) * v0
    theta = np.gradient(v1, x)  # First derivative of v1

    k = -M1 * np.sin(k * (L - x)) / (EI * sin_kL) + \
         M2 * sin_kx / (EI * sin_kL) - \
         q * np.pi**2 * sin_pi_x * v0 / (L**2 * (1 - q))
    
    Fq = np.gradient(k, x)
    
    u_fem = v1  # Displacement
    a_fem = theta  # Angle
    k_fem = k  # Curvature

    N_f = 500
    idx = np.random.choice(X_star.shape[0], N_f, replace=False)
    X_train = X_star[idx, :]

    # 绘制 v1, theta 和 k 的数值图像
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    # plt.plot(x, v1, label='v1', color='blue')
    plt.plot(x, u_fem, label='v1', color='blue')
    plt.title('Displacement v1')
    plt.xlabel('x')
    plt.ylabel('v1')
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(x, theta, label='Theta', color='orange')
    plt.title('Angle Theta')
    plt.xlabel('x')
    plt.ylabel('Theta')
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(x, k, label='Curvature k', color='green')
    plt.title('Curvature k')
    plt.xlabel('x')
    plt.ylabel('k')
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(x, Fq, label='q', color='red')
    plt.title('Fq')
    plt.xlabel('x')
    plt.ylabel('Fq')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('beam1_model1_v1_theta_k_high_res.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

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

    q = np.full_like(X_train, (M1+M2)/(L))
    X_uc = np.array([[0.0], [L]])
    u_c = np.array([[u_fem[0]], [u_fem[-1]]])
    X_ac = np.array([[0.0], [L]])
    a_c = np.array([a_fem[0]], [a_fem[-1]])
    X_kc = np.array([[0.0], [L]])
    k_c = np.array([[k_fem[0]], [k_fem[-1]]])

    model = PINN_model(layers, X_star, q, X_star, u_fem, X_star, a_fem, X_star, k_fem,
                                           X_uc, u_c, X_ac, a_c, X_kc, k_c)
    model.train(2000)

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
