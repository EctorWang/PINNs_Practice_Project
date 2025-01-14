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
    def __init__(self, layers, X, q, X_uc, u_c, X_ac, a_c, X_kc, k_c):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.q = torch.tensor(q, dtype=torch.float32).to(self.device)
        self.X_uc = torch.tensor(X_uc, dtype=torch.float32).to(self.device)
        self.u_c = torch.tensor(u_c, dtype=torch.float32).to(self.device)
        self.X_ac = torch.tensor(X_ac, dtype=torch.float32).to(self.device)
        self.a_c = torch.tensor(a_c, dtype=torch.float32).to(self.device)
        self.X_kc = torch.tensor(X_kc, dtype=torch.float32).to(self.device)
        self.k_c = torch.tensor(k_c, dtype=torch.float32).to(self.device)

        # Initialize networks
        self.net_u = NeuralNet(layers[0]).to(self.device)
        self.net_a = NeuralNet(layers[1]).to(self.device)
        self.net_k = NeuralNet(layers[2]).to(self.device)
        self.net_Q = NeuralNet(layers[3]).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.get_params(), lr=1e-3)
        self.EI = 1.0  # Material property

        # Loss Logs
        self.loss_log = []
        self.loss_c_log = []
        self.loss_f_log = []

    def get_params(self):
        return list(self.net_u.parameters()) + list(self.net_a.parameters()) + \
               list(self.net_k.parameters()) + list(self.net_Q.parameters())

    def net_f(self, x):
        x.requires_grad = True
        Q = self.net_Q(x)  # 剪力
        k = self.net_k(x)  # 曲率，是为了后面计算弯矩
        a = self.net_a(x)  # 角度
        u = self.net_u(x)  # 位移
        
        # 计算弯矩
        M = self.EI * k

        # Gradients
        Q_x = torch.autograd.grad(Q, x, grad_outputs=torch.ones_like(Q), create_graph=True, retain_graph=True)[0]
        M_x = torch.autograd.grad(M, x, grad_outputs=torch.ones_like(M), create_graph=True, retain_graph=True)[0]
        a_x = torch.autograd.grad(a, x, grad_outputs=torch.ones_like(a), create_graph=True, retain_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        f_Q_q = Q_x + self.q
        f_M_Q = M_x - Q
        f_a_k = a_x + k
        f_u_k = u_x - a

        return f_Q_q, f_M_Q, f_a_k, f_u_k

    def train(self, nIter=10000):
        for it in range(nIter):
            self.optimizer.zero_grad()
            
            # Compute losses
            u_c_pred = self.net_u(self.X_uc)
            a_c_pred = self.net_a(self.X_ac)
            k_c_pred = self.net_k(self.X_kc)

            f_Q_q, f_M_Q, f_a_k, f_u_k = self.net_f(self.X)

            loss_c = torch.mean((u_c_pred - self.u_c) ** 2) + torch.mean((k_c_pred - self.k_c) ** 2)
            loss_f = torch.mean(f_Q_q ** 2) + torch.mean(f_M_Q ** 2) + torch.mean(f_a_k ** 2) + torch.mean(f_u_k ** 2)
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
            k = self.net_k(x).cpu().numpy()
            Q = self.net_Q(x).cpu().numpy()
        return Q, k, a, u


if __name__ == '__main__':
    layers = [[1] + 3 * [10] + [1], [1] + 3 * [10] + [1], [1] + 3 * [10] + [1], [1] + 3 * [10] + [1]]

    # Define constants
    q = 1.0  # Example value for distributed load
    EI = 1.0  # Example value for EI
    l = 1.0  # Example value for length
    M_e = 1.0  # Example value for M_e
    a_rate = 0.5  # Example value for a_rate
    a = l * a_rate  # Example value for a
    b = l * (1 - a_rate)  # Example value for b

    # Generate data using the new formula
    X_star = np.linspace(0, l, 1001).reshape(-1, 1)
    u_fem = np.zeros_like(X_star)
    a_fem = np.zeros_like(X_star)
    k_fem = np.zeros_like(X_star)

    for i, x in enumerate(X_star):
        if x <= a:
            u_fem[i] = M_e * x / (6 * EI * l) * (l**2 - 3 * b**2 - x**2)
            a_fem[i] = M_e * (l**2 - 3 * b**2 - 3 * x**2) / (6 * EI * l)
            k_fem[i] = M_e * x / (EI * l)

        else:
            u_fem[i] = M_e / (6 * EI * l) * (-x**3 + 3 * l * (x - a)**2 + (l**2 - 3 * b**2) * x)
            a_fem[i] = M_e * (-3 * x**2 + 6 * l * (x - a) + (l**2 - 3 * b**2)) / (6 * EI * l)
            k_fem[i] = M_e * (-x + l) / (EI * l)

    # Plot the computed displacement u_fem against the position X_star
    plt.figure(figsize=(10, 4))
    plt.plot(X_star, u_fem, label='Computed Displacement $u_{fem}$', linewidth=2, color='blue')
    plt.xlabel('Position $x$')
    plt.ylabel('Displacement $u_{fem}$')
    plt.title('Computed Displacement vs. Position')
    plt.legend()
    plt.grid(True)
    plt.savefig('beam2_model2_computed_displacement_vs_position.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()


    # Plot the computed Corner a_fem against the position X_star
    plt.figure(figsize=(10, 4))
    plt.plot(X_star, a_fem, label='Computed Displacement $u_{fem}$', linewidth=2, color='blue')
    plt.xlabel('Position $x$')
    plt.ylabel('Corner $a_{fem}$')
    plt.title('Computed Corner vs. Position')
    plt.legend()
    plt.grid(True)
    plt.savefig('beam2_model2_computed_Corner_vs_position.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

    # Plot the computed Curvature u_fem against the position X_star
    plt.figure(figsize=(10, 4))
    plt.plot(X_star, k_fem, label='Computed Curvature $u_{fem}$', linewidth=2, color='blue')
    plt.xlabel('Position $x$')
    plt.ylabel('Curvature $k_{fem}$')
    plt.title('Computed Curvature vs. Position')
    plt.legend()
    plt.grid(True)
    plt.savefig('beam2_model2_computed_Curvature_vs_position.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()


    N_f = 50
    idx = np.random.choice(X_star.shape[0], N_f, replace=False)
    X_train = X_star[idx, :]

    # Plot training points
    plt.figure(figsize=(10, 1), dpi=300)
    plt.plot([0, l], [0, 0], '^', markersize=10, label='Boundary')
    plt.xlim(-0.05, l + 0.05)
    plt.ylim(-0.7, 2.0)
    plt.plot(X_train, np.zeros(X_train.shape), 'o', markersize=8, alpha=0.5, label='Random data')
    plt.yticks([])
    plt.xlabel('Location x')
    plt.legend(fontsize=12, ncol=2)
    plt.savefig('beam2_model2_training_points_high_res.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

    # q = np.full_like(X_train, 0)
    q_fem = np.zeros_like(X_train)
    for i, x in enumerate(X_train):
        if x <= a:
            q_fem[i] = -M_e / (l)

        else:
            q_fem[i] = -M_e / (l)
    q = - q_fem
    X_uc = np.array([[0.0], [l]])
    u_c = np.array([[u_fem[0]], [u_fem[-1]]])  # Ensure correct shape
    X_ac = np.array([[0.0], [l]])
    a_c = np.array([[a_fem[0]], [a_fem[-1]]])  # Ensure correct shape
    X_kc = np.array([[0.0], [l]])
    k_c = np.array([[-k_fem[0]], [-k_fem[-1]]])  # Ensure correct shape

    model = PINN_model(layers, X_train, q, X_uc, u_c, X_ac, a_c, X_kc, k_c)
    model.train(8000)

    Q_pred, k_pred, a_pred, u_pred = model.predict(X_star)

    plt.figure(figsize=(10, 2))
    plt.plot(X_star, u_fem, label='Generated Data', linewidth=2)
    plt.plot(X_star, u_pred, '--', label='PINN', linewidth=2)
    plt.legend()
    plt.savefig('beam2_model2_generated_data_vs_PINN_high_res.png', dpi=300, bbox_inches='tight')  # Save the plot
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
    plt.savefig('beam2_model2_training_losses_high_res.png', dpi=300, bbox_inches='tight')  # Save the plot
    plt.show()

