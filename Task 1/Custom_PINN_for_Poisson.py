import torch
import torch.nn as nn
import numpy as np

# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda')
    #print('CUDA')
else:
    device = torch.device('cpu')
    #print('CPU')

class Sampler:
    def __init__(self, domain_bounds, func, name=None):
        """
        domain_bounds: список с границами области [[x_min, y_min], [x_max, y_max]]
        func: функция для граничных условий
        name: имя сэмплера
        """
        self.domain_bounds = np.array(domain_bounds)
        self.func = func
        self.name = name
    
    def sample_interior(self, N):
        """ Выборка точек внутри области """
        x_min, y_min = self.domain_bounds[0]
        x_max, y_max = self.domain_bounds[1]
        x = np.random.uniform(x_min, x_max, (N, 1))
        y = np.random.uniform(y_min, y_max, (N, 1))

        interior_points = np.hstack((x, y))
        return interior_points, self.func(interior_points)
    
    def sample_boundary(self, N = 200):
        """ Выборка точек на границе области """
        x_min, y_min = self.domain_bounds[0]
        x_max, y_max = self.domain_bounds[1]
        
        # Разбиваем точки на 4 стороны границы
        N_side = N // 4
        x_left = np.full((N_side, 1), x_min)
        y_left = np.random.uniform(y_min, y_max, (N_side, 1))
        
        x_right = np.full((N_side, 1), x_max)
        y_right = np.random.uniform(y_min, y_max, (N_side, 1))
        
        x_bottom = np.random.uniform(x_min, x_max, (N_side, 1))
        y_bottom = np.full((N_side, 1), y_min)
        
        x_top = np.random.uniform(x_min, x_max, (N_side, 1))
        y_top = np.full((N_side, 1), y_max)
        
        boundary_points = np.vstack((
            np.hstack((x_left, y_left)),
            np.hstack((x_right, y_right)),
            np.hstack((x_bottom, y_bottom)),
            np.hstack((x_top, y_top))
        ))
        
        boundary_values = self.func(boundary_points)
        return boundary_points, boundary_values
    
# Creating Neural Network
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 2):
        super(NN, self).__init__()
        
        # Входной слой
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        nn.init.xavier_uniform_(layers[0].weight)
        layers[0].bias.data.fill_(0.0)
        
        # Скрытые слои
        for _ in range(num_layers):
            layer = nn.Linear(hidden_size, hidden_size)
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0)
            layers.append(layer)
            layers.append(nn.Tanh())
        
        # Выходной слой
        out_layer = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(out_layer.weight)
        out_layer.bias.data.fill_(0.0)
        layers.append(out_layer)
        
        # Объединяем в nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

    
class PINN:
# Initialize the class
    def __init__(self, layers, operator, bcs_sampler, res_sampler, internal_layers_number = 2, learning_r = 0.1):

        # Samplers
        self.operator = operator
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler
        
        # Neural Network
        self.nn = NN(layers[0], layers[1], layers[-1], internal_layers_number).to(device)

        self.optimizer_Adam = torch.optim.Adam(params=self.nn.parameters(), lr=learning_r, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_Adam, gamma=0.9)
        
        # Logger
        self.loss_bcs_log = []
        self.loss_res_log = []
        
        
    # Forward pass for u
    def net_u(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        return self.nn(inputs)

    # Forward pass for the residual
    def net_r(self, x, y):
        u = self.net_u(x, y)
        
        # u_x = self.gradient(u, x)
        # u_xx = self.gradient(u_x, x)
        
        # u_y = self.gradient(u, y)
        # u_yy = self.gradient(u_y, y)
        
        # residual = self.operator(u_xx + u_yy, x, y)

        residual = self.operator(u, x, y)
        return residual
    

    # Gradient operation
    def gradient(self, u, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(u, requires_grad=True)
        grad = torch.autograd.grad(u, [x], grad_outputs = grad_outputs, create_graph=True)[0]
        return grad


    def fetch_minibatch(self, sampler, N):
        if sampler.name == "sample_interior":
            X, Y = sampler.sample_interior(N)
        elif sampler.name == "sample_boundary":
            X, Y = sampler.sample_boundary(N)
        else:
            raise ValueError("Sampler does not have a valid sampling method.")
    
        return X, Y


    # Trains the model by minimizing the MSE loss
    def train(self, nIter=10000, batch_size=1000, save_path = "best_custom.pth"):
        
        # NTK
        self.nn.train()
        best_loss = float('inf')
        
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_bcs_batch, u_bcs_batch = self.fetch_minibatch(self.bcs_sampler, batch_size // 5 * 1)
            X_res_batch, u_res_batch = self.fetch_minibatch(self.res_sampler, batch_size // 5 * 4)
            
            # Tensor
            X_bcs_batch_tens = torch.from_numpy(X_bcs_batch).float().to(device)
            u_bcs_batch_tens = torch.from_numpy(u_bcs_batch).float().to(device)
            X_res_batch_tens = torch.from_numpy(X_res_batch).requires_grad_().float().to(device)
            u_res_batch_tens = torch.from_numpy(u_res_batch).float().to(device)
            
            u_pred_bcs = self.net_u(X_bcs_batch_tens[:, 0:1], X_bcs_batch_tens[:, 1:2])
            r_pred = self.net_r(X_res_batch_tens[:, 0:1], X_res_batch_tens[:, 1:2])
            
            loss_bcs = torch.mean((u_bcs_batch_tens - u_pred_bcs) ** 2)
            loss_res = torch.mean((u_res_batch_tens - r_pred )** 2)
            
            loss = loss_res + 2 * loss_bcs
            
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            
            if it % 1000 == 0:
                self.lr_scheduler.step()
            
            # Print
            if it % 1000 == 0:

                # Store losses
                self.loss_bcs_log.append(loss_bcs.detach().cpu().numpy())
                self.loss_res_log.append(loss_res.detach().cpu().numpy())
                
                print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e' %
                      (it, loss.item(), loss_res, loss_bcs))

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.nn.state_dict(), save_path)  # Сохранение весов
          
          
    # Evaluates predictions at test points
    def predict_u(self, X_star):
        
        x = torch.tensor(X_star[:, 0:1]).float().to(device)
        y = torch.tensor(X_star[:, 1:2]).float().to(device)

        self.nn.eval()

        with torch.no_grad():
            u_star = self.net_u(x, y)

        return u_star.cpu().detach()

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        
        x = torch.tensor(X_star[:, 0:1]).float().to(device)
        y = torch.tensor(X_star[:, 1:2]).float().to(device)
        
        self.nn.eval()

        #with torch.no_grad():
        r_star = self.net_r(x.requires_grad_(), y.requires_grad_())
            
        return r_star.detach().cpu()
    
    def load_best_model(self, path="best_custom.pth"):
        self.nn.load_state_dict(torch.load(path))
        self.nn.to(device)
        self.nn.eval()