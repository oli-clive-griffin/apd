# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

    
class ParameterComponentModel(torch.nn.Module):
    def __init__(self, n_components: int, input_dim: int, hidden_dim: int, output_dim: int, encoder_bias: nn.Parameter | None = None, decoder_bias: nn.Parameter | None = None):
        super().__init__()
        self.n_components = n_components
        # Initialize encoder weight parameters
        self.W_enc = torch.nn.Parameter(torch.empty(n_components, input_dim, hidden_dim))
        # Apply Kaiming initialization
        for c in range(n_components):
            nn.init.kaiming_normal_(self.W_enc[c])
            
        self.b_enc = torch.nn.Parameter(torch.zeros(hidden_dim)) if encoder_bias is None else encoder_bias.clone()
        self.b_dec = torch.nn.Parameter(torch.zeros(output_dim)) if decoder_bias is None else decoder_bias.clone()
        self.relu = torch.nn.ReLU()

    @property
    def W_dec(self):
        # Tie decoder weights to transposed encoder weights
        return torch.transpose(self.W_enc, 1, 2)
    
    def get_reconstructed_weights(self, component_mask: torch.Tensor | None = None):
        if component_mask is None:
            component_mask = torch.ones(self.n_components, dtype=torch.float)
        reconstructed_W_enc = einops.einsum(self.W_enc, component_mask, "c i j, c -> i j")
        reconstructed_W_dec = einops.einsum(self.W_dec, component_mask, "c i j, c -> i j")
        return reconstructed_W_enc, reconstructed_W_dec

    
    def forward(self, input_BF: torch.Tensor, component_mask: torch.Tensor | None = None):
        reconstructed_W_enc, reconstructed_W_dec = self.get_reconstructed_weights(component_mask)
        x = input_BF @ reconstructed_W_enc + self.b_enc
        return self.relu(x @ reconstructed_W_dec + self.b_dec)

# %%
def generate_batch(batch_size: int, n_features: int, feature_probability: float) -> torch.Tensor:
    """
    Generates a batch of data of shape (batch_size, n_instances, n_features).
    """
    batch_shape = (batch_size, n_features)
    feat_mag = torch.rand(batch_shape)
    feat_seeds = torch.rand(batch_shape)
    return torch.where(feat_seeds <= feature_probability, feat_mag, 0.0)
# %%
input = generate_batch(10, 5, 0.2)
input
# %%
from tqdm import tqdm
# device = "mps"

batch_size = 4096
n_features = 5
hidden_dim = 2
lr = 1e-4
num_steps = 20000
sparsity = 0.2
# n_components = 1

target_model = ParameterComponentModel(1, n_features, hidden_dim, n_features)
optimizer = torch.optim.Adam(target_model.parameters(), lr=lr)
critereon = torch.nn.MSELoss()

# get initial loss
input = generate_batch(batch_size, n_features, sparsity)
output = target_model(input)
loss = critereon(output, input)
print(f"Initial loss: {loss.item()}")

losses = []
pbar = tqdm(range(num_steps))
for step in pbar:
    optimizer.zero_grad()
    input = generate_batch(batch_size, n_features, sparsity)
    output = target_model(input)
    loss = critereon(output, input)
    loss.backward()
    optimizer.step()
    pbar.set_postfix(loss=loss.item())
    if step % 1000 == 0:
        losses.append(loss.item())

# %%
import plotly.express as px
# loglog plot loss
fig = px.line(losses, log_x=True, log_y=True)
fig.show()
# %%
test_batch = generate_batch(10, n_features, sparsity)
test_output = target_model(test_batch)
print(test_batch)
print(test_output)

error = (test_batch - test_output)
px.imshow(error.detach().numpy())


# %%
n_components = 5
reparameterized_model = ParameterComponentModel(2, n_features, hidden_dim, n_features, encoder_bias=target_model.b_enc, decoder_bias=target_model.b_dec)

def L_faithfulness(train_model: ParameterComponentModel, target_model: ParameterComponentModel):
    train_W_enc, train_W_dec = train_model.get_reconstructed_weights()
    criterion = torch.nn.MSELoss()
    loss = criterion(train_W_enc, target_model.W_enc) + criterion(train_W_dec, target_model.W_dec)
    return loss

def L_simplicity(parameter_model, active_component_mask, p=0.9):
    # Extract active components
    n_components = parameter_model.n_components
    W_enc = parameter_model.W_enc  # [n_components, input_dim, hidden_dim]
    
    # Implement Schatten-p norm for each active component
    simplicity_loss = 0.0
    
    for c in range(n_components):
        if active_component_mask[c] > 0:
            comp_W_enc = W_enc[c]
            
            # Using the formulation from Equation 21 and 22
            # λ_c,l,m = (Σ_i,j U²_c,l,m,i V²_c,l,m,j)^(1/2)
            # Where U and V correspond to the factors of our component
            
            # For autoencoder architecture, we have W_enc and W_dec (tied weights)
            # We can use the singular value power method to avoid full SVD
            # For simplicity, we'll use a vectorized approximation
            
            # Calculate U²_c,l,m,i and V²_c,l,m,j
            U_squared = torch.sum(comp_W_enc**2, dim=1)  # [input_dim]
            V_squared = torch.sum(comp_W_enc**2, dim=0)  # [hidden_dim]
            
            # Outer product approximation of singular values squared
            singular_values_squared = torch.outer(U_squared, V_squared)  # [input_dim, hidden_dim]
            print(singular_values_squared.shape)
            
            # Apply p/2 power to approximate Schatten norm as in Equation 22
            schatten_p_norm = torch.sum(singular_values_squared**(p/2))
            
            simplicity_loss += schatten_p_norm
    
    return simplicity_loss

def L_simplicity_2(parameter_model, active_component_mask, p=0.9):
    n_components = parameter_model.n_components
    W_enc = parameter_model.W_enc  # [n_components, input_dim, hidden_dim]
    
    simplicity_loss = 0.0
    for c in range(n_components):
        if active_component_mask[c] > 0:
            comp_W_enc = W_enc[c]
            _, S, _ = comp_W_enc.svd()
            schatten_p_norm = S.norm(p=p)
            simplicity_loss += schatten_p_norm
    return simplicity_loss


def L_minimality(model: ParameterComponentModel, input: torch.Tensor):
    # TODO
    return 0

# %%
print(L_faithfulness(reparameterized_model, target_model))
print(L_simplicity(reparameterized_model, torch.ones(n_components)))  # Claude
print(L_simplicity_2(reparameterized_model, torch.ones(n_components)))  # Me
# TODO: why are these different? ^

# %%
assert False

# %%





def faithfulness(original_params, theta_params):
    reconstructed_model = sum(theta_params)
    return mse(original_params, reconstructed_model)


def simplicity(theta_params, p=0.9):
    loss = 0
    for weight_matrix in theta_params:
        loss += lp_norm(weight_matrix, p) ** p
    return loss


def get_attributions(original_params, theta_params, input):
    attributions = []
    for c in range(len(components)):
        P_c = theta_params[c]
        output_grads = []
        for o in range(output_dim):
            output = forward(original_params, input)
            output[o].backward()

            gradient = 0
            for l in range(layers):
                for i in range(dim_i):
                    for j in range(dim_j):
                        gradient += original_params.grad[l, i, j] * P_c[l, i, j]

            output_grads.append(gradient**2)

        attributions.append(mean(output_grads))

    return attributions


distance = kl_divergence or somethingelse


def minimality(x, topk_theta_params, y):
    y_hat = forward(sum(topk_theta_params), x)
    # and also maybe activations
    return distance(y, y_hat)


input = somedata
theta_params = random_like(model)
original_params = ...


def attr_step():
    attributions = get_attributions(original_params, theta_params, input)
    topk_params = attributions.topk(k)


def train_step(topk_params):
    # output = forward(sum(topk_params), input)

    loss = (
        faithfulness(original_params, theta_params)
        + beta * minimality(topk_params)
        + alpha * simplicity(topk_params)
    )


def train():
    for step in steps:
        topk_params = attr_step(input)
        train_step(topk_params)
