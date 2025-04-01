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
            
        if encoder_bias is None:
            self.b_enc = torch.nn.Parameter(torch.zeros(n_components, hidden_dim))   
        else:
            self.b_enc = encoder_bias.clone()
        if decoder_bias is None:
            self.b_dec = torch.nn.Parameter(torch.zeros(n_components, output_dim))
        else:
            self.b_dec = decoder_bias.clone()
        self.relu = torch.nn.ReLU()

    @property
    def W_dec(self):
        # Tie decoder weights to transposed encoder weights
        return torch.transpose(self.W_enc, 1, 2)

    def forward(self, input: torch.Tensor, component_mask: torch.Tensor | None = None):
        if component_mask is None:
            component_mask = torch.ones(self.n_components, dtype=torch.long)

        reconstructed_W_enc = einops.einsum(self.W_enc, component_mask, "c i j, c -> i j")
        reconstructed_b_enc = einops.einsum(self.b_enc, component_mask, "c i, c -> i")
        # Use the property to get transposed weights
        W_dec = self.W_dec
        reconstructed_W_dec = einops.einsum(W_dec, component_mask, "c i j, c -> i j")
        reconstructed_b_dec = einops.einsum(self.b_dec, component_mask, "c i, c -> i")

        x = input @ reconstructed_W_enc + reconstructed_b_enc
        return self.relu(x @ reconstructed_W_dec + reconstructed_b_dec)

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
