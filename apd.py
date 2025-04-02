# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class ParameterComponentModel(torch.nn.Module):
    def __init__(
        self,
        n_components: int,
        features_dim: int,
        hidden_dim: int,
        encoder_bias: torch.Tensor | None = None,
        decoder_bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_components = n_components
        # Initialize encoder weight parameters
        W_enc_CFH = torch.nn.Parameter(
            torch.empty(n_components, features_dim, hidden_dim)
        )

        H = min(features_dim, hidden_dim)  # m
        self.U_CFM = torch.nn.Parameter(torch.empty(n_components, features_dim, H))
        self.V_CMH = torch.nn.Parameter(torch.empty(n_components, H, hidden_dim))
        for c in range(n_components):
            W_enc_FH = nn.init.kaiming_normal_(W_enc_CFH[c])
            U_FM, S_, V_MH = torch.linalg.svd(W_enc_FH, full_matrices=False)
            with torch.no_grad():
                self.U_CFM[c].copy_(U_FM)
                self.V_CMH[c].copy_(V_MH)
        # Apply Kaiming initialization

        self.b_enc_H = torch.nn.Parameter(
            torch.zeros(hidden_dim)
            if encoder_bias is None
            else encoder_bias.clone()
            .detach()
            .requires_grad_(False)  # freeze if passed in
        )
        self.b_dec_F = torch.nn.Parameter(
            torch.zeros(features_dim)
            if decoder_bias is None
            else decoder_bias.clone()
            .detach()
            .requires_grad_(False)  # freeze if passed in
        )
        self.relu = torch.nn.ReLU()

    @property
    def W_enc_CFH(self):
        return einops.einsum(self.U_CFM, self.V_CMH, "c f m, c m h -> c f h")

    @property
    def W_dec_HF(self):
        # Tie decoder weights to transposed encoder weights
        return einops.rearrange(self.W_enc_CFH, "c f h -> c h f")

    def get_reconstructed_weights(self, component_mask: torch.Tensor | None = None):
        if component_mask is None:
            component_mask = torch.ones(self.n_components, dtype=torch.float)
        reconstructed_W_enc_FH = einops.einsum(
            self.W_enc_CFH, component_mask, "c f h, c -> f h"
        )
        reconstructed_W_dec_HF = einops.einsum(
            self.W_dec_HF, component_mask, "c h f, c -> h f"
        )
        return reconstructed_W_enc_FH, reconstructed_W_dec_HF

    def forward(
        self, input_BF: torch.Tensor, component_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        reconstructed_W_enc_FH, reconstructed_W_dec_HF = self.get_reconstructed_weights(
            component_mask
        )
        x_BH = (
            einops.einsum(input_BF, reconstructed_W_enc_FH, "b f , f h -> b h")
            + self.b_enc_H
        )
        out_preact_BF = (
            einops.einsum(x_BH, reconstructed_W_dec_HF, "b h, h f -> b f")
            + self.b_dec_F
        )
        return self.relu(out_preact_BF)


# %%
def generate_batch(
    batch_size: int, n_features: int, feature_probability: float
) -> torch.Tensor:
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
from tqdm import tqdm  # type: ignore  # noqa: E402
# device = "mps"

batch_size = 4096
n_features = 5
hidden_dim = 2
lr = 1e-4
num_steps = 5000
sparsity = 0.2
# n_components = 1

target_model = ParameterComponentModel(1, n_features, hidden_dim)
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
import plotly.express as px  # type: ignore  # noqa: E402

# loglog plot loss
fig = px.line(losses, log_y=True)
fig.show(renderer="browser")
# %%
test_batch = generate_batch(10, n_features, sparsity)
test_output = target_model(test_batch)
print(test_batch)
print(test_output)

error = test_batch - test_output
px.imshow(error.detach().numpy())


# %%
target_model.b_enc_H
# %%

n_components = 5
reparameterized_model = ParameterComponentModel(
    n_components,
    n_features,
    hidden_dim,
    encoder_bias=target_model.b_enc_H,
    decoder_bias=target_model.b_dec_F,
)


def L_faithfulness(
    train_model: ParameterComponentModel, target_model: ParameterComponentModel
):
    train_W_enc, train_W_dec = train_model.get_reconstructed_weights()
    criterion = torch.nn.MSELoss()
    loss = criterion(train_W_enc, target_model.W_enc_CFH) + criterion(
        train_W_dec, target_model.W_dec_HF
    )
    return loss


def L_simplicity(
    params_U_CLMI: torch.Tensor,
    params_V_CLMJ: torch.Tensor,
    active_component_mask: torch.Tensor,
    p=0.9,
):
    approx_sing_values_CLM = einops.einsum(
        params_U_CLMI**2, params_V_CLMJ**2, "c l m i, c l m j -> c l m"
    ) ** (p / 2)
    approx_sing_values_C = einops.einsum(approx_sing_values_CLM, "c l m -> c")
    loss = einops.einsum(approx_sing_values_C, active_component_mask, "c, c ->")
    return loss


def L_minimality(model: ParameterComponentModel, input: torch.Tensor, active_component_mask: torch.Tensor):
    output = model.forward(input, active_component_mask)
    loss = F.mse_loss(output, input)
    return loss


# %%


# print("claude", L_simplicity(reparameterized_model, torch.ones(n_components)))  # Claude
# print("me", L_simplicity_2(reparameterized_model, torch.ones(n_components)))  # Me
# print("3",
#     L_simplicity_3(
#         params_U_CLMI=einops.rearrange(reparameterized_model.U_CFM, "c f m -> c 1 m f"),
#         params_V_CLMJ=einops.rearrange(reparameterized_model.V_CMH, "c m h -> c 1 m h"),
#         active_component_mask=torch.ones(n_components),
#     ),
# )
# TODO: why are these different? ^

def get_attributions(original_model: ParameterComponentModel, theta_model: ParameterComponentModel, input: torch.Tensor):
    n_components = theta_model.n_components
    output_dim = input.shape[1]


    attributions = []
    output = original_model.forward(input)
    for c in range(n_components):
        output_grads = []
        assert original_model.W_enc_CFH.shape[0] == 1
        W_enc_grad_FH = original_model.W_enc_CFH.grad[0] # type: ignore
        W_enc_FH = original_model.W_enc_CFH[0]  # type: ignore
        for o in range(output_dim):
            output[o].backward()
            output_grads.append((W_enc_FH * W_enc_grad_FH).sum() ** 2)
            model.zero_grad()

        attributions.append(sum(output_grads) / len(output_grads))

    return attributions


distance = kl_divergence or somethingelse

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


# %%

torch.randn(10, 10, requires_grad=True).backward()
# %%
