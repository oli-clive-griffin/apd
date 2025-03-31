# import torch
# from torch import Tensor

# class DecomposedModel:
#     components_CLIJ: Tensor

#     def forward()


def forward(params, input): ...


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
