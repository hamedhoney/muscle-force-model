import torch

from muscle_force_model.constants import N_DISC, N_MUSCLE
from muscle_force_model.models import CNN, AutoEncoder


def test_autoencoder_training_forward_shape():
    model = AutoEncoder(encode_size=2, encode_feature=16, n_muscle=N_MUSCLE)
    forces = torch.rand(4, 192, N_MUSCLE)
    recon, enc = model(forces, muscle=0, training=True)
    assert recon.shape == (4, 192, N_MUSCLE)
    assert enc.shape == (4, 2, 48)


def test_autoencoder_inference_forward_shape():
    model = AutoEncoder(encode_size=2, encode_feature=16, n_muscle=N_MUSCLE)
    code = torch.rand(4, 2, 48)
    decoded, passthrough = model(code, muscle=0, training=False)
    assert decoded.shape == (4, 192, N_MUSCLE)
    assert torch.equal(passthrough, code)


def test_cnn_forward_shape():
    model = CNN(encode_size=2, encode_features=16, n_muscle=N_MUSCLE, n_disc=N_DISC)
    batch = 3
    time = 192
    chest_pelvis = torch.rand(batch, time, 1 + 6 * 3)
    nmk = torch.rand(batch, time, 2 * N_MUSCLE)
    mom = torch.rand(batch, time, N_DISC, 3)
    mom_arms = torch.rand(batch, time, N_MUSCLE, N_DISC, 3)
    dem = torch.rand(batch, 3)
    mca = torch.rand(batch, N_MUSCLE, 1)

    out = model(chest_pelvis, nmk, mom, mom_arms, dem, mca)
    assert out.shape == (batch, time // 4, 2)


def test_cnn_output_is_valid_softmax_distribution():
    model = CNN(encode_size=2, encode_features=16, n_muscle=N_MUSCLE, n_disc=N_DISC)
    chest_pelvis = torch.rand(1, 192, 1 + 6 * 3)
    nmk = torch.rand(1, 192, 2 * N_MUSCLE)
    mom = torch.rand(1, 192, N_DISC, 3)
    mom_arms = torch.rand(1, 192, N_MUSCLE, N_DISC, 3)
    dem = torch.rand(1, 3)
    mca = torch.rand(1, N_MUSCLE, 1)

    out = model(chest_pelvis, nmk, mom, mom_arms, dem, mca)
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)
