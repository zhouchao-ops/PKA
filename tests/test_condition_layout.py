import torch

from ominicontrol.flux.block import _merge_condition_heads, _split_condition_heads


def test_corrected_split_preserves_condition_and_head_axes():
    batch, heads, conditions, tokens, dim = 1, 4, 2, 3, 2
    packed = torch.arange(
        batch * heads * conditions * tokens * dim
    ).reshape(batch, heads, conditions * tokens, dim)
    expected = torch.stack(
        [
            packed[:, :, index * tokens : (index + 1) * tokens]
            for index in range(conditions)
        ]
    )
    actual = _split_condition_heads(
        packed, batch, heads, conditions, tokens
    )
    assert torch.equal(actual, expected)


def test_corrected_merge_inverts_corrected_split():
    batch, heads, conditions, tokens, dim = 2, 4, 3, 5, 2
    packed = torch.randn(batch, heads, conditions * tokens, dim)
    split = _split_condition_heads(
        packed, batch, heads, conditions, tokens
    )
    merged = _merge_condition_heads(
        split, batch, heads, conditions, tokens
    )
    assert torch.equal(merged, packed)

