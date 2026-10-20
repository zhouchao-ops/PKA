import torch
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F
from .lora_controller import enable_lora
import math
import os
import ast 
import pickle
from ..utils import GLOBAL_CONFIG,conditional_gpu_profile,subject_intermediate


def _split_condition_heads(
    tensor: Optional[torch.Tensor],
    batch: int,
    heads: int,
    condition_num: int,
    tokens_per_condition: int
) -> Optional[torch.Tensor]:
    """Split packed condition tokens while preserving condition/head axes."""
    if tensor is None:
        return None
    # Input is [B, H, C*T, D]. Split C inside the token dimension, then move
    # it to the front explicitly; reshape alone cannot swap C and H.
    return tensor.reshape(
        batch, heads, condition_num, tokens_per_condition, -1
    ).permute(2, 0, 1, 3, 4)


def _merge_condition_heads(
    tensor: torch.Tensor,
    batch: int,
    heads: int,
    condition_num: int,
    tokens_per_condition: int
) -> torch.Tensor:
    """Inverse of the corrected condition-head split."""
    return tensor.permute(1, 2, 0, 3, 4).reshape(
        batch, heads, condition_num * tokens_per_condition, -1
    )


def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
    use_cache = False,
    kv_cache = None,
) -> torch.FloatTensor:
    batch_size,_,_ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), model_config.get("latent_lora", False)
    ):
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )
        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)
    
    # calculate qkv 
    L_text = 512
    query_text = query[:,:,:L_text]
    query_latent = query[:,:,L_text:]
    key_text = key[:,:,:L_text]
    key_latent = key[:,:,L_text:]
    value_text = value[:,:,:L_text]
    value_latent = value[:,:,L_text:]

    if condition_latents is not None:
        cond_query = attn.to_q(condition_latents)
        cond_key = attn.to_k(condition_latents)
        cond_value = attn.to_v(condition_latents)

        cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        if attn.norm_q is not None:
            cond_query = attn.norm_q(cond_query)
        if attn.norm_k is not None:
            cond_key = attn.norm_k(cond_key)

    if cond_rotary_emb is not None:
        cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
        cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)

    ## kv cache
    if kv_cache is not None:
        
        layer_type = GLOBAL_CONFIG.get("LAYER_TYPE","mm")
        layer = int(GLOBAL_CONFIG.get("LAYER","0"))
        if not use_cache:
            kv_cache.update(f"{layer_type}_{layer}",cond_key,cond_value)
        else:
            cond_key,cond_value = kv_cache[f"{layer_type}_{layer}"]
            cond_query = None

    L_cond_single = 1024
    scale = None
    dropout_p = 0.0
    def attn_product(query,key,value):
        return F.scaled_dot_product_attention(query, key, value)
    # @conditional_gpu_profile
    def attn_product_cond(query,key,value,
                            query_text,key_text,value_text,
                            condition_query,condition_key,condition_value):
        key = torch.cat([key_text,key],dim = -2)
        value = torch.cat([value_text,value],dim=-2)
        out_text = attn_product(query_text,key,value)

        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        batch,head = query.size(0),query.size(1)
        attn_weight = query @ key.transpose(-2,-1) * scale_factor
        L_cond = condition_key.size(-2)
        condition_num = L_cond // L_cond_single
        # reshape condition
        condition_query = _split_condition_heads(
            condition_query, batch, head, condition_num, L_cond_single
        )
        condition_key = _split_condition_heads(
            condition_key, batch, head, condition_num, L_cond_single
        )
        condition_value = _split_condition_heads(
            condition_value, batch, head, condition_num, L_cond_single
        )
        subject =  model_config.get("subject_region",False)

        spatial_begin = 1 if subject else 0
        spatial_key = condition_key[spatial_begin:]
        spatial_value = condition_value[spatial_begin:]
        # 低效代码
        # spatial_attn = torch.einsum("...nd,...nd->...n",query,spatial_key) * scale_factor
        # spatial_attn = spatial_attn.squeeze(1)
        # spatial_attn = spatial_attn.permute(1,2,0)
        # spatial_attn = spatial_attn.unsqueeze(0)
        query_expanded = query.unsqueeze(0)
        spatial_attn_raw = (query_expanded * spatial_key).sum(dim=-1)
        spatial_attn = spatial_attn_raw.permute(1, 2, 3, 0)
        spatial_attn = spatial_attn * scale_factor

        
        if subject: 
            # cond is [cond_subject, cond_spatialxN]
            import pickle
            step = GLOBAL_CONFIG.get("STEP","0")
            subject_key = condition_key[0]
            subject_value = condition_value[0]
            mask = None
            if step != "0" and model_config.get("threshold",0.0) > 0:
                attn_type = "key_attn_0"
                mask = subject_intermediate["mask"].to(query.device)
                # with open(f"attn_weights/{attn_type}/{step}/mask.pkl",'rb')as f:
                #     mask = pickle.load(f).to(query.device)
            if mask is None:
                subject_attn = query @ subject_key.transpose(-2,-1) * scale_factor
                attn = torch.softmax(torch.cat([attn_weight,
                                subject_attn,spatial_attn],dim=-1),dim=-1)
            
                spatial_attn = attn[...,-condition_num+1:]
                out_spatial = torch.einsum('bhnc,cbhnd->bhnd', spatial_attn, spatial_value)
                out = attn[...,:-condition_num+1] @ torch.cat([value,subject_value],dim=-2)
                out = out + out_spatial
            else:
                # query_wo_subject = query[:,:, ~mask] 
                query_w_subject = query[:,:,mask]
                subject_attn = query_w_subject @ subject_key.transpose(-2,-1) * scale_factor
                attn_w_subject = torch.softmax(torch.cat([attn_weight[:,:,mask],
                                subject_attn,spatial_attn[:,:,mask]],dim=-1),dim=-1)
                attn_wo_subject = torch.softmax(torch.cat([attn_weight[:,:,~mask],
                                spatial_attn[:,:,~mask]],dim=-1),dim=-1)
                spatial_attn_w_subject = attn_w_subject[...,-condition_num+1:]
                spatial_attn_wo_subject = attn_wo_subject[...,-condition_num+1:]
                B, H, M_total = attn_weight.shape[:3]
                condition_num = spatial_attn_w_subject.shape[-1]
                spatial_attn = torch.empty(B, H, M_total, condition_num, dtype=spatial_attn_w_subject.dtype, device=spatial_attn_w_subject.device)
                # 直接使用布尔掩码在第2个维度(M_total)上进行赋值，代码更简洁高效
                spatial_attn[:, :, mask] = spatial_attn_w_subject
                spatial_attn[:, :, ~mask] = spatial_attn_wo_subject
                out_spatial = (spatial_attn.unsqueeze(0).reshape(condition_num,batch,head,L_cond_single,-1)* spatial_value).sum(dim=0)
                # out_spatial = torch.einsum('bhnc,cbhnd->bhnd', spatial_attn, spatial_value)
                out_w_subject = attn_w_subject[...,:-condition_num] @ torch.cat([value,subject_value],dim=-2)
                out_wo_subject = attn_wo_subject[...,:-condition_num] @ value
                # 同样，使用 torch.empty 和布尔掩码赋值
                out = torch.empty(B, H, M_total, out_wo_subject.shape[-1], dtype=out_wo_subject.dtype, device=out_wo_subject.device)
                out[:, :, mask] = out_w_subject
                out[:, :, ~mask] = out_wo_subject
                out = out + out_spatial

            if model_config.get("threshold",0.0) > 0:
                # mask extraction
                attn_weight = torch.softmax(attn_weight, dim = -1)
                step = GLOBAL_CONFIG.get("STEP")
                layer = GLOBAL_CONFIG.get("LAYER")
                layer_type = GLOBAL_CONFIG.get("LAYER_TYPE")
                key_positions = GLOBAL_CONFIG.get("KEY_POSITIONS").split(";")
                for i,key_position in enumerate(key_positions):
                    begin, end = ast.literal_eval(key_position)
                    key_attn = (attn_weight[0,:,:,begin:end]).sum(dim = -1).cpu().mean(dim=0)
                    # os.makedirs(f"attn_weights/key_attn_{i}/{step}",exist_ok=True)
                    # with open(f"attn_weights/key_attn_{i}/{step}/{layer_type}_{layer}.pkl",'wb')as f:
                    #     pickle.dump(key_attn,f)
                    subject_intermediate[f"{i}_{layer_type}_{layer}"] = key_attn
                # cond_attn = (attn_weight[0,:,:,-1024:]).sum(dim = -1).cpu().mean(dim=0)
                # os.makedirs(f"attn_weights/cond_attn/{step}",exist_ok=True)
                # with open(f"attn_weights/cond_attn/{step}/{layer_type}_{layer}.pkl",'wb')as f:
                #     pickle.dump(cond_attn,f)
        else:
            attn = torch.cat([attn_weight,spatial_attn],dim=-1)
            attn = torch.softmax(attn,dim=-1)
            spatial_attn = attn[...,-condition_num:]
            out_spatial = (spatial_attn.unsqueeze(0).reshape(condition_num,batch,head,L_cond_single,-1) * spatial_value).sum(dim=0)
            # out_spatial = torch.einsum('bhnc,cbhnd->bhnd', spatial_attn, spatial_value)
            out = attn[...,:-condition_num] @ value + out_spatial
        out = torch.cat([out_text,out],dim=-2)
        if not model_config.get("wo_cond_self_attn",False) and not condition_query is None:
            attn_weight_cond = condition_query @ condition_key.transpose(-2,-1) * scale_factor
            attn_weight_cond = torch.softmax(attn_weight_cond, dim=-1)
            # attn_weight_cond = torch.dropout(attn_weight_cond,dropout_p,train=True)
        
            attn_cond = attn_weight_cond @ condition_value
            attn_cond = _merge_condition_heads(
                attn_cond, batch, head, condition_num, L_cond_single
            )
            out = torch.cat([out,attn_cond], dim = -2)
        return out
    def attn_product_cond_compact(query,key,value,
                            query_text,key_text,value_text,
                            condition_query,condition_key,condition_value):
        key = torch.cat([key_text, key], dim=-2)
        value = torch.cat([value_text, value], dim=-2)
        out_text = attn_product(query_text, key, value)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        batch, head, target_tokens = query.shape[:3]
        L_cond = condition_key.size(-2)
        condition_num = model_config["_condition_count"]
        token_counts = model_config["_condition_token_counts"]
        if len(token_counts) != condition_num or len(set(token_counts)) != 1:
            raise ValueError("Compact PAA requires equal token counts for all conditions")
        condition_tokens = token_counts[0]
        if L_cond != condition_num * condition_tokens:
            raise ValueError("Condition token metadata does not match condition K/V shape")
        condition_side = math.isqrt(condition_tokens)
        target_side = math.isqrt(target_tokens)
        if condition_side**2 != condition_tokens or target_side**2 != target_tokens:
            raise ValueError("Compact PAA requires square condition and target token grids")
        if target_side % condition_side:
            raise ValueError("Target PAA grid must be an integer multiple of condition grid")
        repeat = target_side // condition_side
        condition_query = _split_condition_heads(
            condition_query, batch, head, condition_num, condition_tokens
        )
        condition_key = _split_condition_heads(
            condition_key, batch, head, condition_num, condition_tokens
        )
        condition_value = _split_condition_heads(
            condition_value, batch, head, condition_num, condition_tokens
        )
        subject = model_config.get("subject_region", False)
        spatial_begin = 1 if subject else 0
        spatial_count = condition_num - spatial_begin
        if spatial_count <= 0:
            raise ValueError("Compact PAA requires at least one spatial condition")
        spatial_key = condition_key[spatial_begin:]
        spatial_value = condition_value[spatial_begin:]
        query_grid = query.reshape(
            batch, head, condition_side, repeat, condition_side, repeat, query.shape[-1]
        )
        key_grid = spatial_key.reshape(
            spatial_count, batch, head, condition_side, 1, condition_side, 1, spatial_key.shape[-1]
        )
        spatial_attn = (query_grid.unsqueeze(0) * key_grid).sum(dim=-1)
        spatial_attn = spatial_attn.reshape(spatial_count,batch,head,target_tokens)
        spatial_attn = spatial_attn.permute(1,2,3,0) * scale_factor
        attn_weight = query @ key.transpose(-2,-1) * scale_factor

        def apply_spatial_values(spatial_weights, target_indices=None):
            selected_tokens = spatial_weights.shape[2]
            if selected_tokens == target_tokens:
                weights_grid = spatial_weights.permute(3,0,1,2).reshape(
                    spatial_count,batch,head,condition_side,repeat,condition_side,repeat,1
                )
                value_grid = spatial_value.reshape(
                    spatial_count,batch,head,condition_side,1,condition_side,1,spatial_value.shape[-1]
                )
                return (weights_grid * value_grid).sum(dim=0).reshape(
                    batch,head,target_tokens,spatial_value.shape[-1]
                )
            # Subject masking selects an arbitrary subset of target positions. Map
            # every selected target token back to its aligned compact token.
            if target_indices is None:
                raise ValueError("Compact PAA requires target indices for masked spatial attention")
            rows = torch.div(target_indices, target_side, rounding_mode="floor")
            cols = target_indices.remainder(target_side)
            compact_indices = torch.div(rows, repeat, rounding_mode="floor") * condition_side
            compact_indices += torch.div(cols, repeat, rounding_mode="floor")
            selected_values = spatial_value[:,:, :, compact_indices, :]
            return (spatial_weights.permute(3,0,1,2).unsqueeze(-1) * selected_values).sum(dim=0)

        if subject:
            subject_key = condition_key[0]
            subject_value = condition_value[0]
            step = GLOBAL_CONFIG.get("STEP", "0")
            ksa_enabled = model_config.get(
                "ksa_enabled", model_config.get("threshold", 0.0) > 0
            )
            mask = None
            if step != "0" and ksa_enabled:
                mask = subject_intermediate["mask"]
            if mask is None:
                subject_attn = query @ subject_key.transpose(-2,-1) * scale_factor
                attn = torch.softmax(
                    torch.cat([attn_weight, subject_attn, spatial_attn], dim=-1), dim=-1
                )
                out_spatial = apply_spatial_values(attn[..., -spatial_count:])
                out = attn[..., :-spatial_count] @ torch.cat([value,subject_value],dim=-2)
                out = out + out_spatial
            else:
                subject_attn = query[:,:,mask] @ subject_key.transpose(-2,-1) * scale_factor
                attn_w_subject = torch.softmax(
                    torch.cat([attn_weight[:,:,mask], subject_attn, spatial_attn[:,:,mask]], dim=-1),
                    dim=-1,
                )
                attn_wo_subject = torch.softmax(
                    torch.cat([attn_weight[:,:,~mask], spatial_attn[:,:,~mask]], dim=-1),
                    dim=-1,
                )
                out_w_subject = (
                    attn_w_subject[..., :-spatial_count] @ torch.cat([value,subject_value],dim=-2)
                    + apply_spatial_values(
                        attn_w_subject[..., -spatial_count:], mask.nonzero(as_tuple=True)[0]
                    )
                )
                out_wo_subject = (
                    attn_wo_subject[..., :-spatial_count] @ value
                    + apply_spatial_values(
                        attn_wo_subject[..., -spatial_count:], (~mask).nonzero(as_tuple=True)[0]
                    )
                )
                out = torch.empty(
                    batch,head,target_tokens,out_wo_subject.shape[-1],
                    dtype=out_wo_subject.dtype,device=out_wo_subject.device,
                )
                out[:,:,mask] = out_w_subject
                out[:,:,~mask] = out_wo_subject

            # Collect keyword-to-latent maps for KSA while retaining the
            # spatial PAA branch. Raw-logit mode uses the already-computed QK
            # scores; softmax mode matches the original base-attention map.
            if not bool(os.getenv("TRAIN", False)) and ksa_enabled:
                score_mode = model_config.get("ksa_score_mode", "softmax")
                if score_mode not in ("softmax", "raw_logit"):
                    raise ValueError(
                        f"Unknown ksa_score_mode={score_mode!r}; expected "
                        "'softmax' or 'raw_logit'"
                    )
                base_probs = (
                    torch.softmax(attn_weight, dim=-1)
                    if score_mode == "softmax"
                    else None
                )
                layer = GLOBAL_CONFIG.get("LAYER")
                layer_type = GLOBAL_CONFIG.get("LAYER_TYPE")
                key_positions = GLOBAL_CONFIG.get("KEY_POSITIONS").split(";")
                for index, key_position in enumerate(key_positions):
                    begin, end = ast.literal_eval(key_position)
                    if score_mode == "raw_logit":
                        key_attn = attn_weight[0, :, :, begin:end].mean(dim=-1)
                    else:
                        key_attn = base_probs[0, :, :, begin:end].sum(dim=-1)
                    subject_intermediate[f"{index}_{layer_type}_{layer}"] = key_attn
        else:
            attn = torch.softmax(torch.cat([attn_weight, spatial_attn], dim=-1), dim=-1)
            out_spatial = apply_spatial_values(attn[..., -spatial_count:])
            out = attn[..., :-spatial_count] @ value + out_spatial
        out = torch.cat([out_text,out],dim=-2)
        if not model_config.get("wo_cond_self_attn",False) and condition_query is not None:
            attn_cond = attn_product(condition_query,condition_key,condition_value)
            attn_cond = _merge_condition_heads(
                attn_cond, batch, head, condition_num, condition_tokens
            )
            out = torch.cat([out,attn_cond],dim=-2)
        return out

    def attn_product_cond_compact_local4(query,key,value,
                            query_text,key_text,value_text,
                            condition_query,condition_key,condition_value):
        """Spatial-only compact PAA with the four nearest compact-grid keys."""
        if model_config.get("subject_region", False):
            raise ValueError("compact_paa_mode=local4 supports spatial-only conditions")
        key = torch.cat([key_text, key], dim=-2)
        value = torch.cat([value_text, value], dim=-2)
        out_text = attn_product(query_text, key, value)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        batch, head, target_tokens = query.shape[:3]
        condition_num = model_config["_condition_count"]
        token_counts = model_config["_condition_token_counts"]
        if len(token_counts) != condition_num or len(set(token_counts)) != 1:
            raise ValueError("Compact local4 PAA requires equal condition token counts")
        condition_tokens = token_counts[0]
        condition_side = math.isqrt(condition_tokens)
        target_side = math.isqrt(target_tokens)
        if condition_side**2 != condition_tokens or target_side**2 != target_tokens:
            raise ValueError("Compact local4 PAA requires square token grids")
        if target_side != 2 * condition_side:
            raise ValueError("compact_paa_mode=local4 requires compact factor 2")
        L_cond = condition_key.size(-2)
        if L_cond != condition_num * condition_tokens:
            raise ValueError("Condition token metadata does not match condition K/V shape")
        condition_query = _split_condition_heads(
            condition_query, batch, head, condition_num, condition_tokens
        )
        condition_key = _split_condition_heads(
            condition_key, batch, head, condition_num, condition_tokens
        )
        condition_value = _split_condition_heads(
            condition_value, batch, head, condition_num, condition_tokens
        )

        # Compact RoPE centers are 0.5, 2.5, ..., so bracket each integer
        # target coordinate by its lower/upper compact centers.
        coords = torch.arange(target_side, device=query.device)
        lower_raw = torch.div(coords - 1, 2, rounding_mode="floor")
        upper_raw = lower_raw + 1
        lower_valid = lower_raw >= 0
        upper_valid = upper_raw < condition_side
        lower = lower_raw.clamp(0, condition_side - 1)
        upper = upper_raw.clamp(0, condition_side - 1)
        rows = torch.arange(target_side, device=query.device)[:, None].expand(target_side, target_side).reshape(-1)
        cols = torch.arange(target_side, device=query.device)[None, :].expand(target_side, target_side).reshape(-1)
        row_idx = torch.stack([lower[rows], lower[rows], upper[rows], upper[rows]], dim=-1)
        col_idx = torch.stack([lower[cols], upper[cols], lower[cols], upper[cols]], dim=-1)
        neighbor_idx = row_idx * condition_side + col_idx
        neighbor_valid = torch.stack([
            lower_valid[rows] & lower_valid[cols],
            lower_valid[rows] & upper_valid[cols],
            upper_valid[rows] & lower_valid[cols],
            upper_valid[rows] & upper_valid[cols],
        ], dim=-1)

        local_key = condition_key[:, :, :, neighbor_idx, :]
        local_value = condition_value[:, :, :, neighbor_idx, :]
        spatial_scores = (query.unsqueeze(0).unsqueeze(-2) * local_key).sum(dim=-1)
        spatial_scores = spatial_scores.permute(1,2,3,0,4) * scale_factor
        valid = neighbor_valid[None,None,:,None,:].expand_as(spatial_scores)
        spatial_scores = spatial_scores.masked_fill(~valid, torch.finfo(spatial_scores.dtype).min)
        spatial_scores = spatial_scores.reshape(batch,head,target_tokens,condition_num * 4)

        base_scores = query @ key.transpose(-2,-1) * scale_factor
        probs = torch.softmax(torch.cat([base_scores, spatial_scores], dim=-1), dim=-1)
        spatial_probs = probs[..., -condition_num * 4:].reshape(
            batch,head,target_tokens,condition_num,4
        )
        out_spatial = (
            spatial_probs.permute(3,0,1,2,4).unsqueeze(-1) * local_value
        ).sum(dim=(0,4))
        out = probs[..., :-condition_num * 4] @ value + out_spatial
        out = torch.cat([out_text,out],dim=-2)
        if not model_config.get("wo_cond_self_attn",False) and condition_query is not None:
            attn_cond = attn_product(condition_query,condition_key,condition_value)
            attn_cond = _merge_condition_heads(
                attn_cond, batch, head, condition_num, condition_tokens
            )
            out = torch.cat([out,attn_cond],dim=-2)
        return out

    # @conditional_gpu_profile
    def attn_product_subject(query,key,value,
                            query_text,key_text,value_text,
                            condition_query,condition_key,condition_value):
        key = torch.cat([key_text,key],dim = -2)
        value = torch.cat([value_text,value],dim=-2)
        out_text = attn_product(query_text,key,value)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        B,H = query.size(0),query.size(1)
        attn_weight = query @ key.transpose(-2,-1) * scale_factor
        import pickle
        step = GLOBAL_CONFIG.get("STEP","0")
        attn_type = "key_attn_0"
        subject_key = condition_key
        subject_value = condition_value
        ksa_enabled = model_config.get(
            "ksa_enabled", model_config.get("threshold", 0.0) > 0
        )
        if step != "0" and ksa_enabled:
            # with open(f"attn_weights/{attn_type}/{step}/mask.pkl",'rb')as f:
                B, H, M_total = attn_weight.shape[:3]
                # mask = pickle.load(f).to(attn_weight.device)
                mask = subject_intermediate["mask"]
                query_w_subject = query[:,:,mask]
                subject_attn = query_w_subject @ subject_key.transpose(-2,-1) * scale_factor
                attn_w_subject = torch.softmax(torch.cat([attn_weight[:,:,mask],
                                subject_attn],dim=-1),dim=-1)
                attn_wo_subject = torch.softmax(torch.cat([attn_weight[:,:,~mask],
                                ],dim=-1),dim=-1)
                mask_idx = subject_intermediate["mask_idx"]
                non_mask_idx = subject_intermediate["non_mask_idx"]
                out_w_subject = attn_w_subject @ torch.cat([value,subject_value],dim=-2)
                out_wo_subject = attn_wo_subject @ value
                # Both index sets overwrite the complete output tensor.
                out = torch.empty(B, H, M_total, out_wo_subject.shape[-1], dtype=out_wo_subject.dtype, device=out_wo_subject.device)
                out = out.index_copy(2,mask_idx,out_w_subject)
                out = out.index_copy(2,non_mask_idx,out_wo_subject)
        else:
            subject_attn = query @ subject_key.transpose(-2,-1) * scale_factor
            attn = torch.softmax(torch.cat([attn_weight,subject_attn],dim = -1),dim=-1)
            out = attn @ torch.cat([value,subject_value],dim=-2)
        if not model_config.get("wo_cond_self_attn",False) and not condition_query is None:
            attn_cond = attn_product(condition_query,condition_key,condition_value)
            out = torch.cat([out,attn_cond], dim = -2)
        
        # mask extraction
        if not bool(os.getenv("TRAIN",False)) and ksa_enabled:
            step = GLOBAL_CONFIG.get("STEP")
            layer = GLOBAL_CONFIG.get("LAYER")
            layer_type = GLOBAL_CONFIG.get("LAYER_TYPE")
            key_positions = GLOBAL_CONFIG.get("KEY_POSITIONS").split(";")
            score_mode = model_config.get("ksa_score_mode", "softmax")
            if score_mode not in ("softmax", "raw_logit"):
                raise ValueError(
                    f"Unknown ksa_score_mode={score_mode!r}; expected "
                    "'softmax' or 'raw_logit'"
                )
            reuse_attention = model_config.get("ksa_reuse_attention", True)

            if score_mode == "raw_logit":
                base_probs = None
            elif reuse_attention:
                if step != "0":
                    base_with_subject = torch.softmax(
                        attn_weight[:, :, mask], dim=-1
                    )
                    base_without_subject = attn_wo_subject
                else:
                    base_probs = torch.softmax(attn_weight, dim=-1)
            else:
                # Reference path for numerical-equivalence checks.
                base_probs = torch.softmax(attn_weight, dim=-1)

            for i,key_position in enumerate(key_positions):
                begin, end = ast.literal_eval(key_position)
                if score_mode == "raw_logit":
                    key_attn = attn_weight[0, :, :, begin:end].mean(dim=-1)
                elif reuse_attention and step != "0":
                    key_attn = torch.empty(
                        H, M_total, dtype=base_with_subject.dtype,
                        device=base_with_subject.device,
                    )
                    key_attn[:, mask] = base_with_subject[0, :, :, begin:end].sum(dim=-1)
                    key_attn[:, ~mask] = base_without_subject[0, :, :, begin:end].sum(dim=-1)
                else:
                    key_attn = base_probs[0, :, :, begin:end].sum(dim=-1)
                subject_intermediate[f"{i}_{layer_type}_{layer}"] = key_attn
                # os.makedirs(f"attn_weights/key_attn_{i}/{step}",exist_ok=True)
                # with open(f"attn_weights/key_attn_{i}/{step}/{layer_type}_{layer}.pkl",'wb')as f:
                #     pickle.dump(key_attn,f)

        return torch.cat([out_text,out],dim=-2)
    @conditional_gpu_profile
    def slide_window_attn(query,key,value,
                            query_text,key_text,value_text,
                            condition_query,condition_key,condition_value):
        def prepare_sliding_window_inputs(
            key: torch.Tensor, 
            value: torch.Tensor, 
            window_size: int,
            target_len: int
        ) -> (torch.Tensor, torch.Tensor):
            """
            为滑动窗口注意力准备 K 和 V 输入 (使用 pad + unfold)。
            
            Args:
                key, value: (B, H, N_kv, D)
                window_size: 窗口大小 (W)，必须为奇数。
                target_len: 目标查询序列的长度 (N_q)。

            Returns:
                k_windows: (B, H, N_q, D, W)
                v_windows: (B, H, N_q, D, W)
            """
            B, H, N_kv, D = key.shape
            
            assert window_size % 2 == 1, "Window size must be an odd number."
            
            padding = window_size // 2
            
            # 对 K 和 V 进行 padding
            key_padded = F.pad(key, (0, 0, padding, padding), 'constant', 0)
            value_padded = F.pad(value, (0, 0, padding, padding), 'constant', 0)
            
            # 确保长度与 query 匹配 (简单的对齐策略)
            if target_len != N_kv:
                current_len = key_padded.shape[2]
                if target_len + 2 * padding > current_len:
                    key_padded = F.pad(key_padded, (0, 0, 0, target_len + 2 * padding - current_len), 'constant', 0)
                    value_padded = F.pad(value_padded, (0, 0, 0, target_len + 2 * padding - current_len), 'constant', 0)
                else:
                    key_padded = key_padded[:, :, :target_len + 2 * padding]
                    value_padded = value_padded[:, :, :target_len + 2 * padding]

            # 使用 unfold 创建滑动窗口
            k_windows = key_padded.unfold(dimension=2, size=window_size, step=1)
            v_windows = value_padded.unfold(dimension=2, size=window_size, step=1)

            return k_windows, v_windows

        WINDOW_SIZE = model_config["window_size"]
        key_self_text = torch.cat([key_text, key], dim=-2)
        value_self_text = torch.cat([value_text, value], dim=-2)
        out_text = attn_product(query_text, key_self_text, value_self_text)
        head_dim = query.shape[-1]
        scale_factor = 1 / math.sqrt(head_dim) if scale is None else scale
        # 3a. 计算 Self/Text 的得分 (标准注意力)
        # (B, H, N_latent, D) @ (B, H, D, N_self_text) -> (B, H, N_latent, N_self_text)
        scores_self_text = (query @ key_self_text.transpose(-2, -1)) * scale_factor
        # 3b. 计算 Condition 的得分 (滑动窗口注意力)
        # 使用 cond_key 和 cond_value (它们对应于 condition_latents)
        # cond_key shape: (B, H, N_cond, D)

        # 准备窗口化的 K 和 V
        k_cond_windows, v_cond_windows = prepare_sliding_window_inputs(
            condition_key, condition_value, WINDOW_SIZE, target_len=query.shape[2]
        )
        # k_cond_windows shape: (B, H, N_latent, D, W)  
        # 计算得分: (B,H,N_latent,1,D) @ (B,H,N_latent,D,W) -> (B,H,N_latent,W)
        query_latent_unsqueezed = query.unsqueeze(-2)
        scores_cond = (query_latent_unsqueezed @ k_cond_windows).squeeze(-2) * scale_factor
        # 4. 统一 Softmax
        len_self_text = scores_self_text.shape[-1]
        len_cond = scores_cond.shape[-1] # 长度等于 WINDOW_SIZE

        # 拼接所有得分
        all_scores = torch.cat([scores_self_text, scores_cond], dim=-1)
        
        # 应用统一 Softmax
        all_probs = F.softmax(all_scores, dim=-1)
        # 5. 拆分概率并计算加权输出
        probs_self_text, probs_cond = torch.split(all_probs, [len_self_text, len_cond], dim=-1)

        # 5a. 计算 Self/Text 的输出
        out_self_text = probs_self_text @ value_self_text

        # 5b. 计算 Condition 的输出 (滑动窗口)
        # v_cond_windows shape: (B, H, N_latent, D, W) -> (B, H, N_latent, W, D)
        v_cond_windows_permuted = v_cond_windows.permute(0, 1, 2, 4, 3)
        # probs_cond shape: (B, H, N_latent, W) -> (B, H, N_latent, 1, W)
        probs_cond_unsqueezed = probs_cond.unsqueeze(-2)
        # (B,H,N,1,W) @ (B,H,N,W,D) -> (B,H,N,1,D) -> (B,H,N,D)
        out_cond = (probs_cond_unsqueezed @ v_cond_windows_permuted).squeeze(-2)

        # 6. 合并 Latent 的所有输出
        out_latent = out_self_text + out_cond
        out = torch.cat([out_text, out_latent], dim=-2)
        if not model_config.get("wo_cond_self_attn",False) and not condition_query is None:
            attn_cond = attn_product(condition_query,condition_key,condition_value)
            out = torch.cat([out,attn_cond], dim = -2)
        return out

    if model_config.get("spatial_condition_independent",False):
        # from torch.profiler import profile, record_function, ProfilerActivity
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("attn_product_cond_call"):

        #         hidden_states = attn_product_cond(
        #             query = query_latent, key = key_latent, value= value_latent,
        #             query_text = query_text, key_text = key_text, value_text = value_text,
        #             condition_query = cond_query, condition_key = cond_key, condition_value = cond_value,
        #         )
        # input(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        compact_mode = model_config.get("compact_paa_mode", "aligned1")
        if model_config.get("_compact_paa", False) and compact_mode == "local4":
            paa_fn = attn_product_cond_compact_local4
        else:
            paa_fn = attn_product_cond_compact if model_config.get("_compact_paa", False) else attn_product_cond
        hidden_states = paa_fn(
            query = query_latent, key = key_latent, value= value_latent,
            query_text = query_text, key_text = key_text, value_text = value_text,
            condition_query = cond_query, condition_key = cond_key, condition_value = cond_value,
        )
    elif model_config.get("subject_region",False):
        hidden_states = attn_product_subject(
            query = query_latent, key = key_latent, value= value_latent,
            query_text = query_text, key_text = key_text, value_text = value_text,
            condition_query = cond_query, condition_key = cond_key, condition_value = cond_value,
        )
    elif model_config.get("window_size",0) > 0:
        hidden_states = slide_window_attn(
            query = query_latent, key = key_latent, value= value_latent,
            query_text = query_text, key_text = key_text, value_text = value_text,
            condition_query = cond_query, condition_key = cond_key, condition_value = cond_value,
        )

    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)
    if encoder_hidden_states is not None:
        if condition_latents is not None:
            encoder_hidden_states, hidden_states, condition_latents = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[
                    :, encoder_hidden_states.shape[1] : -condition_latents.shape[1]
                ],
                hidden_states[:, -condition_latents.shape[1] :],
            )
        else:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

        with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if condition_latents is not None:
            condition_latents = attn.to_out[0](condition_latents)
            condition_latents = attn.to_out[1](condition_latents)

        return (
            (hidden_states, encoder_hidden_states, condition_latents)
            if condition_latents is not None
            else (hidden_states, encoder_hidden_states)
        )
    elif condition_latents is not None:
        # if there are condition_latents, we need to separate the hidden_states and the condition_latents
        hidden_states, condition_latents = (
            hidden_states[:, : -condition_latents.shape[1]],
            hidden_states[:, -condition_latents.shape[1] :],
        )
        return hidden_states, condition_latents
    else:
        return hidden_states

def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    use_cache = False,
    kv_cache = None,
    cond_rotary_emb=None,
    image_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):
    use_cond = condition_latents is not None
    with enable_lora((self.norm1.linear,), model_config.get("latent_lora", False)):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    if use_cond:
        (
            norm_condition_latents,
            cond_gate_msa,
            cond_shift_mlp,
            cond_scale_mlp,
            cond_gate_mlp,
        ) = self.norm1(condition_latents, emb=cond_temb)

    # Attention.
    result = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
        use_cache = use_cache,
        kv_cache = kv_cache,
    )
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    # 3. condition_latents
    if use_cond:
        cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
        condition_latents = condition_latents + cond_attn_output
        if model_config.get("add_cond_attn", False):
            hidden_states += cond_attn_output

    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    # 2. encoder_hidden_states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )
    # 3. condition_latents
    if use_cond:
        norm_condition_latents = self.norm2(condition_latents)
        norm_condition_latents = (
            norm_condition_latents * (1 + cond_scale_mlp[:, None])
            + cond_shift_mlp[:, None]
        )

    # Feed-forward.
    with enable_lora((self.ff.net[2],), model_config.get("latent_lora", False)):
        # 1. hidden_states
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    # 2. encoder_hidden_states
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
    # 3. condition_latents
    if use_cond:
        cond_ff_output = self.ff(norm_condition_latents)
        cond_ff_output = cond_gate_mlp.unsqueeze(1) * cond_ff_output

    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_cond:
        condition_latents = condition_latents + cond_ff_output

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None


def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    condition_latents: torch.FloatTensor = None,
    cond_temb: torch.FloatTensor = None,
    cond_rotary_emb=None,
    use_cache = False,
    kv_cache = None,
    model_config: Optional[Dict[str, Any]] = {},
):

    using_cond = condition_latents is not None
    residual = hidden_states
    with enable_lora(
        (
            self.norm.linear,
            self.proj_mlp,
        ),
        model_config.get("latent_lora", False),
    ):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    if using_cond:
        residual_cond = condition_latents
        norm_condition_latents, cond_gate = self.norm(condition_latents, emb=cond_temb)
        mlp_cond_hidden_states = self.act_mlp(self.proj_mlp(norm_condition_latents))

    attn_output = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        use_cache = use_cache,
        kv_cache = kv_cache,
        **(
            {
                "condition_latents": norm_condition_latents,
                "cond_rotary_emb": cond_rotary_emb if using_cond else None,
            }
            if using_cond
            else {}
        ),
    )
    if using_cond:
        attn_output, cond_attn_output = attn_output

    with enable_lora((self.proj_out,), model_config.get("latent_lora", False)):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    if using_cond:
        condition_latents = torch.cat([cond_attn_output, mlp_cond_hidden_states], dim=2)
        cond_gate = cond_gate.unsqueeze(1)
        condition_latents = cond_gate * self.proj_out(condition_latents)
        condition_latents = residual_cond + condition_latents

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states if not using_cond else (hidden_states, condition_latents)
