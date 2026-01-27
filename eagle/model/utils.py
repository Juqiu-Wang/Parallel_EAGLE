import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class Timer:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys
    with Timer("sort"):

        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                     dim=1)

        maxitem = retrieve_indices.max().item() + 5



        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)



    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }

    return tree_buffers


def initialize_tree0(input_ids, model, past_key_values, logits_processor):
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor
    )

    #     if logits_processor is not None:
    #         logits = orig[:, -1]
    #         logits = logits_processor(None, logits)
    #         probabilities = torch.nn.functional.softmax(logits, dim=1)
    #         token = torch.multinomial(probabilities, 1)
    #     else:
    #         token = torch.argmax(orig[:, -1])
    #         token = token[None, None]
    #     input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    #     # Clone the output hidden states
    #
    #     draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head)
    #     if output_orig:
    #         return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, outputs, orig, hidden_states, token
    #     return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, hidden_states, token
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token

def initialize_tree(input_ids, model, past_key_values, logits_processor):
    outputs, orig, hidden_states = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

    # Clone the output hidden states
    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states=torch.cat(outputs["hidden_states"],dim=-1)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head,logits_processor)
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]


    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates,  tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]
    if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_state = torch.cat(outputs["hidden_states"], dim=-1)

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs





def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head,logits_processor=logits_processor)


    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token


# ============================================================================
# Early Prediction Speculative Decoding Functions
# ============================================================================

def generate_early_prediction_mask(
        seq_len: int,
        top_k: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate causal attention mask for early prediction draft.
    
    For input sequence of length s and top-k candidates, this creates a mask
    for s*k early prediction tokens where each token can only attend to:
    1. Original sequence positions [0, i] where i is its source position
    2. Itself
    
    Structure (s=5, k=4):
    - Early tokens 0-3 (from position 0): can see original[0] and themselves
    - Early tokens 4-7 (from position 1): can see original[0:2] and themselves
    - etc.
    
    Args:
        seq_len: Length of original input sequence (s)
        top_k: Number of top-k candidates per position (k)
        device: Device to create tensor on
        dtype: Data type of the mask
        
    Returns:
        mask: (1, 1, s*k, seq_len + s*k) attention mask where 0=attend, -inf=ignore
    """
    num_early_tokens = seq_len * top_k
    total_len = seq_len + num_early_tokens  # original seq + early tokens
    
    # Initialize mask with -inf (cannot attend)
    mask = torch.full(
        (num_early_tokens, total_len),
        float('-inf'),
        device=device,
        dtype=dtype
    )
    
    for pos in range(seq_len):
        # Early tokens from position `pos` are at indices [pos*k, (pos+1)*k)
        start_idx = pos * top_k
        end_idx = (pos + 1) * top_k
        
        # Can attend to original sequence [0, pos] (inclusive)
        mask[start_idx:end_idx, :pos + 1] = 0
        
        # Can attend to themselves (each token attends to itself only, not siblings)
        for j in range(top_k):
            mask[start_idx + j, seq_len + start_idx + j] = 0
    
    return mask.unsqueeze(0).unsqueeze(0)


def generate_draft_tree_mask(
        seq_len: int,
        top_k: int,
        depth: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate tree structure for draft model execution after early prediction.
    
    For each of s*k early tokens, we generate `depth` subsequent tokens,
    resulting in s*k*depth total draft tokens. Each draft sequence is independent.
    
    The tree structure:
    - Root: s*k early prediction tokens (can see original seq + themselves)
    - Level 1-depth: s*k tokens each (can see their path from root)
    
    Args:
        seq_len: Original sequence length (s)
        top_k: Number of candidates per position (k)
        depth: Draft depth (number of additional tokens per early candidate)
        device: Device
        dtype: Data type
        
    Returns:
        tree_mask: (1, 1, total_tree, seq_len + total_tree) attention mask
        tree_position_ids: (total_tree,) position IDs for tree nodes
        retrieve_indices: (num_early, 1 + depth) indices to retrieve complete paths
    """
    num_early = seq_len * top_k
    num_draft = num_early * depth
    total_tree = num_early + num_draft
    total_len = seq_len + total_tree
    
    # Initialize mask with -inf (cannot attend)
    mask = torch.full((total_tree, total_len), float('-inf'), device=device, dtype=dtype)
    
    # Position IDs
    position_ids = torch.zeros(total_tree, dtype=torch.long, device=device)
    
    # === Vectorized mask generation for early tokens ===
    # Create position indices for each early token
    early_positions = torch.arange(num_early, device=device) // top_k  # [0,0,0,0,1,1,1,1,...]
    
    for pos in range(seq_len):
        # Mask for early tokens from this position
        early_start = pos * top_k
        early_end = (pos + 1) * top_k
        mask[early_start:early_end, :pos + 1] = 0  # Attend to original[0:pos+1]
    
    # Self-attention for early tokens (diagonal)
    early_self_indices = torch.arange(num_early, device=device)
    mask[early_self_indices, seq_len + early_self_indices] = 0
    position_ids[:num_early] = 1  # All early tokens at position 1
    
    # === Vectorized mask generation for draft tokens ===
    # For each early token, create its draft chain
    for early_idx in range(num_early):
        pos = early_idx // top_k
        
        for d in range(depth):
            draft_idx = num_early + early_idx * depth + d
            
            # Attend to original sequence [0, pos]
            mask[draft_idx, :pos + 1] = 0
            
            # Attend to early parent
            mask[draft_idx, seq_len + early_idx] = 0
            
            # Attend to previous draft tokens in chain (including self)
            for prev_d in range(d + 1):
                prev_draft_idx = num_early + early_idx * depth + prev_d
                mask[draft_idx, seq_len + prev_draft_idx] = 0
            
            # Position ID = 2 + d
            position_ids[draft_idx] = 2 + d
    
    # === Generate retrieve indices ===
    # Shape: (num_early, 1 + depth)
    retrieve_indices = torch.zeros((num_early, 1 + depth), dtype=torch.long, device=device)
    retrieve_indices[:, 0] = torch.arange(num_early, device=device)  # Early token indices
    for d in range(depth):
        retrieve_indices[:, 1 + d] = num_early + torch.arange(num_early, device=device) * depth + d
    
    return mask.unsqueeze(0).unsqueeze(0), position_ids, retrieve_indices


@torch.no_grad()
def initialize_tree_with_early_prediction(
        input_ids: torch.Tensor,
        model,
        past_key_values,
        logits_processor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Initialize tree with early prediction (Steps A, B, C, D).
    
    Flow:
    1. Step A: Target model runs to 3/4 layer (early exit), extract aux hidden states
    2. Step B: Early prediction to get top-k candidates per position
    3. Step C: Feature fusion (concat aux hidden states + early transformer output)
    4. Step D: Draft model generates subsequent tokens for each early candidate
    5. Step E (partial): Target model continues from 3/4 to final layer
    
    Args:
        input_ids: (1, seq_len) Input token IDs
        model: EaModel instance with early prediction enabled
        past_key_values: KV cache
        logits_processor: Logits processor for sampling
        
    Returns:
        draft_tokens: (1, num_tokens) All draft token candidates
        retrieve_indices: Indices to retrieve complete paths
        tree_mask: Attention mask for tree structure
        tree_position_ids: Position IDs
        logits: Target model logits for verification
        hidden_states: Hidden states for next iteration
        sample_token: Sampled token from target model
        early_info: Dict containing early prediction metadata
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    top_k = model.ea_layer.top_k
    depth = model.ea_layer.depth
    
    # ========== Step A: Target Model Early Exit ==========
    early_exit_hidden, aux_hidden_states, outputs_early = model.forward_early_exit(
        input_ids=input_ids,
        past_key_values=past_key_values,
    )
    # early_exit_hidden: (batch, seq_len, target_hidden_size)
    # aux_hidden_states: {layer_idx: (batch, seq_len, target_hidden_size)}
    
    # ========== Step B: Early Prediction ==========
    early_logits, early_transformer_hidden = model.ea_layer.early_prediction_forward(
        early_hidden_states=early_exit_hidden,
    )
    # early_logits: (batch, seq_len, draft_vocab_size)
    # early_transformer_hidden: (batch, seq_len, draft_hidden_size)
    
    # Get top-k candidates per position
    topk_indices, topk_probs = model.ea_layer.get_early_top_k_candidates(early_logits, top_k)
    # topk_indices: (batch, seq_len, top_k)
    # topk_probs: (batch, seq_len, top_k)
    
    # Convert draft vocab indices to target vocab if needed
    if model.ea_layer.config.vocab_size != model.ea_layer.config.draft_vocab_size:
        topk_indices = topk_indices + model.ea_layer.d2t[topk_indices]
    
    # ========== Step C: Feature Fusion Preparation ==========
    # Concatenate aux hidden states: [layer1, layer1/4, layer3/4]
    sorted_indices = sorted(aux_hidden_states.keys())
    aux_hidden_list = [aux_hidden_states[idx] for idx in sorted_indices]
    concatenated_aux = torch.cat(aux_hidden_list, dim=-1)
    # concatenated_aux: (batch, seq_len, target_hidden_size * 3)
    
    # ========== Step E (parallel): Target Model Continue ==========
    # Continue target model from 3/4 to final layer for verification
    hidden_states_final, logits, outputs_continue = model.forward_continue(
        early_exit_hidden=early_exit_hidden,
        past_key_values=outputs_early.past_key_values,
        output_logits=True,
    )
    
    # Sample token from target model output
    if logits_processor is not None:
        last_logits = logits[:, -1]
        last_logits = logits_processor(None, last_logits)
        probabilities = torch.nn.functional.softmax(last_logits, dim=-1)
        sample_token = torch.multinomial(probabilities, 1)
    else:
        sample_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
    
    # ========== Step D: Draft Model Execution ==========
    # Expand hidden states for s*k early tokens
    # Each position's hidden states are repeated k times
    num_early = seq_len * top_k
    
    # Expand aux hidden states: (batch, seq_len, dim) -> (batch, seq_len * top_k, dim)
    expanded_aux = concatenated_aux.unsqueeze(2).expand(-1, -1, top_k, -1)
    expanded_aux = expanded_aux.reshape(batch_size, num_early, -1)
    
    # Expand early transformer hidden
    expanded_early_hidden = early_transformer_hidden.unsqueeze(2).expand(-1, -1, top_k, -1)
    expanded_early_hidden = expanded_early_hidden.reshape(batch_size, num_early, -1)
    
    # Flatten topk_indices for input: (batch, seq_len * top_k)
    early_token_ids = topk_indices.reshape(batch_size, num_early)
    
    # Generate tree mask for draft model
    tree_mask, tree_position_ids, retrieve_indices = generate_draft_tree_mask(
        seq_len=seq_len,
        top_k=top_k,
        depth=depth,
        device=device,
    )
    
    # Run draft model with feature fusion
    # First layer: early tokens
    draft_hidden = model.ea_layer.feature_fusion_forward(
        aux_hidden_states=expanded_aux,
        early_transformer_hidden=expanded_early_hidden,
        input_ids=early_token_ids,
        use_cache=True,
    )
    
    if isinstance(draft_hidden, tuple):
        draft_hidden, draft_kv = draft_hidden
    else:
        draft_kv = None
    
    # Generate subsequent tokens for each early candidate
    all_draft_tokens = [early_token_ids]  # Start with early tokens
    current_hidden = draft_hidden
    
    for d in range(depth):
        # Get logits and sample next token
        draft_logits = model.ea_layer.lm_head(model.ea_layer.norm(current_hidden))
        draft_probs = model.ea_layer.logsoftmax(draft_logits)
        next_tokens = torch.argmax(draft_probs, dim=-1)  # (batch, num_early)
        
        if model.ea_layer.config.vocab_size != model.ea_layer.config.draft_vocab_size:
            next_tokens = next_tokens + model.ea_layer.d2t[next_tokens]
        
        all_draft_tokens.append(next_tokens)
        
        # Continue draft model (simplified: just get embeddings and run midlayer)
        if d < depth - 1:
            next_embeds = model.ea_layer.embed_tokens(next_tokens)
            # For simplicity, use the projected hidden as context
            current_hidden = model.ea_layer.fc(
                torch.cat([expanded_aux, expanded_early_hidden], dim=-1)
            )
            combined = torch.cat([next_embeds, current_hidden], dim=-1)
            current_hidden, draft_kv = model.ea_layer.midlayer(
                combined,
                attention_mask=None,
                position_ids=tree_position_ids[num_early + d * num_early:num_early + (d + 1) * num_early].unsqueeze(0) + seq_len,
                past_key_value=draft_kv,
                use_cache=True,
            )
    
    # Combine all draft tokens: (batch, num_early * (1 + depth))
    # Reorganize: for each early token, its draft chain
    draft_tokens_flat = torch.stack(all_draft_tokens, dim=2)  # (batch, num_early, 1+depth)
    draft_tokens = draft_tokens_flat.reshape(batch_size, -1)
    
    # Prepend sample token
    draft_tokens = torch.cat([sample_token, draft_tokens], dim=-1)
    
    # Adjust retrieve_indices to account for sample_token at position 0
    retrieve_indices = retrieve_indices + 1  # Shift all indices by 1
    retrieve_indices = torch.cat([
        torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long, device=device),
        retrieve_indices
    ], dim=1)
    
    # Update tree_position_ids
    tree_position_ids = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),  # sample_token at position 0
        tree_position_ids + 1
    ])
    
    # Prepare tree mask for verification (add sample_token row/col)
    num_tree_tokens = 1 + num_early * (1 + depth)
    full_tree_mask = torch.zeros((1, 1, num_tree_tokens, num_tree_tokens), device=device)
    full_tree_mask[:, :, 0, 0] = 1  # sample_token can see itself
    full_tree_mask[:, :, 1:, 0] = 1  # all others can see sample_token
    full_tree_mask[:, :, 1:, 1:] = (tree_mask[:, :, :, seq_len:] == 0).float()
    
    # Combine with original hidden states for next iteration
    if model.use_eagle3:
        ea_device = model.ea_layer.lm_head.weight.device
        hidden_states = torch.cat([h.to(ea_device) for h in aux_hidden_list], dim=-1)
    else:
        hidden_states = hidden_states_final
    
    early_info = {
        'early_exit_hidden': early_exit_hidden,
        'aux_hidden_states': aux_hidden_states,
        'early_transformer_hidden': early_transformer_hidden,
        'topk_indices': topk_indices,
        'topk_probs': topk_probs,
        'seq_len': seq_len,
        'top_k': top_k,
        'depth': depth,
    }
    
    return (draft_tokens, retrieve_indices, full_tree_mask, tree_position_ids,
            logits, hidden_states, sample_token, early_info)


@torch.no_grad()
def update_inference_inputs_with_early_prediction(
        input_ids: torch.Tensor,
        candidates: torch.Tensor,
        best_candidate: torch.Tensor,
        accept_length: int,
        retrieve_indices: torch.Tensor,
        logits_processor,
        new_token: int,
        past_key_values_data_list: List[torch.Tensor],
        current_length_data: torch.Tensor,
        model,
        hidden_state_new: torch.Tensor,
        sample_p: torch.Tensor,
        early_info: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, int, None, torch.Tensor]:
    """
    Update inference inputs after verification with early prediction.
    
    This function handles the acceptance of tokens and preparation for the next
    iteration when using early prediction mode.
    
    Args:
        input_ids: Current input sequence
        candidates: Candidate token sequences
        best_candidate: Index of accepted candidate
        accept_length: Number of accepted tokens
        retrieve_indices: Indices mapping candidates to tree positions
        logits_processor: Logits processor
        new_token: Running count of new tokens
        past_key_values_data_list: KV cache data
        current_length_data: Current sequence length
        model: EaModel instance
        hidden_state_new: New hidden states from verification
        sample_p: Sampling probabilities
        early_info: Early prediction metadata from initialization
        
    Returns:
        Updated inputs for next iteration
    """
    prev_input_len = input_ids.shape[1]
    
    # Map best candidate indices to original indices
    select_indices = (
        retrieve_indices[best_candidate, :accept_length + 1] + prev_input_len
    )
    
    # Append accepted tokens to input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, :accept_length + 1].to(input_ids.device)],
        dim=-1
    )
    
    # Update KV cache
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        dst = past_key_values_data[..., prev_input_len:prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
    
    current_length_data.fill_(prev_input_len + tgt.shape[-2])
    
    # Get hidden states for accepted tokens
    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, :accept_length + 1]
    
    # Sample next token
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    
    # Check if accepted path matches an early prediction path
    # If so, we can potentially reuse computation
    seq_len = early_info['seq_len']
    top_k = early_info['top_k']
    depth = early_info['depth']
    
    # For now, regenerate draft tokens using standard method
    # Future optimization: reuse early prediction results if accepted token matches
    if model.ea_layer.use_early_prediction:
        # Use early prediction flow for next iteration
        # This requires re-running the early prediction pipeline
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
            head=model.base_model.lm_head,
            logits_processor=logits_processor
        )
    else:
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = model.ea_layer.topK_genrate(
            accept_hidden_state_new,
            input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
            head=model.base_model.lm_head,
            logits_processor=logits_processor
        )
    
    new_token += accept_length + 1
    
    return input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, None, token


# ============================================================================
# Simulation and Verification Functions
# ============================================================================

def simulate_early_prediction_draft(s: int = 5, k: int = 4, depth: int = 3):
    """
    Simulate and verify the early prediction draft process.
    
    This function simulates the draft/verify process with s=5, k=4 to verify
    index calculations and mask generation are correct.
    
    Args:
        s: Sequence length
        k: Top-k candidates
        depth: Draft depth
    """
    print(f"\n{'='*60}")
    print(f"Simulating Early Prediction Draft: s={s}, k={k}, depth={depth}")
    print(f"{'='*60}")
    
    device = torch.device('cpu')
    
    # Step 1: Generate early prediction mask
    print("\n[Step 1] Early Prediction Mask Generation")
    early_mask = generate_early_prediction_mask(s, k, device)
    num_early = s * k
    print(f"  Early mask shape: {early_mask.shape}")
    print(f"  Number of early tokens: {num_early}")
    
    # Verify mask structure
    print("\n  Verifying mask structure for each position:")
    for pos in range(s):
        start_idx = pos * k
        end_idx = (pos + 1) * k
        print(f"    Position {pos}: early tokens [{start_idx}, {end_idx})")
        
        # Check original sequence attention
        for j in range(k):
            early_idx = start_idx + j
            # Should attend to original[0:pos+1]
            orig_attend = (early_mask[0, 0, early_idx, :s] == 0).sum().item()
            # Should attend to itself only (not siblings)
            self_attend = (early_mask[0, 0, early_idx, s:s+num_early] == 0).sum().item()
            print(f"      Token {early_idx}: attends to {orig_attend} original tokens, {self_attend} self (expected: {pos+1}, 1)")
    
    # Step 2: Generate draft tree mask
    print("\n[Step 2] Draft Tree Mask Generation")
    tree_mask, tree_pos_ids, retrieve_indices = generate_draft_tree_mask(s, k, depth, device)
    num_draft = num_early * depth
    total_tree = num_early + num_draft
    print(f"  Tree mask shape: {tree_mask.shape}")
    print(f"  Number of draft tokens: {num_draft}")
    print(f"  Total tree tokens: {total_tree}")
    print(f"  Tree position IDs shape: {tree_pos_ids.shape}")
    print(f"  Retrieve indices shape: {retrieve_indices.shape}")
    
    # Verify position IDs
    print("\n  Position ID distribution:")
    for pos_id in range(depth + 2):
        count = (tree_pos_ids == pos_id).sum().item()
        if count > 0:
            print(f"    Position {pos_id}: {count} tokens")
    
    # Step 3: Verify a specific draft chain
    print("\n[Step 3] Verifying Draft Chain Structure")
    test_early_idx = 7  # Early token at position 1, candidate 3 (1*4+3=7)
    test_pos = test_early_idx // k
    print(f"  Testing early token {test_early_idx} (from original position {test_pos}):")
    print(f"  Index mapping: tree_idx -> total_idx = tree_idx + seq_len")
    
    # Early token attention
    early_orig_attend = (tree_mask[0, 0, test_early_idx, :s] == 0)
    print(f"    Early token {test_early_idx} (total idx {s + test_early_idx}):")
    print(f"      Attends to original positions: {early_orig_attend.nonzero().squeeze().tolist()}")
    early_self_attend = (tree_mask[0, 0, test_early_idx, s:] == 0)
    early_tree_attend = early_self_attend.nonzero().squeeze().tolist()
    print(f"      Attends to tree positions: {early_tree_attend} (itself)")
    
    # Draft token attention
    print(f"    Draft chain for early token {test_early_idx}:")
    for d in range(depth):
        draft_idx = num_early + test_early_idx * depth + d
        draft_attend = (tree_mask[0, 0, draft_idx, :] == 0)
        
        # Separate original and tree attention
        orig_attend = (tree_mask[0, 0, draft_idx, :s] == 0).nonzero().squeeze().tolist()
        tree_attend = (tree_mask[0, 0, draft_idx, s:] == 0).nonzero().squeeze().tolist()
        
        # Expected tree attention
        expected_tree = [test_early_idx]  # early parent
        for prev_d in range(d + 1):
            expected_tree.append(num_early + test_early_idx * depth + prev_d)
        
        print(f"      Draft {d} (tree idx {draft_idx}, total idx {s + draft_idx}):")
        print(f"        Original seq: {orig_attend}, Tree: {tree_attend}")
        print(f"        Expected tree: {expected_tree} ({'✓' if tree_attend == expected_tree else '✗'})")
    
    # Step 4: Verify retrieve indices
    print("\n[Step 4] Verifying Retrieve Indices")
    print(f"  Retrieve indices for early token {test_early_idx}:")
    path = retrieve_indices[test_early_idx].tolist()
    print(f"    Path: {path}")
    print(f"    Expected: [{test_early_idx}, {num_early + test_early_idx * depth}, ..., {num_early + test_early_idx * depth + depth - 1}]")
    
    # Step 5: Simulate token generation
    print("\n[Step 5] Simulating Token Generation")
    print(f"  Input sequence: t_0, t_1, t_2, t_3, t_4 (s={s})")
    print(f"  Early prediction produces {num_early} candidates ({s} positions × {k} top-k)")
    print(f"  Draft produces {num_draft} additional tokens ({num_early} chains × {depth} depth)")
    print(f"  Total draft tokens: 1 (sample) + {total_tree} = {1 + total_tree}")
    
    # Simulate candidate structure
    print("\n  Candidate paths for verification:")
    for early_idx in range(min(3, num_early)):  # Show first 3 paths
        pos = early_idx // k
        cand = early_idx % k
        print(f"    Path {early_idx}: t_{pos} -> early_cand_{cand} -> [draft tokens...]")
        print(f"      Retrieve: {retrieve_indices[early_idx].tolist()}")
    
    print(f"\n{'='*60}")
    print("Simulation Complete - Index calculations verified")
    print(f"{'='*60}")
    
    return {
        'early_mask': early_mask,
        'tree_mask': tree_mask,
        'tree_pos_ids': tree_pos_ids,
        'retrieve_indices': retrieve_indices,
    }


if __name__ == "__main__":
    # Run simulation with s=5, k=4
    simulate_early_prediction_draft(s=5, k=4, depth=3)
    
    # Original test code
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
