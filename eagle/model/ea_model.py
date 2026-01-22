import copy
import json
import time
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
#from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
from .modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .cnets1 import Model as Model1
from .configs import EConfig


class EaModel(nn.Module):

    def __init__(
            self,
            use_eagle3,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)
        self.use_eagle3 = use_eagle3
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        if use_eagle3:
            self.ea_layer = Model(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)
        else:
            self.ea_layer = Model1(config, bias=bias, total_tokens=total_token, depth=depth, top_k=top_k,
                                  threshold=threshold, path=base_model_name_or_path,load_emb=True)

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        if self.use_eagle3 and config.vocab_size==config.draft_vocab_size:
            del self.ea_layer.d2t,self.ea_layer.t2d
        load_=self.ea_layer.load_state_dict(ea_layer_state_dict, strict=False)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()

        # ============ Early Prediction Configuration ============
        self.use_early_prediction = getattr(config, 'use_early_prediction', False)
        self.ea_config = config
        
        if self.use_early_prediction:
            # Get target model's number of layers
            self.num_target_layers = len(self.base_model.model.layers)
            
            # Calculate layer indices for auxiliary hidden states extraction
            # Default: layer 1, layer 1/4, layer 3/4
            aux_ratios = getattr(config, 'aux_hidden_layers_ratio', [0.03125, 0.25, 0.75])
            self.aux_hidden_layer_indices = [
                max(1, int(self.num_target_layers * ratio)) for ratio in aux_ratios
            ]
            
            # Calculate early exit layer index (default 3/4)
            early_exit_ratio = getattr(config, 'early_exit_layer_ratio', 0.75)
            self.early_exit_layer_idx = int(self.num_target_layers * early_exit_ratio)
            
            # Early prediction top-k
            self.early_prediction_top_k = getattr(config, 'early_prediction_top_k', 5)
        # ============ End Early Prediction Configuration ============

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def get_aux_hidden_layer_indices(self) -> List[int]:
        """Get the layer indices for auxiliary hidden states extraction.
        
        Returns:
            List[int]: List of layer indices to extract hidden states from.
        """
        if self.use_early_prediction:
            return self.aux_hidden_layer_indices
        return None

    def get_early_exit_layer_idx(self) -> int:
        """Get the early exit layer index.
        
        Returns:
            int: The layer index for early exit.
        """
        if self.use_early_prediction:
            return self.early_exit_layer_idx
        return None

    @classmethod
    def from_pretrained(
            cls,
            use_eagle3=True,
            base_model_path=None,
            ea_model_path=None,
            total_token=60,
            depth=7,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        # ... existing code ...
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen2ForCausalLM':
            base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        elif Type == 'Qwen3ForCausalLM':
            base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )
        else:
            base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath = os.path.join(ea_model_path, "config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")

        try:
            load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            use_eagle3,
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.ea_layer.total_tokens = total_token - 1

        return model

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
            aux_hidden_layer_indices: Optional[List[int]] = None,
            early_exit_layer_idx: Optional[int] = None,
            start_layer_idx: int = 0,
            initial_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        """
        Forward pass through target model with optional early prediction support.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            past_key_values: Cached key values
            output_orig: Whether to output original logits
            position_ids: Position ids
            aux_hidden_layer_indices: Layer indices to extract auxiliary hidden states (for early prediction)
            early_exit_layer_idx: Layer index to early exit (for early prediction Step A)
            start_layer_idx: Layer index to start from (for continuing from early exit)
            initial_hidden_states: Initial hidden states (for continuing from early exit)
            
        Returns:
            If early_exit_layer_idx is set:
                (outputs, hidden_states, aux_hidden_states_dict)
            Otherwise:
                Standard return based on output_orig flag
        """
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                aux_hidden_layer_indices=aux_hidden_layer_indices,
                early_exit_layer_idx=early_exit_layer_idx,
                start_layer_idx=start_layer_idx,
                initial_hidden_states=initial_hidden_states,
            )
            
            hidden_states = outputs[0]
            
            # Check if this is an early exit (no final norm applied, no logits needed yet)
            is_early_exit = (early_exit_layer_idx is not None and 
                           hasattr(outputs, 'early_exit_layer_idx') and 
                           outputs.early_exit_layer_idx is not None)
            
            if is_early_exit:
                # Return early exit results with auxiliary hidden states
                aux_hidden_states = getattr(outputs, 'aux_hidden_states', {})
                return outputs, hidden_states, aux_hidden_states
            
            # Standard forward: compute logits if needed
            if output_orig:
                orig = self.base_model.lm_head(hidden_states)
                return outputs, orig, hidden_states
            else:
                return outputs, hidden_states

    def forward_early_exit(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
            position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Dict[int, torch.FloatTensor], any]:
        """
        Run target model to early exit layer (Step A).
        
        This method runs the target model forward pass only up to the early exit layer
        (default 3/4 of total layers), and extracts auxiliary hidden states from
        specified layers for feature fusion.
        
        Args:
            input_ids: Input token ids (batch, seq_len)
            attention_mask: Attention mask
            past_key_values: Cached key values
            position_ids: Position ids
            
        Returns:
            Tuple of:
                - early_exit_hidden: Hidden states at early exit layer (batch, seq_len, hidden_size)
                - aux_hidden_states: Dict of {layer_idx: hidden_states} for feature fusion
                - outputs: Model outputs containing past_key_values etc.
        """
        if not self.use_early_prediction:
            raise RuntimeError("Early prediction is not enabled. Set use_early_prediction=True in config.")
        
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                aux_hidden_layer_indices=self.aux_hidden_layer_indices,
                early_exit_layer_idx=self.early_exit_layer_idx,
            )
            
            early_exit_hidden = outputs.early_exit_hidden_state
            aux_hidden_states = outputs.aux_hidden_states
            
            return early_exit_hidden, aux_hidden_states, outputs

    def forward_continue(
            self,
            early_exit_hidden: torch.FloatTensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values=None,
            position_ids: Optional[torch.LongTensor] = None,
            output_logits: bool = True,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], any]:
        """
        Continue target model from early exit layer to final layer (Step E).
        
        This method continues the target model forward pass from the early exit layer
        to the final layer, used for verification after drafting is complete.
        
        Args:
            early_exit_hidden: Hidden states from early exit layer
            attention_mask: Attention mask
            past_key_values: Cached key values (should contain KV up to early exit layer)
            position_ids: Position ids
            output_logits: Whether to compute and return logits
            
        Returns:
            Tuple of:
                - hidden_states: Final hidden states (batch, seq_len, hidden_size)
                - logits: LM head logits if output_logits=True, else None
                - outputs: Model outputs containing updated past_key_values
        """
        if not self.use_early_prediction:
            raise RuntimeError("Early prediction is not enabled. Set use_early_prediction=True in config.")
        
        with torch.inference_mode():
            # Continue from early exit layer
            outputs = self.base_model.model(
                input_ids=None,  # Not needed when continuing
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                start_layer_idx=self.early_exit_layer_idx,
                initial_hidden_states=early_exit_hidden,
            )
            
            hidden_states = outputs[0]
            
            logits = None
            if output_logits:
                logits = self.base_model.lm_head(hidden_states)
            
            return hidden_states, logits, outputs

    def early_prediction_draft(
            self,
            early_exit_hidden: torch.FloatTensor,
            aux_hidden_states: Dict[int, torch.FloatTensor],
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute early prediction and draft model (Steps B, C, D).
        
        This method:
        1. Step B: Run early prediction to get top-k candidates
        2. Step C: Feature fusion (concatenate layer hidden states)
        3. Step D: Run draft model to generate subsequent tokens
        
        Args:
            early_exit_hidden: Hidden states from early exit (3/4) layer
            aux_hidden_states: Dict of auxiliary hidden states {layer_idx: tensor}
            attention_mask: Attention mask for early prediction transformer
            position_ids: Position ids
            
        Returns:
            Tuple of:
                - early_topk_indices: Top-k token indices from early prediction (batch, seq_len, k)
                - early_topk_probs: Top-k log probabilities (batch, seq_len, k)
                - early_transformer_hidden: Hidden states after early transformer (for feature fusion)
                - fused_hidden: Fused hidden states ready for draft model
        """
        if not self.use_early_prediction:
            raise RuntimeError("Early prediction is not enabled.")
        
        # Step B: Early Prediction
        early_logits, early_transformer_hidden = self.ea_layer.early_prediction_forward(
            early_hidden_states=early_exit_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        # Get top-k candidates
        early_topk_indices, early_topk_probs = self.ea_layer.get_early_top_k_candidates(
            early_logits, 
            top_k=self.early_prediction_top_k
        )
        
        # Step C: Feature Fusion
        # Concatenate auxiliary hidden states: [layer1, layer1/4, layer3/4]
        sorted_indices = sorted(aux_hidden_states.keys())
        aux_hidden_list = [aux_hidden_states[idx] for idx in sorted_indices]
        concatenated_aux = torch.cat(aux_hidden_list, dim=-1)
        
        # Concatenate with early transformer hidden: [aux, early_transformer_hidden]
        fused_hidden = torch.cat([concatenated_aux, early_transformer_hidden], dim=-1)
        
        return early_topk_indices, early_topk_probs, early_transformer_hidden, fused_hidden

    # ... existing code ...
    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        # prefill
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # Target model forward, get logits
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            # verification
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # Adjusting the input sequence, draft model forward
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            # with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break

    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")


        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model,max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0
        max_length = max_length - self.ea_layer.total_tokens - 10
        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
