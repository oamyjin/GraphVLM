from typing import List, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
import math

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .qwen.configuration_qwen import QWenConfig
from .qwen.modeling_qwen import QWenModel

# from .mpt.modeling_mpt import MPTConfig, MPTForCausalLM, MPTModel
from ..llaga_arch import LlagaMetaModel, LlagaMetaForCausalLM
from utils.constants import IGNORE_INDEX


class LlagaQWenConfig(QWenConfig):
    print("LlagaQWenConfig")
    model_type = "llaga_qwen"


class LlagaQWenModel(LlagaMetaModel, QWenModel):
    print("LlagaQWenModel")
    config_class = LlagaQWenConfig

    def __init__(self, config: QWenConfig):
        config.hidden_size = config.d_model
        super(LlagaQWenModel, self).__init__(config)
    
    def embed_tokens(self, x):
        return self.wte(x)


class LlagaQWenForCausalLM(AutoModelForCausalLM, LlagaMetaForCausalLM):
    print("LlagaQWenForCausalLM")
    config_class = LlagaQWenConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(AutoModelForCausalLM, self).__init__(config)

        if not config.tie_word_embeddings:
            raise ValueError('AutoModelForCausalLM only supports tied word embeddings')
        self.transformer = LlagaQWenModel(config)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlagaQWenModel):
            module.gradient_checkpointing = value

    def forward(self, input_ids: torch.LongTensor,
                past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None,
                attention_mask: Optional[torch.ByteTensor]=None,
                prefix_mask: Optional[torch.ByteTensor]=None,
                sequence_id: Optional[torch.LongTensor]=None,
                labels: Optional[torch.LongTensor]=None,
                return_dict: Optional[bool]=None,
                output_attentions: Optional[bool]=None,
                output_hidden_states: Optional[bool]=None,
                use_cache: Optional[bool]=None,
                graph: Optional[torch.FloatTensor] = None,
                graph_emb: Optional[torch.FloatTensor] = None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph, graph_emb)
        outputs = self.transformer(input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache)
        # FIXME: this is a hack to fix the multiple gpu inference issue in https://github.com/haotian-liu/LLaVA/issues/338
        logits = F.linear(outputs.last_hidden_state.to(self.transformer.wte.weight.device), self.transformer.wte.weight)
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale
        loss = None
        if labels is not None:
            labels = torch.roll(labels, shifts=-1)
            labels[:, -1] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.to(logits.device).view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
            }
        )
        return model_inputs
        
        # if inputs_embeds is not None:
        #     raise NotImplementedError('inputs_embeds is not implemented for Qwen yet')
        # attention_mask = kwargs['attention_mask'].bool()
        # if attention_mask[:, -1].sum() != attention_mask.shape[0]:
        #     raise NotImplementedError('Qwen does not support generation with right padding.')
        # if self.transformer.attn_uses_sequence_id and self.training:
        #     sequence_id = torch.zeros_like(input_ids[:1])
        # else:
        #     sequence_id = None
        # if past_key_values is not None:
        #     input_ids = input_ids[:, -1].unsqueeze(-1)
        # if self.transformer.prefix_lm:
        #     prefix_mask = torch.ones_like(attention_mask)
        #     if kwargs.get('use_cache') == False:
        #         raise NotImplementedError('Qwen with prefix_lm=True does not support use_cache=False.')
        # else:
        #     prefix_mask = None
        # return {'input_ids': input_ids,
        #         'attention_mask': attention_mask,
        #         'prefix_mask': prefix_mask,
        #         'sequence_id': sequence_id,
        #         'past_key_values': past_key_values,
        #         'use_cache': kwargs.get('use_cache', True),
        #         "graph": kwargs.get("graph", None),
        #         "graph_emb": kwargs.get("graph_emb", None),}

print("auto register LlagaQWenConfig...")
AutoConfig.register("llaga_qwen", LlagaQWenConfig)
print("done")
print("auto register LlagaQWenForCausalLM...")
AutoModelForCausalLM.register(LlagaQWenConfig, LlagaQWenForCausalLM)
print("done")
