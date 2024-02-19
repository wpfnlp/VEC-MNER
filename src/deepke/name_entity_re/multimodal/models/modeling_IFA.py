from typing import Any, Optional, Tuple
import math

import copy
import torch
from torch import nn, Tensor, device

from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from SoftPool import soft_pool1d, SoftPool1d

# some function
def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


def get_head_mask(
        head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        head_mask = [None] * num_hidden_layers

        return head_mask


# models
class IFAConfig(PretrainedConfig):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class IFAPreTrainedModel(PreTrainedModel):
    config_class = IFAConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init_weights(self, module):
        pass


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        self.pixel_position_embedding = nn.Embedding(49, self.embed_dim)
        self.register_buffer("pixel_position_ids", torch.arange(49).expand((1, -1)))

        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(48).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(12).expand((1, -1)))

    def forward(self, pixel_values, aux_embeddings=None, rcnn_embeddings=None):
        batch_size = pixel_values.shape[0]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = class_embeds
        
        pixel_values = self.patch_embedding(pixel_values)
        pixel_values = pixel_values.flatten(2).transpose(1, 2)   # 7*7, 768 *个子图
        pixel_values = pixel_values + self.pixel_position_embedding(self.pixel_position_ids)
        embeddings = torch.cat((embeddings, pixel_values), dim=1) # bsz, 49, 768

        if aux_embeddings is not None:
            aux_embeds = []
            for aux_embedding in aux_embeddings:
                aux_embed = self.patch_embedding(aux_embedding)
                aux_embed = aux_embed.flatten(2).transpose(1, 2).flatten(0, 1)    # 3*16, 768 3个子图
                aux_embeds.append(aux_embed)
            aux_embeds = torch.stack(aux_embeds) # bsz, 48, 768
            aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
            embeddings = torch.cat((embeddings, aux_embeds), dim=1)

        if rcnn_embeddings is not None:
            rcnn_embeds = []
            for rcnn_embedding in rcnn_embeddings:
                rcnn_embed = self.patch_embedding(rcnn_embedding)
                rcnn_embed = rcnn_embed.flatten(2).transpose(1, 2).flatten(0, 1)    # 3*4, 768 3个子图
                rcnn_embeds.append(rcnn_embed)
            rcnn_embeds = torch.stack(rcnn_embeds) # bsz, 12, 768
            rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
            embeddings = torch.cat((embeddings, rcnn_embeds), dim=1)
        return embeddings


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# class TSENet(nn.Module):

#     def __init__(self, config, reduction_ratio=16):
#         super().__init__()
#         self.config = config
#         in_channels = config.hidden_size
#         # Squeeze operation
#         self.squeeze = nn.AdaptiveAvgPool1d(1)

#         # Excitation operation
#         self.excitation = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x): # [32, 12, 40, 64]
#         bsz, ahead, length, dsize = x.size()
#         x = x.transpose(1,2).contiguous().view(bsz, -1, ahead*dsize) # [32, 40, 12*64]

#         # Squeeze: Global average pooling
#         x_squeezed = self.squeeze(x.permute(0, 2, 1)).squeeze(dim=-1) # [32, 12*64]

#         # Excitation: Fully connected layers with ReLU and Sigmoid
#         x_excited = self.excitation(x_squeezed).view(x.size(0), 1, -1) # [32, 1, 12*64]

#         # Scale the input feature maps
#         x = x * x_excited + x # [32, 40, 12*64]

#         x = x.view(bsz, length, ahead, self.config.hidden_size//12).transpose(1,2) # [32, 12, 40, 64]

#         return x

# class PAMNet(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, config, in_dim=110):
#         super(PAMNet, self).__init__()
#         self.chanel_in = in_dim
#         self.config = config

#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//11, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//11, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.softmax = nn.Softmax(dim=-1)
#     def forward(self, x): # [32, 12, 110, 64]( B X H X L X D)
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """

#         x = x.transpose(1,2).contiguous() # B X H X L X D -> B X L X H X D
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)

#         out = self.gamma*out + x
#         out = out.transpose(1,2)
#         return out
    
# class VSENet(nn.Module):

#     def __init__(self, config, reduction_ratio=16):
#         super().__init__()
#         self.config = config
#         in_channels = config.hidden_size
#         in_channels = 110
#         # Squeeze operation
#         self.squeeze = nn.AdaptiveAvgPool1d(1)

#         # Excitation operation
#         self.excitation = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels),
#             # nn.Sigmoid()
#         )
#         self.sigmoid = nn.Sigmoid()
#         # self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x): # [32, 12, 110, 64]
#         bsz, ahead, length, dsize = x.size()
#         x = x.transpose(1,2).contiguous().view(bsz, -1, ahead*dsize) # [32, 110, 12*64]

#         # Squeeze: Global average pooling
#         x_squeezed = self.squeeze(x).squeeze(dim=-1) # [32, 110]

#         # Excitation: Fully connected layers with ReLU and Sigmoid
#         x_excited = self.excitation(x_squeezed)
#         x_excited = self.sigmoid(x_excited) # + self.softmax(x_excited)
#         x_excited = x_excited.view(x.size(0), -1, 1) # [32, 110, 1]

#         # Scale the input feature maps
#         x = x * x_excited + x # [32, 110, 12*64]

#         x = x.view(bsz, length, ahead, self.config.hidden_size//12).transpose(1,2) # [32, 12, 110, 64]

#         return x
    
class Vision2TextAttention(nn.Module):
    def __init__(self, config, reduction_ratio=16):
        super().__init__()
        self.config = config
        in_channels = 110
        # Squeeze operation
        self.ada_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.ada_max_pool = nn.AdaptiveMaxPool1d(1)
        self.ada_soft_pool = SoftPool1d(config.hidden_size, force_inplace=True)


        # Excitation operation
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            # nn.Linear(in_channels // reduction_ratio, in_channels),
            # nn.Sigmoid()
        )
        self.incr = nn.Linear(in_channels // reduction_ratio, in_channels)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x): # [32, 12, 110, 64]( B X H X L X D)
        bsz, ahead, length, dsize = x.size()
        x = x.transpose(1,2).contiguous().view(bsz, -1, ahead*dsize) # [32, 110, 12*64]

        # Squeeze: Global average pooling
        x_avg_squeezed = self.ada_avg_pool(x).squeeze(dim=-1) # [32, 110]
        x_max_squeezed = self.ada_max_pool(x).squeeze(dim=-1) # [32, 110]
        x_soft_squeezed = self.ada_soft_pool(x).squeeze(dim=-1) # [32, 110]

        # Excitation: Fully connected layers with ReLU and Sigmoid
        x_avg_excited = self.excitation(x_avg_squeezed)
        x_max_excited = self.excitation(x_max_squeezed)
        x_soft_excited = self.excitation(x_soft_squeezed)

        x_excited = (x_avg_excited + x_max_excited) * x_soft_excited
        
        x_excited = self.incr(x_excited)
        x_excited = self.softmax(x_excited) # + self.softmax(x_excited)
        x_excited = x_excited.view(x.size(0), -1, 1) # [32, 110, 1]
        # Scale the input feature maps
        x = x * x_excited # [32, 110, 12*64]

        x = x.view(bsz, length, ahead, self.config.hidden_size//12).transpose(1,2) # [32, 12, 110, 64]

        return x
    
class CorrelationAwareModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.vsenet = VSENet(self.config)
        # self.pamnet = PAMNet(self.config)
        self.v2t_attn = Vision2TextAttention(self.config)

    def forward(self, x): # [32, 12, 110, 64]( B X H X L X D)
        # out = self.vsenet(x) + self.pamnet(x)
        # out = self.pamnet(x) + x
        # out = self.vsenet(x)
        out = self.v2t_attn(x)

        return out
    
# class CAMNet(nn.Module):
#     """ Channel attention module"""
#     def __init__(self, in_dim):
#         super(CAMNet, self).__init__()
#         self.chanel_in = in_dim

#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.softmax  = nn.Softmax(dim=-1)
#     def forward(self, x): # [32, 12, 110, 64]( B X H X L X D)
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         x = x.transpose(1,2).contiguous() # B X H X L X D -> B X L X H X D
#         m_batchsize, C, height, width = x.size()
#         proj_query = x.view(m_batchsize, C, -1)
#         proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         attention = self.softmax(energy_new)
#         proj_value = x.view(m_batchsize, C, -1)

#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width)

#         out = self.gamma*out + x
#         out = out.transpose(1,2)
#         return out

# class VSENetV2(nn.Module):

#     def __init__(self, config, reduction_ratio=16):
#         super().__init__()
#         self.config = config
#         in_channels = config.hidden_size
#         in_channels = 110
#         # Squeeze operation
#         self.squeeze = nn.AdaptiveAvgPool1d(1)
        
#         self.func1a = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True)
#         )
#         self.func1b = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True)
#         )
#         self.func1c = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True)
#         )
#         self.func1d = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True)
#         )

#         # Excitation operation
#         self.excitation = nn.Sequential(
#             nn.Linear(in_channels // reduction_ratio * 4, in_channels),
#             nn.Softmax()
#         )

#     def forward(self, x): # [32, 12, 110, 64]
#         bsz, ahead, length, dsize = x.size()
#         x = x.transpose(1,2).contiguous().view(bsz, -1, ahead*dsize) # [32, 110, 12*64]

#         # Squeeze: Global average pooling
#         x_squeezed = self.squeeze(x).squeeze(dim=-1) # [32, 110]

#         # Excitation: Fully connected layers with ReLU and Sigmoid
#         x_squeezed = torch.cat([self.func1a(x_squeezed), self.func1b(x_squeezed), 
#                                self.func1c(x_squeezed), self.func1d(x_squeezed)], dim=-1)
#         x_excited = self.excitation(x_squeezed).view(x.size(0), -1, 1) # [32, 110, 1]

#         # Scale the input feature maps
#         x = x * x_excited + x # [32, 110, 12*64]

#         x = x.view(bsz, length, ahead, self.config.hidden_size//12).transpose(1,2) # [32, 12, 110, 64]

#         return x    

# class VAdapter(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.layer_norm = nn.LayerNorm(self.config.hidden_size)
#         self.activation_fn = quick_gelu
#         self.fc1 = nn.Linear(self.config.hidden_size, 1024)
#         self.fc2 = nn.Linear(1024, self.config.hidden_size)

#     def forward(self, hidden_states):
#         bsz, ahead, length, dsize = hidden_states.size()
#         hidden_states = hidden_states.transpose(1,2).view(bsz, -1, ahead*dsize)
#         hidden_states = self.layer_norm(hidden_states)
#         hidden_states = self.fc1(hidden_states)
#         hidden_states = self.activation_fn(hidden_states)
#         hidden_states = self.fc2(hidden_states)
#         hidden_states = hidden_states.view(bsz, length, ahead, self.config.hidden_size//12).transpose(1,2)
#         return hidden_states, hidden_states


# class TAdapter(nn.Module):

#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.layer_norm = nn.LayerNorm(self.config.hidden_size)
#         self.activation_fn = quick_gelu
#         self.fc1 = nn.Linear(self.config.hidden_size, 1024)
#         self.fc2 = nn.Linear(1024, self.config.hidden_size)

#     def forward(self, hidden_states):
#         outputs = self.layer_norm(hidden_states) # [30, 40, 768]
#         outputs = self.fc1(outputs)
#         outputs = self.activation_fn(outputs)
#         outputs = self.fc2(outputs)
#         outputs = outputs + hidden_states
#         return outputs
    
# class BertFusion(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.fusion_function = 'softmax'

#     def forward(
#         self,
#         hidden_states,
#         visual_hidden_state=None,
#     ):
#         fusion_scores = torch.matmul(hidden_states, visual_hidden_state.transpose(-1, -2))  # bsz, 128, 49
#         if self.fusion_function == 'softmax':
#             fusion_probs = nn.Softmax(dim=-1)(fusion_scores)
#             fusion_output = torch.matmul(fusion_probs, visual_hidden_state)
#         elif self.fusion_function == 'max':
#             fusion_probs = fusion_scores.max(dim=-1)
#         return fusion_output
    

class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
        current_layer: int = None,
        output_qks=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # print('CLIPAttention', hidden_states.size(), past_key_values[0].size(), past_key_values[1].size())
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        qks = (key_states, value_states) if output_qks else None

        if past_key_values is not None:
            if current_layer == 1:
                key_states = past_key_values[0]
                value_states = past_key_values[1]
            elif current_layer > 1:
                key_states = torch.cat([past_key_values[0], key_states], dim=2)
                value_states = torch.cat([past_key_values[1], value_states], dim=2)

            # if current_layer >= 0:
            #     key_states = past_key_values[0]
            #     value_states = past_key_values[1]
            #     # key_states = torch.cat([past_key_values[0], key_states], dim=2)
            #     # value_states = torch.cat([past_key_values[1], value_states], dim=2)


        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
        
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )       
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, qks

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = quick_gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads   # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size    # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        bsz, tgt_len, embed_dim = hidden_states.size()
        # print('BertSelfAttention', hidden_states.size(), past_key_values[0].size(), past_key_values[1].size())

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        qks = (key_layer, value_layer) if output_qks else None

        if past_key_values is not None:
            if current_layer == 1:
                key_layer = past_key_values[0]
                value_layer = past_key_values[1]
            elif current_layer > 1:
                key_layer = torch.cat([past_key_values[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_values[1], value_layer], dim=2)

            # if current_layer >= 0:
            #     key_layer = past_key_values[0]
            #     value_layer = past_key_values[1]
            #     # key_layer = torch.cat([past_key_values[0], key_layer], dim=2)
            #     # value_layer = torch.cat([past_key_values[1], value_layer], dim=2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            if past_key_values is not None:
                bsz, nheads, length, dsize = past_key_values[0].size()
                visual_attention_mask = torch.ones((bsz, 1, 1, length)).to(attention_mask.device)  # bsz, 12, len, 64
                if current_layer == 1:
                    attention_mask = visual_attention_mask
                elif current_layer > 1:
                    attention_mask = torch.cat((visual_attention_mask, attention_mask), dim=-1)
                # if current_layer >= 0:
                #     attention_mask = visual_attention_mask
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)    # bsz, 128, 768
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, qks


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        self_outputs, qks = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            visual_hidden_state,
            output_qks,
            current_layer,
            past_key_values,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, qks


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, current_layer=None):
        hidden_states = self.dense(hidden_states)
        # if fusion_output is not None:
        # if current_layer > 1:
        #     fusion_states = self.fusion_dense(fusion_output)
        #     hidden_states = hidden_states + fusion_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.senet = SENet(config)
        # self.t_adapter = TAdapter(config)

    def forward(self, hidden_states, input_tensor, current_layer=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Add Text Adapter
        # if current_layer > 1:
        #     hidden_states = self.senet(hidden_states)
        # t_adapter = self.t_adapter(hidden_states)
        
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
        current_layer: int = None,
        output_qks = None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, qks = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            output_qks=output_qks,
            current_layer=current_layer,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)
        
        if output_qks:
            outputs += (qks, )
    
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs, qks = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            visual_hidden_state=visual_hidden_state,
            output_qks=output_qks,
            current_layer=current_layer,
            past_key_values=past_key_values,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, current_layer
        )
        outputs = (layer_output,) + outputs
        if output_qks: 
            outputs += (qks,)

        return outputs

    def feed_forward_chunk(self, attention_output, current_layer=None):
        intermediate_output = self.intermediate(attention_output, current_layer)
        layer_output = self.output(intermediate_output, attention_output, current_layer)
        return layer_output


class IFAEncoder(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config

        # self.tsenet = TSENet(self.text_config)
        # self.vsenet = VSENet(self.vision_config)
        # self.camnet = CAMNet(self.vision_config)
        # self.vsenetv2=VSENetV2(self.vision_config)
        self.corr_aware_module = CorrelationAwareModule(self.vision_config)

        self.vision_layers = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])
    
    def forward(
        self,
        vision_embeds=None,
        text_embeds=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None
        
        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds
        for idx in range(self.vision_config.num_hidden_layers):
            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )
            
            # vision
            # TODO: 9-12 layers past text as pkv to vision
            output_qks = True
            if idx == 0:
                bsz, length, dsize = text_embeds.size()
                visual_past_key_values = (text_embeds.view(bsz, 12, length, dsize//12),
                                          text_embeds.view(bsz, 12, length, dsize//12))
            else:
                visual_past_key_values = text_layer_output[-1]
                # text_aware = self.tsenet(text_layer_output[-1][0])
                # visual_past_key_values = (text_aware, text_aware)

            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(
                    vision_hidden_states,
                    output_attentions=output_attentions,
                    past_key_values=visual_past_key_values,
                    current_layer=idx,
                    output_qks=output_qks,
            )
            vision_hidden_states = vision_layer_output[0]

            # text
            # TODO: 9-12 layers past vison qks to text
            if idx == 0:
                bsz, length, dsize = vision_embeds.size()
                text_past_key_values = (vision_embeds.view(bsz, 12, length, dsize//12),
                                          vision_embeds.view(bsz, 12, length, dsize//12))
            else:
                # text_past_key_values = vision_layer_output[-1]
                vision_aware = self.corr_aware_module(vision_layer_output[-1][0])
                # vision_aware = self.vsenetv2(vision_layer_output[-1][0])
                text_past_key_values = (vision_aware, vision_aware)

            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_module = self.text_layer[idx]
            text_layer_output = text_layer_module(
                    text_hidden_states,
                    attention_mask=attention_mask,
                    head_mask=layer_head_mask,
                    visual_hidden_state=None,
                    past_key_values=text_past_key_values,
                    output_attentions=output_attentions,
                    output_qks=output_qks,
                    current_layer=idx,
            )
            text_hidden_states = text_layer_output[0]
            if output_attentions:
                all_vision_attentions = all_vision_attentions + (vision_layer_output[1], )
                all_text_attentions = all_text_attentions + (text_layer_output[1], )
        
        if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states, )
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states, )
        
        if not return_dict:
            return tuple(
                v for v in [
                    text_hidden_states,
                    all_text_hidden_states,
                    all_text_attentions,
                ] if v is not None)
        return BaseModelOutput(
            last_hidden_state=[text_hidden_states, vision_hidden_states], hidden_states=all_text_hidden_states, attentions=all_text_attentions
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def noised_embed(orig_embed, noise_alpha = 5):
    # print(orig_embed.size())
    # during training, we add noise to the embedding
    # during generation, we don't add noise to the embedding
    dims = torch.tensor(orig_embed.size(1) * orig_embed.size(2))
    mag_norm = noise_alpha/torch.sqrt(dims)
    return orig_embed + torch.zeros_like(orig_embed).uniform_(-mag_norm, mag_norm)

class IFAModel(nn.Module):
    def __init__(self, vision_config, text_config, add_pooling_layer=True):
        super(IFAModel, self).__init__()
        # vision model
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        # all
        self.encoder = IFAEncoder(vision_config, text_config)

        self.device = vision_config.device

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        
        pixel_values=None,
        aux_values=None, 
        rcnn_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # pre vision
        vision_embedding_output = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        if self.training:
            vision_embedding_output = noised_embed(vision_embedding_output)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            raise ValueError("token_type_ids is None!")

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)    # [None]*12

        text_embedding_output = self.text_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        if self.training:
            text_embedding_output = noised_embed(text_embedding_output)

        # all encoder
        encoder_outputs = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0][0]
        pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # return BaseModelOutputWithPooling(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )
        return BaseModelOutputWithPooling(
            last_hidden_state=encoder_outputs,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings