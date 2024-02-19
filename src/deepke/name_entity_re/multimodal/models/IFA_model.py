import torch
from torch import nn
from torchcrf import CRF
import numpy as np
import torch.nn.functional as F
from .modeling_IFA import IFAModel
from transformers import BertConfig, BertModel, CLIPConfig, CLIPModel

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
# def contrastive_loss(logits: torch.Tensor, dim: int) -> torch.Tensor:
#     neg_ce = torch.diag(nn.functional.log_softmax(logits, dim=dim))
#     return -neg_ce.mean()


# def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
#     caption_loss = contrastive_loss(similarity, dim=0)
#     image_loss = contrastive_loss(similarity, dim=1)
#     return (caption_loss + image_loss) / 2.0

def get_logits( image_features, text_features, logit_scale):
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # 计算image_features @ text_features.T相似度矩阵
    # cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.t()
    # logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale
    logits_per_text = logits_per_image.t()

    return logits_per_image, logits_per_text

def cal_clip_loss(image_features, text_features, logit_scale):
    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    # contrastive_loss = clip_loss(logits_per_text)
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    contrastive_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2
    # contrastive_loss = F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)

    return contrastive_loss


class IFANERCRFModel(nn.Module):
    def __init__(self, label_list, args):
        super(IFANERCRFModel, self).__init__()
        self.args = args
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        print(self.vision_config)
        print(self.text_config)

        self.vision_config.device = args.device
        self.ifa_model = IFAModel(self.vision_config, self.text_config)

        self.num_labels  = len(label_list)-1 # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.text_config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

        # self.conv1d = nn.Conv1d(in_channels=110, out_channels=40, kernel_size=3, padding=1)
        # self.vision_embed_dim = self.vision_config.hidden_size
        # self.text_embed_dim = self.text_config.hidden_size
        # self.projection_dim = 512
        # self.v_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        # self.t_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # load:
        vision_names, text_names = [], []
        ifa_model_dict = self.ifa_model.state_dict()
        for name in ifa_model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    ifa_model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    ifa_model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.ifa_model.load_state_dict(ifa_model_dict)

    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None, 
            labels=None, 
            images=None, 
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        # bsz = input_ids.size(0)

        # with torch.no_grad():
        output = self.ifa_model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs, 
                            rcnn_values=rcnn_imgs,
                            return_dict=True,
                            output_hidden_states=True)


        # sequence_output = torch.cat([output.hidden_states[-1],output.hidden_states[-2]], dim=-1)
        sequence_output = output.last_hidden_state[0][0]      # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)             # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            text_features, vision_features = output.last_hidden_state[0][0], output.last_hidden_state[0][1]
            text_features = torch.max(text_features, dim=1)[0]
            vision_features = torch.max(vision_features, dim=1)[0]

            # vision_features = self.v_projection(vision_features)
            # text_features = self.t_projection(text_features)

            contrastive_loss = cal_clip_loss(vision_features, text_features, self.logit_scale)

            crf_loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')  # 去掉CLS

            loss = crf_loss + contrastive_loss
            return logits, loss
        return logits, None