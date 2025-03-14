import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = resnet50(pretrained=True)
    
    def forward(self, x, aux_imgs=None):
        # full image prompt
        prompt_guids = self.get_resnet_prompt(x)    # 4x[bsz, 256, 2, 2]
        
        # aux_imgs: bsz x 3(nums) x 3 x 224 x 224
        if aux_imgs is not None:
            aux_prompt_guids = []   # goal: 3 x (4 x [bsz, 256, 2, 2])
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])  # 3(nums) x bsz x 3 x 224 x 224
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_resnet_prompt(aux_imgs[i]) # 4 x [bsz, 256, 2, 2]
                aux_prompt_guids.append(aux_prompt_guid)   
            return prompt_guids, aux_prompt_guids
        return prompt_guids, None

    def get_resnet_prompt(self, x):
        """generate image prompt

        Args:
            x ([torch.tensor]): bsz x 3 x 224 x 224

        Returns:
            prompt_guids ([List[torch.tensor]]): 4 x List[bsz x 256 x 2 x 2]
        """
        # image: bsz x 3 x 224 x 224
        prompt_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue # 跳过分类层（全连接层和平均池化层）
            x = layer(x)    # 逐层前向传播(bsz, 256, 56, 56)
            if 'layer' in name: # 仅处理残差块组（layer1-layer4）
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2 # 池化核尺寸为原特征图高度的一半
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)    # 平均池化
                prompt_guids.append(prompt_kv)  # 将池化后的提示键值对添加到prompt_guids列表中
        return prompt_guids


class HMNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(HMNeTREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        # 扩展BERT模型的词表大小，新增用于标记头尾实体的token(<s>、</s>、<o>、</o>)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args

        self.dropout = nn.Dropout(0.5)
        # 定义一个线性层，将BERT输出的双实体向量映射到关系类别
        # hidden_size*2表示拼接头实体和尾实体的特征（各768维 → 共1536维）
        # num_labels表示关系类别的数量
        self.classifier = nn.Linear(self.bert.config.hidden_size*2, num_labels)

        # 获取特殊标记<s><o>对应的token ID，用于匹配
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

        if self.args.use_prompt:
            self.image_model = ImageModel()

            # 输入：3840维图像特征 (来自4个ResNet层的拼接3840=256+512+1024+2048)
            # 输出：6144维特征 (ResNet4个层级 * 每个头包含key/value 2个参数矩阵 * 768维)
            self.encoder_conv =  nn.Sequential(
                                    nn.Linear(in_features=3840, out_features=800), # 降维
                                    nn.Tanh(), # 非线性激活
                                    nn.Linear(in_features=800, out_features=4*2*768) # 对齐BERT维度
                                )

            # 为每个Transformer层(共12层)学习不同的特征融合权重
            # 输入维度：4*768*2 = 6144
            # - 4表示来自ResNet的4个层级特征
            # - 768*2对应每个层级的key/value拼接特征
            # 输出维度：4 → 对应4个层级的融合权重
            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        images=None,
        aux_imgs=None,
    ):

        bsz = input_ids.size(0) # 获得批次大小
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)  # 创建视觉提示的注意力掩码（全1）
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)  # 将视觉提示掩码与文本掩码拼接
        else:
            prompt_guids = None
            prompt_attention_mask = attention_mask

        output = self.bert(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_guids,
                    output_attentions=True,
                    return_dict=True
        )

        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output # last_hidden_state是每个token的上下文化表示,pooler_output是基于[CLS]标记的整个序列的单一向量表示
        # 获取批次大小、序列长度（token数量）、隐藏状态维度（768）
        bsz, seq_len, hidden_size = last_hidden_state.shape
        # 创建实体表示存储张量,存储两个拼接的实体向量(头实体+尾实体)
        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        for i in range(bsz):
            # 寻找头实体位置索引
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            # 寻找尾实体位置索引
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        # 根据提取的实体向量输出关系分类预测结果，并计算分数
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            # 使用交叉熵计算损失
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        # full image prompt
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs) # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....,对应ResNet四个残差块的输出
        # 将主图像的将4个层级的特征在通道维度拼接并重塑
        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1) # [bsz, 3840, 2, 2] -> [bsz, 4, 3840]

        # 对每个辅助图像执行相同操作，生成3个类似张量
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        # 使用encoder_conv网络将3840维特征转换为6144维(4×2×768)，与BERT层兼容
        prompt_guids = self.encoder_conv(prompt_guids) # [bsz, 4, 3840] -> [bsz, 4, 6144]
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 * [bsz, 4, 6144]

        # 将6144维特征分成4份，每份1536维，重新分配给4个ResNet层级
        split_prompt_guids = prompt_guids.split(768*2, dim=-1) # [bsz, 4, 6144] -> 4 * [bsz, 4, 1536]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids] # 3*[4*[bsz, 4, 1536]]
        
        # 先堆叠，再在层级维度求和，最后重塑算平均，得到所有层级特征的均值表示
        # [4, bsz, 4, 1536](stack) -> [bsz, 4, 1536](sum) -> [bsz, 6144](view) -> [bsz, 6144](/4取平均)
        sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4
        
        result = []
        for idx in range(12):  # 12层BERT
            # 使用当前BERT层对应的门控网络处理平均特征，计算层级权重,通过LeakyReLU和Softmax确保权重为正且总和为1
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            # 创建全零向量，存储多个ResNet层级加权融合的特征[bsz, 4, 1536]
            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)

            # 使用einsum操作实现批量加权计算，将不同层级的特征按权重融合。
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            # 为每个辅助图像应用相同的加权融合过程
            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                # 计算辅助图像的平均特征
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                # 为辅助图像计算权重
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                # 创建全零向量，存储多个ResNet层级加权融合的特征[bsz, 4, 1536]
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                # 加权融合辅助图像特征
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            # 将主图像和辅助图像特征列表合并[bsz, 4, 1536]
            key_val = [key_val] + aux_key_vals
            # 在序列长度维度(dim=1)上拼接，形成一个更长的视觉序列[bsz, 16, 1536]
            key_val = torch.cat(key_val, dim=1)
            # 将1536维(768*2)特征分成两半：key和value各768维
            key_val = key_val.split(768, dim=-1)
            # 重塑成BERT注意力机制所需的格式：[bsz, num_heads(12), 视觉序列长度(4), 每头维度(64)],contiguous()确保内存连续性，优化性能
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()
            # 创建一个key - value对元组
            temp_dict = (key, value)
            # 添加到结果列表中，形成12层的提示列表
            result.append(temp_dict)
        return result


class HMNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HMNeTNERModel, self).__init__()
        self.args = args
        self.prompt_dim = args.prompt_dim
        self.prompt_len = args.prompt_len
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config

        if args.use_prompt:
            self.image_model = ImageModel()  # bsz, 6, 56, 56
            self.encoder_conv =  nn.Sequential(
                            nn.Linear(in_features=3840, out_features=800),
                            nn.Tanh(),
                            nn.Linear(in_features=800, out_features=4*2*768)
                            )
            self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])

        self.num_labels  = len(label_list)  # pad
        print(self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None):
        if self.args.use_prompt:
            prompt_guids = self.get_visual_prompt(images, aux_imgs)
            prompt_guids_length = prompt_guids[0][0].shape[2]
            # attention_mask: bsz, seq_len
            # prompt attention， attention mask
            bsz = attention_mask.size(0)
            prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)
        else:
            prompt_attention_mask = attention_mask
            prompt_guids = None

        bert_output = self.bert(input_ids=input_ids,
                            attention_mask=prompt_attention_mask,
                            token_type_ids=token_type_ids,
                            past_key_values=prompt_guids,
                            return_dict=True)
        sequence_output = bert_output['last_hidden_state']  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)    # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean') 
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )

    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        prompt_guids, aux_prompt_guids = self.image_model(images, aux_imgs)  # [bsz, 256, 2, 2], [bsz, 512, 2, 2]....

        prompt_guids = torch.cat(prompt_guids, dim=1).view(bsz, self.args.prompt_len, -1)   # bsz, 4, 3840
        aux_prompt_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prompt_len, -1) for aux_prompt_guid in aux_prompt_guids]  # 3 x [bsz, 4, 3840]

        prompt_guids = self.encoder_conv(prompt_guids)  # bsz, 4, 4*2*768
        aux_prompt_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prompt_guids] # 3 x [bsz, 4, 4*2*768]
        split_prompt_guids = prompt_guids.split(768*2, dim=-1)   # 4 x [bsz, 4, 768*2]
        split_aux_prompt_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prompt_guids]   # 3x [4 x [bsz, 4, 768*2]]

        result = []
        for idx in range(12):  # 12
            sum_prompt_guids = torch.stack(split_prompt_guids).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
            prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_prompt_guids)), dim=-1)

            key_val = torch.zeros_like(split_prompt_guids[0]).to(self.args.device)  # bsz, 4, 768*2
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prompt_gate[:, i].view(-1, 1), split_prompt_guids[i])

            aux_key_vals = []   # 3 x [bsz, 4, 768*2]
            for split_aux_prompt_guid in split_aux_prompt_guids:
                sum_aux_prompt_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4     # bsz, 4, 768*2
                aux_prompt_gate = F.softmax(F.leaky_relu(self.gates[idx](sum_aux_prompt_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)  # bsz, 4, 768*2
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prompt_gate[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()  # bsz, 12, 4, 64
            temp_dict = (key, value)
            result.append(temp_dict)
        return result
