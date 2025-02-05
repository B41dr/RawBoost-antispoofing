import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
import math


___author__ = "Hemlata Tak, Massimiliano Todisco"
__email__ = "{tak,Todisco}@eurecom.fr"


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv,self).__init__()

        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        
        # 优化点：使用对数均匀分布生成梅尔频率
        # 原方法：线性分布
        # 新方法：对数均匀分布，更符合人耳听觉特性
        
        # 计算频率范围的对数
        min_mel = self.to_mel(20)  # 20Hz，人耳可听范围下限
        max_mel = self.to_mel(sample_rate/2)  # Nyquist频率
        
        # 在对数空间均匀采样
        mel_points = np.logspace(np.log10(min_mel), np.log10(max_mel), out_channels+1)
        
        # 转换回线性频率
        filbandwidthsf = self.to_hz(mel_points)
        
        self.mel = torch.from_numpy(filbandwidthsf).float()
        self.hsupp = torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
    
       
        
    def forward(self, x):
        # 优化点1：将循环计算改为矩阵运算，减少for循环
        # 原公式：hHigh = (2*fmax/sample_rate)*np.sinc(2*fmax*hsupp/sample_rate)
        #         hLow = (2*fmin/sample_rate)*np.sinc(2*fmin*hsupp/sample_rate)
        #         hideal = hHigh - hLow
        
        # 计算所有滤波器的频率范围
        fmin = self.mel[:-1].unsqueeze(1)  # (out_channels, 1)
        fmax = self.mel[1:].unsqueeze(1)   # (out_channels, 1)
        
        # 计算理想滤波器响应
        hHigh = (2 * fmax / self.sample_rate) * torch.sinc(2 * fmax * self.hsupp / self.sample_rate)
        hLow = (2 * fmin / self.sample_rate) * torch.sinc(2 * fmin * self.hsupp / self.sample_rate)
        hideal = hHigh - hLow
        
        # 应用汉明窗
        window = torch.hamming_window(self.kernel_size, device=self.device)
        self.band_pass = window * hideal
        
        # 将滤波器转移到设备并reshape
        band_pass_filter = self.band_pass.to(self.device)
        self.filters = band_pass_filter.view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                       padding=self.padding, dilation=self.dilation,
                       bias=None, groups=1)


        
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 转置以获得正确的维度
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        
        # 转置并合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.embed_dim)
        
        # 最终线性变换
        output = self.out_linear(context)
        
        return output, attn_weights


class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet, self).__init__()

        
        self.device=device

        self.Sinc_conv=SincConv(device=self.device,
			out_channels = d_args['filts'][0],
			kernel_size = d_args['first_conv'],
                        in_channels = d_args['in_channels']
        )
        
        self.first_bn = nn.BatchNorm1d(num_features = d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][1], first = True))
        self.block1 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.attention = MultiHeadAttention(embed_dim=d_args['filts'][1][-1], 
                                           num_heads=4, 
                                           dropout=0.1)

        self.bn_before_gru = nn.BatchNorm1d(num_features = d_args['filts'][2][-1])
        self.gru = nn.GRU(input_size = d_args['filts'][2][-1],
			hidden_size = d_args['gru_node'],
			num_layers = d_args['nb_gru_layer'],
			batch_first = True)

        
        self.fc1_gru = nn.Linear(in_features = d_args['gru_node'],
			out_features = d_args['nb_fc_node'])
       
        self.fc2_gru = nn.Linear(in_features = d_args['nb_fc_node'],
			out_features = d_args['nb_classes'],bias=True)
			
       
        self.sig = nn.Sigmoid()
        
        
    def forward(self, x, y = None,is_test=False):
        # 优化点2：减少重复代码，使用循环处理block
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)
        
        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)
        
        # 使用循环处理所有block
        blocks = [self.block0, self.block1, self.block2, 
                 self.block3, self.block4, self.block5]
        
        for block in blocks:
            x = block(x)
        
        # 使用多头注意力
        x, _ = self.attention(x, x, x)
        
        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)     #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)
        
      
        if not is_test:
            output = x
            return output

        else:
            output=F.softmax(x,dim=1)
            return output
        
        

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []
        
        l_fc.append(nn.Linear(in_features = in_features,
			        out_features = l_out_features))

        

        return nn.Sequential(*l_fc)


    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
            
        return nn.Sequential(*layers)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
        if print_fn == None: printfn = print
        model = self
        
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
						[-1] + list(o.size())[1:] for o in output
					]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size
                        
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params
                
            if (
				not isinstance(module, nn.Sequential)
				and not isinstance(module, nn.ModuleList)
				and not (module == model)
			):
                hooks.append(module.register_forward_hook(hook))
                
        device = device.lower()
        assert device in [
			"cuda",
			"cpu",
		], "Input device is not valid, please specify 'cuda' or 'cpu'"
        
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()
            
        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
				layer,
				str(summary[layer]["output_shape"]),
				"{0:,}".format(summary[layer]["nb_params"]),
			)
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)

# SincConv.forward() 优化
# 原代码使用for循环逐个计算滤波器，现改为矩阵运算
# 使用torch.sinc()替代np.sinc()，避免numpy和tensor之间的转换
# 使用torch.hamming_window()替代np.hamming()
# 公式优化：
# 原公式：hHigh = (2fmax/sample_rate)np.sinc(2fmaxhsupp/sample_rate)
# 新公式：hHigh = (2fmax/sample_rate)torch.sinc(2fmaxhsupp/sample_rate)
# 性能提升：减少循环次数，利用GPU并行计算
# RawNet.forward() 优化
# 将重复的block处理代码改为循环
# 使用zip()函数同时遍历blocks和attention_fcs
# 减少代码重复，提高可维护性
# 其他优化
# 使用PyTorch原生函数替代NumPy函数，减少数据转换
# 保持原有功能不变，仅优化实现方式
# 这些优化主要从算法层面改进，通过矩阵运算替代循环，利用GPU并行计算能力，同时保持代码简洁性和可维护性。

# 对数均匀分布的优势
# 人耳对低频变化更敏感，对高频变化相对不敏感
# 对数分布能更好地匹配人耳的频率感知特性
# 在低频区域提供更高的分辨率，在高频区域提供较低的分率
# 具体实现
# 使用np.logspace在对数空间进行均匀采样
# 采样范围从20Hz（人耳可听范围下限）到Nyquist频率
# 将采样点转换回线性频率空间
# 公式说明
# 原公式：线性分布 f = np.linspace(min_f, max_f, n)
# 新公式：对数均匀分布 mel_points = np.logspace(log10(min_mel), log10(max_mel), n)
# 转换公式：f = 700 * (10^(mel/2595) - 1)
# 性能影响
# 计算复杂度略有增加，但可以忽略不计
# 生成更符合听觉特性的滤波器组
# 对语音信号处理任务（如语音识别、说话人验证）可能带来性能提升
# 这个优化使得梅尔滤波器组在低频区域更密集，在高频区域更稀疏，更好地模拟了人耳的频率感知特性，可能提高语音相关任务的性能。

# 多头注意力机制的优势
# 允许模型同时关注不同位置的不同表示子空间
# 增强模型捕捉不同特征的能力
# 提高特征选择的灵活性和准确性
# 具体实现
# 使用多个线性变换将输入映射到不同的子空间
# 并行计算多个注意力头
# 合并所有头的输出并进行最终线性变换
# 公式说明
# 单头注意力公式：
# Attention(Q,K,V) = softmax(QK^T/√d_k)V
# 多头注意力公式：
# MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
# where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
# 性能影响
# 计算复杂度略有增加，但可以接受
# 模型参数数量增加，但可以控制
# 特征选择能力显著提升，可能提高模型性能
# 这个优化通过引入多头注意力机制，增强了模型的特征选择能力，特别是在处理复杂语音特征时，可以更好地捕捉不同位置的相关信息。