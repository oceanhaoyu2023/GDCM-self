"""Top-level GDCM-EOF generator model."""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import (
    ASPP,
    ConvLSTM,
    Decoder_Block,
    FrameStack,
    RobustMeanInterpolate,
    SequenceGRU,
    TimeStack,
    outputStack,
    outputStack_fuse,
)

class GDCMEOFGenerator(nn.Module):
    def __init__(self, input_channel):
        super(GDCMEOFGenerator, self).__init__()
        self.conditioningStack = TimeStack(input_channel)
        self.simple_complete = RobustMeanInterpolate(channels=7, max_kernel_size=15)
        self.LCStack = FrameStack(4)

        #         self.num_hidden_4 = [128,128]
        self.num_hidden_3 = [128, 128]
        self.num_hidden_2 = [128, 128]
        self.num_hidden_1 = [128, 128]

        self.convlstm_num = 2

        #         self.convlstm_list_4 = []
        self.convlstm_list_3 = []
        self.convlstm_list_2 = []
        self.convlstm_list_1 = []

        #         self.height_4 = 8*4
        #         self.width_4 = 16*4

        self.height_3 = 45
        self.width_3 = 90

        self.height_2 = 90
        self.width_2 = 180

        self.height_1 = 180
        self.width_1 = 360

        self.frame_channel = 24

        #         self.attention_size_4 = 384
        self.attention_size_3 = 192
        self.attention_size_2 = 96
        self.attention_size_1 = 48

        self.input_length = 12
        self.out_len = 1

        #         self.cell_list_4 = ConvLSTM(input_dim=4,
        #                  hidden_dim=[128, 256, 12],
        #                  kernel_size=[(3, 3),(3,3),(3,3)],
        #                  num_layers=3,
        #                  batch_first=True,
        #                  bias = True,
        #                  return_all_layers = True)

        #         self.cell_list_3 = ConvLSTM(input_dim=1,
        #                  hidden_dim=[128, 128+64, 12],
        #                  kernel_size=[(3, 3),(3,3),(3,3)],
        #                  num_layers=3,
        #                  batch_first=True,
        #                  bias = True,
        #                  return_all_layers = True)

        self.cell_list_2 = ConvLSTM(input_dim=65,
                                    hidden_dim=[128, 128 + 64, 12],
                                    kernel_size=[(9, 9), (9, 9), (9, 9)],
                                    num_layers=3,
                                    batch_first=True,
                                    bias=True,
                                    return_all_layers=True)

        self.cell_list_1 = ConvLSTM(input_dim=65,
                                    hidden_dim=[128, 128 + 64, 12],
                                    kernel_size=[(9, 9), (9, 9), (9, 9)],
                                    num_layers=3,
                                    batch_first=True,
                                    bias=True,
                                    return_all_layers=True)

        #         self.attention_func_4 = nn.Sequential(
        #             nn.AdaptiveAvgPool2d([1, 1]),
        #             nn.Flatten(),
        #             nn.Linear(512-128+288, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, self.attention_size_4),
        #             nn.Sigmoid())

        #         self.attention_func_3 = nn.Sequential(
        #             nn.AdaptiveAvgPool2d([1, 1]),
        #             nn.Flatten(),
        #             nn.Linear(276, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, self.attention_size_3),
        #             nn.Sigmoid())

        self.attention_func_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d([1, 1]),
            nn.Flatten(),
            nn.Linear(180 + 64*7+7, 256),
            nn.ReLU(),
            nn.Linear(256, self.attention_size_2),
            nn.Sigmoid())

        self.attention_func_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d([1, 1]),
            nn.Flatten(),
            nn.Linear(388 + 64*7+7, 256),
            nn.ReLU(),
            nn.Linear(256, self.attention_size_1),
            nn.Sigmoid())

        #         self.outputStack_4=outputStack_fuse(512-128+288)
        #         self.outputStack_3=outputStack_fuse(276)
        self.outputStack_2 = outputStack_fuse(180)
        self.outputStack_1 = outputStack(128)

        self.AvgPool2d = nn.AvgPool2d(kernel_size=2)

        #         self.squenceCell_4 = SequenceGRU(256)
        #         self.squenceCell_3 = SequenceGRU(256)
        self.squenceCell_2 = SequenceGRU(256)
        self.squenceCell_1 = SequenceGRU(256)

        # Decoder
        #         self.d1 = Decoder_Block([84, 192], 256)
        self.d2 = Decoder_Block([84 + 64*7+7, 96], 256)
        self.d3 = Decoder_Block([84 + 64*7+7, 48], 256)

        self.aspp = ASPP(256, 128)

    #         self.output = nn.Conv2d(16, 16, kernel_size=1, padding=0)

    #     def forward(self,CD_input,LCS_input):
    def forward(self, x_input, decoder_x_input):
        device = x_input.device
        #         feature_List_4_up = []
        feature_List_3_up = []
        feature_List_2_up = []
        feature_List_1_up = []

        x_input_dim1 = self.simple_complete(x_input[:, :, 0, :])
        x_input = torch.cat((x_input_dim1[:, :, None, :], x_input[:, :, 1:, :]), 2)
        input_x_1 = x_input

        input_x_2 = torch.zeros(input_x_1.shape[0], input_x_1.shape[1], input_x_1.shape[2], int(input_x_1.shape[3] / 2),
                                int(input_x_1.shape[4] / 2)).to(device)
        for i in range(input_x_1.shape[2]):
            input_x_2_temp = self.AvgPool2d(input_x_1[:, :, i, :])
            input_x_2[:, :, i, :] = input_x_2_temp
        #         print(x_stack2.shape)
        #         input_x_3 = torch.zeros(input_x_2.shape[0],input_x_2.shape[1],input_x_2.shape[2],int(input_x_2.shape[3]/2),int(input_x_2.shape[4]/2)).to(device)
        #         for i in range(input_x_2.shape[2]):
        #             input_x_3_temp = self.AvgPool2d(input_x_2[:,:,i,:])
        #             input_x_3[:,:,i,:] = input_x_3_temp
        #         print(x_stack3.shape)
        #         input_x_4 = torch.zeros(input_x_3.shape[0],input_x_3.shape[1],input_x_3.shape[2],int(input_x_3.shape[3]/2),int(input_x_3.shape[4]/2)).to(device)
        #         for i in range(input_x_3.shape[2]):
        #             input_x_4_temp = self.AvgPool2d(input_x_3[:,:,i,:])
        #             input_x_4[:,:,i,:] = input_x_4_temp

        CD_input = x_input.permute(0, 2, 1, 3, 4)
        #         print(CD_input.shape)
        CD_output = self.conditioningStack(CD_input[:, 0:1, :])

        CD_output.reverse()  # listé

        input_x_stack2 = input_x_2
        layer_output_list_2, last_state_list_2 = self.cell_list_2(input_x_stack2)
        last_layer_output_2 = layer_output_list_2[-1]
        #         last_layer_output_2 = torch.mean(last_layer_output_2,1)
        for i in range(last_layer_output_2.shape[1]):
            if i == 0:
                last_layer_output_2_cat = last_layer_output_2[:, i, :]
            else:
                last_layer_output_2_cat = torch.cat((last_layer_output_2_cat, last_layer_output_2[:, i, :]), 1)

        for i in range(decoder_x_input.shape[1]):
            if i == 0:
                decoder_x_input_cat = decoder_x_input[:, i, :]
            else:
                decoder_x_input_cat = torch.cat((decoder_x_input_cat, decoder_x_input[:, i, :]), 1)

        last_layer_output_2_cat = torch.cat((last_layer_output_2_cat, self.AvgPool2d(decoder_x_input_cat)), 1)
        memory_feature_2 = CD_output[1]
        #             print(torch.cat([c[-1], memory_feature_2], dim=1).shape)
        attention = self.attention_func_2(torch.cat([last_layer_output_2_cat, memory_feature_2], dim=1))
        attention = torch.reshape(attention, (-1, self.attention_size_2, 1, 1))
        memory_feature_att = memory_feature_2 * attention

        #         print(last_layer_output_2_cat.shape)
        #         print(memory_feature_att.shape)
        x_gen = self.d2(last_layer_output_2_cat, memory_feature_att)
        #         x_gen = self.outputStack_2(torch.cat([last_layer_output_2_cat, memory_feature_att], dim=1)) # [B, 256, 96, 96]
        x_gen_up_2 = self.squenceCell_2(x_gen)
        #         feature_List_2_up.append(x_gen_up_2)
        #             print(x_gen.shape)

        input_x_stack1 = input_x_1
        layer_output_list_1, last_state_list_1 = self.cell_list_1(input_x_stack1)
        last_layer_output_1 = layer_output_list_1[-1]
        for i in range(last_layer_output_1.shape[1]):
            if i == 0:
                last_layer_output_1_cat = last_layer_output_1[:, i, :]
            else:
                last_layer_output_1_cat = torch.cat((last_layer_output_1_cat, last_layer_output_1[:, i, :]), 1)
        last_layer_output_1_cat = torch.cat((last_layer_output_1_cat, decoder_x_input_cat), 1)

        #             print(c[-1].shape)
        memory_feature_1 = CD_output[2]
        #             print(torch.cat([c[-1], memory_feature_1], dim=1).shape)
        attention = self.attention_func_1(torch.cat([last_layer_output_1_cat, memory_feature_1, x_gen_up_2], dim=1))
        attention = torch.reshape(attention, (-1, self.attention_size_1, 1, 1))
        memory_feature_att = memory_feature_1 * attention
        #             print(torch.cat([h[-1], memory_feature_att], dim=1).shape)
        #         print(last_layer_output_1_cat.shape)
        #         print(memory_feature_att.shape)
        #         print(memory_feature_att.shape)
        #         x_gen = self.outputStack_1(torch.cat([last_layer_output_1_cat, memory_feature_att], dim=1)) # [B, 256, 96, 96]
        #         print(last_layer_output_1_cat.shape)
        #         print(last_layer_output_1_cat.shape)
        x_gen = self.d3(last_layer_output_1_cat, memory_feature_att)
        decoder_x_input
        x_gen = self.aspp(x_gen)
        x_gen = self.outputStack_1(x_gen)
        #         print(x_gen.shape)
        #         next_frames.append(x_gen)
        out_pred = x_gen
        #     torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        return out_pred


# Backward-compatible alias for checkpoints/scripts that used the notebook class name.
generator = GDCMEOFGenerator
