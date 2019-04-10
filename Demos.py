import torch

a = torch.Tensor([[1, 2, 3]])
print(a.size())
a = a.squeeze(0)
print(a.size())


b = [1 ,2, 3]
b = torch.Tensor(b)
b = b.unsqueeze(0)
print(b)




# print('output: ', output.size())
# output = F.softmax(output, -1)
# print('output softmax: ', output.size())
# sents = torch.max(output, -1)[1] # 1 is index, 0 is max malue
# print('sents: ', sents.size())


# import torch
# from Text import Lang
# from Text import indexes_from_sentence
# from Text import padding_both

# from Mask import create_masks

# def translate(sent, model, input_lang, output_lang, max_len):
#     sent_as_index = indexes_from_sentence(input_lang, sent)
#     input_sent = padding_both(sent_as_index, max_len)
#     input_sent = torch.Tensor(input_sent)
#     source = input_sent.unsqueeze(0) # add a dimension

#     target = torch.zeros((1, max_len))
#     target[0][0] = 1

#     source_mask, target_mask = create_masks(source, target)
#     output = model(source, target, source_mask, target_mask)

#     output = F.softmax(output, -1)
#     out = torch.max(output, -1)[1]  # 1 is index, 0 is max malue
#     out = out.squeeze(0)
# #     print('out: ', out)
#     result = ''
#     for idx in out:
#         if idx == 0:
#             break
#         result += output_lang.index2word[idx]
#     print(result)
#     return result