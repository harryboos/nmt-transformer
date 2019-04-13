import torch.nn.functional as F
import torch
from Text import Lang
from Text import indexes_from_sentence
from Text import padding_both
from Model import Transformer
from Mask import create_masks

def translate_sentence(sent, model, input_lang, output_lang, maxlen):
    sent_as_index = indexes_from_sentence(input_lang, sent)
    input = padding_both(sent_as_index, maxlen)
    input = torch.Tensor(input)
    source = input.unsqueeze(0) # add a dimension

    target = torch.zeros((1, maxlen))
    target[0][0] = 1

    source_mask, target_mask = create_masks(source, target)
    output = model(source, target, source_mask, target_mask)

    output = F.softmax(output, -1)
    out = torch.max(output, -1)[1]  # 1 is index, 0 is max malue
    out = out.squeeze(0)
    print('out: ', out)
    result = ''
    for idx in out:
        if idx == 0:
            break
        print('idx: ', idx)
        index = idx.data()
        print(index)
        result += output_lang.index2word[index]
    print(result)
    return result

print(1)
t_model = Transformer(*args, **kwargs)
t_model.load_state_dict(torch.load(models/mytraining8.pt))
print(2)
t_model.cuda()
t_model.eval()
sent = 'sie kennen die reise der pinguine'
result = translate_sentence(sent, t_model, 'ger', 'en', 10)
print(result)
