import re
import unicodedata

# filename  = 'data/dev.de'
# with open(filename) as f:
#     input = f.read().splitlines()



# data = input[0:10]
# vocab = set(data)
# print('data: ',data)
# print('vocab: ',vocab)
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )



s = 'Ärger übermut Als ich in meinen ~ ^^ & 20ern war, Hatte, ich Meine erste Psychotherapie-Patientin.'
print(s)
# s = unicode_to_ascii(s)

print(s)
s = s.lower()#unicode_to_ascii(s.lower().strip())
print(s)
s = s = re.sub(r"([.!,?])", r" \1", s)
print(s)
s = re.sub(r"[^a-zA-Z.!?,ÄäÖöÜüẞß]+", r" ", s)
print(s)