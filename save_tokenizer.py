from transformers import AutoTokenizer
tokenizer_EN = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
tokenizer_VN = AutoTokenizer.from_pretrained("vinai/phobert-base")
tokenizer_EN.save_pretrained("./static/tokenizer_EN")
tokenizer_VN.save_pretrained("./static/tokenizer_VN")
