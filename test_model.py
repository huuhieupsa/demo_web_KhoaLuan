import torch
import torch.nn as nn
from transformers import AutoTokenizer
from langdetect import detect

class LSTMNet(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim,n_layers,bidirectional,dropout):
        super(LSTMNet,self).__init__()
        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        # LSTM layer process the vector sequences
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True
                           )
        # Dense layer to predict
        self.fc = nn.Linear(hidden_dim * 2,output_dim)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()
    def forward(self,text,text_lengths):
        embedded = self.embedding(text).to(device)
        # Thanks to packing, LSTM don't see padding tokens
        # and this makes our model better
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        packed_output,(hidden_state,cell_state) = self.lstm(packed_embedded)
        # Concatenating the final forward and backward hidden states
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        #Final activation function
        outputs=self.sigmoid(dense_outputs)

        return outputs

# Load tokenizer
tokenizer_EN = AutoTokenizer.from_pretrained("./static/tokenizer_EN")
tokenizer_VN = AutoTokenizer.from_pretrained("./static/tokenizer_VN")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer_VN.vocab_size - 1
embedding_dim = 100
hidden_dim = 64
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.2
model = LSTMNet(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
model.load_state_dict(torch.load('./static/model.pth', map_location=device, weights_only=True))

# Predict
text = 'KING OF PRUSSIA, Pennsylvania/WASHINGTON (Reuters) - In the Fox & Hound sports bar, next to a shopping mall in suburban Philadelphia, four Democrats are giving speeches to potential voters as they begin their journey to try to unseat Republican congressman Pat Meehan in next yearâ€™s elections. Winning this congressional district - Pennsylvaniaâ€™s 7th - is key to Democratsâ€™ hopes of gaining the 24 seats they need to retake the U.S. House of Representatives next November. The stakes are high - control of the House would allow them to block President Donald Trumpâ€™s legislative agenda. On the surface, Democrats face a significant hurdle. In nearly two-thirds of 34 Republican-held districts that are top of the partyâ€™s target list, household income or job growth, and often both, have risen faster than state and national averages over the past two years, according to a Reuters analysis of census data.Â (Graphic:Â tmsnrt.rs/2Bgq29K) That is potentially vote-winning news for Republican incumbents, who in speeches and television ads can trumpet a strengthening economy as a product of Republican control of Washington, even though incomes and job growth began improving under former Democratic President Barack Obama. â€œThe good economy is really the only positive keeping Republicans afloat,â€ said David Wasserman, a congressional analyst with the non-partisan Cook Political Report. Still, trumpeting the good economy may have limited impact among voters in competitive districts like this mostly white southeast region of Pennsylvania bordering Delaware and New Jersey, which has switched between both parties twice in the past 15 years. Many of the two dozen voters that Reuters interviewed in the 6th and 7th districts agreed the economy was strong, that jobs were returning and wages were growing. A handful were committed Republicans and Democrats who always vote the party line. About half voted for Meehan last year, but most of those said they were unsure whether they would vote for him again in 2018. Some said they were disappointed with the Republican Partyâ€™s handling of healthcare and tax reform as well as Trumpâ€™s erratic performance. About half also felt that despite an improving economy, living costs are squeezing the middle class. Drew McGinty, one of the Democratic hopefuls at the Fox & Hound bar hoping to unseat Meehan, said the good economic numbers were misleading. â€œWhen I talk to people across the district, I hear about stagnant wages. I hear about massive debt young people are getting when they finish college. Thereâ€™s a lot out there not being told by the numbers,â€ he said. Still, Meehan, who won by 19 points in last Novemberâ€™s general election, is confident the strong economy will help him next year. He plans to run as a job creator and a champion of the middle class. â€œThe first thing people look at is whether they have got a job and income,â€ Meehan said in a telephone interview. Democratic presidential candidate Hillary Clinton carried the district by more than two points in the White House race, giving Democrats some hope that they can peel it away from Republicans next November. Kyle Kondik, a political analyst at the University of Virginia Center for Politics, said the election will essentially be a referendum on Trump. The economy might help Republicans, he said, but other issues will likely be uppermost in votersâ€™ minds, like the Republican tax overhaul - which is seen by some as favoring the rich over the middle class - and Trumpâ€™s dismantling of President Barack Obamaâ€™s initiative to expand healthcare to millions of Americans, popularly known as Obamacare. Indeed, healthcare is Americansâ€™ top concern, according to a Reuters/Ipsos poll conducted earlier this month. Next is terrorism and then the economy. â€œHealthcare will be the No. 1 issue,â€ in the election, predicted Molly Sheehan, another Democrat running to unseat Meehan. Democrats have warned that dismantling Obamacare will leave millions of Americans without health coverage, and political analysts say Republicans in vulnerable districts could be punished by angry voters. Republicans argue that Obamacare drives up costs for consumers and interferes with personal medical decisions. In Broomall, a hamlet in the 7th District, local builder Greg Dulgerian, 55, said he voted for Trump and Meehan. He still likes Trump because of his image as a political outsider, but he is less certain about Meehan. â€œIâ€™m busy, which is good,â€ Dulgerian said. â€œBut I actually make less than I did 10 years ago, because my living costs and costs of materials have gone up.â€ Dulgerian said he was not sure what Meehan was doing to address this, and he was open to a Democratic candidate with a plan to help the middle class. Ida McCausland, 65, is a registered Republican but said she is disappointed with the party. She views the overhaul of the tax system as a giveaway to the rich that will hit the middle class. â€œI will probably just go Democrat,â€ she said. Still, others interviewed said the good economy was the most important issue for them and would vote for Meehan. Â Â Â  Mike Allard, 35, a stocks day trader, voted for Clinton last year but did not cast a ballot in the congressional vote. He thinks the economy will help Meehan next year and is leaning toward voting for him. â€œLocal businesses like the way the economy is going right now,â€ he said. In the 7th district median household income jumped more than 10 percent from 2014 to 2016, from $78,000 to around $86,000, above the national average increase of 7.3 percent, while job growth held steady, the analysis of the census data shows. Overall, the U.S. economy has grown 3 percent in recent quarters, and some forecasters now think theÂ stimulus from the Republican tax cuts will sustain that rate of growth through next year. Unemployment has dropped to 4.1 percent, a 17-year low. In midterm congressional elections, history shows that voters often focus on issues other than the economy. In 1966 the economy was thriving, but President Lyndon B. Johnsonâ€™s Democrats suffered a net loss of 47 seats, partly because of growing unhappiness with the Vietnam War. In 2006, again the economy was humming, but Republicans lost a net 31 seats in the House, as voters focused on the Iraq war and the unpopularity of Republican President George W. Bush. In 2010, despite pulling the economy out of a major recession, Democrats lost control of the House to Republicans, mainly because of the passage of Obamacare, which at the time was highly unpopular with many voters. â€œWhen times are bad, the election is almost always about the economy. When the economy is good, people have the freedom and the ability to worry about other issues,â€ said Stu Rothenberg, a veteran political analyst. '
tokenizer = None
if (detect(text) == 'vi'):
    tokenizer = tokenizer_VN
else:
    tokenizer = tokenizer_EN

tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

ids = ids.to(device)
# Get the length of the input sequence
length = torch.tensor(ids.shape[1], dtype=torch.long).unsqueeze(0)
    # Evaluate the model on the input text
with torch.no_grad():
    model.eval()
    predictions = model(ids, length)

binary_predictions = torch.round(predictions).cpu().numpy()

if (int(binary_predictions[0][0]) == 0):
  print("Đây là tin giả")
else:
  print("Đây là tin thật")