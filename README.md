# Sentence mood-detection

This is a lesson practice and the sample code is　from Course of <a href="https://speech.ee.ntu.edu.tw/~tlkagk/courses.html">Professor Haungyi Lee in NTU</a>.
This work here is a process of understanding NLP model structure and Pytorch API through implementation practice.

We use LSTM model and word embedding to accomplish the training model.
Label 1 means the sentence contains positive mood (happy etc.  
Label 0 means the sentence contains negative mood ( sad, angry etc.)  

## Data description
There are three files in the dataset:
1. Training data with label:  
  0 +++$+++ i dont need love . true true story . or am i just saying that because my heart isn ' t working anymore ......
  0 +++$+++ i wish it was one week in the future ... hello i need a holiday
  0 +++$+++ at the ups store , shipping a piece of my childhood that i sold on cl .
  1 +++$+++ i am realizing how mean it was to say i hate when people like my facebook status . i dont hate it at all , just would love if you added
  0 +++$+++ taking spy ! shots .. of the council dude cutting down my fav tree
  1 +++$+++ is tired from partyin it up till 4 am last night !
  1 +++$+++ laying in bed with my babies !!!

2. Training data with no_label:
  1.mkhang mlbo . dami niang followers ee . di q rin naman sia masisisi . desperate n kng desperate , pero dpt tlga replyn nia q = d
  2.don ' t you hate it when you hang on to a seemingly interesting movie to see the ending only to find out that the ending sucks ?
  3.ok so never went to the movies because friend wasn ' t feeling well but next weekend . back to work today , wasn ' t too bad .
  4.can ' t wait to see diversity ' s performance !
  5.i love britney spears haha joey this is what u do go party with eric or do things haha
  6.wish i could call in but i can ' t do blogtalk from work
  7.1 more day !

3. Testing data:
  0,my dog ate our dinner . no , seriously ... he ate it .
  1,omg last day sooon n of primary noooooo x im gona be swimming out of school wif the amount of tears am gona cry
  2,stupid boys .. they ' re so .. stupid !
  3,hi ! do u know if the nurburgring is open for tourists today ? we want to go , but there is an event today
  4,having lunch in the office , and thinking of how to resolve this discount form issue
  5,shopping was fun
  6,wondering where all the nice weather has gone .
  7,morning ! yeeessssssss new mimi in aug
  8,umm ... maybe that ' s how the british spell it ?
  9,yes it ' s 3 : 50 am . yes i ' m still awake . yes i can ' t sleep . yes i ' ll regret it tomorrow . haha i love you mr saturday

## The training process:
start training, parameter total:6415351, trainable:241351  
Epoch 1
Train | Loss:0.49571 Acc: 75.343  
Valid | Loss:0.45264 Acc: 78.652   
saving model with acc 78.652  

Epoch 2  
Train | Loss:0.44404 Acc: 79.132  
Valid | Loss:0.43643 Acc: 79.215   
saving model with acc 79.215  

Epoch 3
Train | Loss:0.42707 Acc: 80.055  
Valid | Loss:0.42880 Acc: 79.852   
saving model with acc 79.852  

Epoch 4  
Train | Loss:0.41364 Acc: 80.867  
Valid | Loss:0.42656 Acc: 79.847   

Epoch 5  
Train | Loss:0.40156 Acc: 81.538  
Valid | Loss:0.42488 Acc: 79.951   
saving model with acc 79.951  
