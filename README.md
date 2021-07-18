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
[ Epoch1: 1407/1407 ] loss:0.469 acc:19.531 Epoch1: 27/1407 ] loss:0.690 acc:53.125 [ Epoch1: 53/1407 ] loss:0.687 acc:53.125 [ Epoch1: 79/1407 ] loss:0.671 acc:62.500 [ Epoch1: 105/1407 ] loss:0.621 acc:70.312 [ Epoch1: 131/1407 ] loss:0.533 acc:72.656 [ Epoch1: 158/1407 ] loss:0.533 acc:76.562 [ Epoch1: 184/1407 ] loss:0.491 acc:77.344 [ Epoch1: 210/1407 ] loss:0.518 acc:75.000 [ Epoch1: 236/1407 ] loss:0.514 acc:71.875 [ Epoch1: 261/1407 ] loss:0.593 acc:71.875 [ Epoch1: 288/1407 ] loss:0.555 acc:72.656 [ Epoch1: 314/1407 ] loss:0.465 acc:80.469 [ Epoch1: 340/1407 ] loss:0.464 acc:77.344 [ Epoch1: 366/1407 ] loss:0.452 acc:77.344 [ Epoch1: 393/1407 ] loss:0.482 acc:78.906 [ Epoch1: 419/1407 ] loss:0.500 acc:74.219 [ Epoch1: 444/1407 ] loss:0.575 acc:69.531 [ Epoch1: 470/1407 ] loss:0.546 acc:72.656 [ Epoch1: 496/1407 ] loss:0.434 acc:79.688 [ Epoch1: 522/1407 ] loss:0.447 acc:76.562 [ Epoch1: 549/1407 ] loss:0.452 acc:81.250 [ Epoch1: 575/1407 ] loss:0.524 acc:69.531 [ Epoch1: 601/1407 ] loss:0.561 acc:73.438 [ Epoch1: 627/1407 ] loss:0.483 acc:75.781 [ Epoch1: 654/1407 ] loss:0.479 acc:75.781 [ Epoch1: 680/1407 ] loss:0.503 acc:74.219 [ Epoch1: 706/1407 ] loss:0.446 acc:76.562 [ Epoch1: 733/1407 ] loss:0.478 acc:77.344 [ Epoch1: 758/1407 ] loss:0.415 acc:76.562 [ Epoch1: 784/1407 ] loss:0.409 acc:79.688 [ Epoch1: 809/1407 ] loss:0.481 acc:78.906 [ Epoch1: 836/1407 ] loss:0.567 acc:70.312 [ Epoch1: 862/1407 ] loss:0.474 acc:75.000 [ Epoch1: 888/1407 ] loss:0.417 acc:82.812 [ Epoch1: 914/1407 ] loss:0.444 acc:75.781 [ Epoch1: 939/1407 ] loss:0.429 acc:79.688 [ Epoch1: 964/1407 ] loss:0.525 acc:72.656 [ Epoch1: 991/1407 ] loss:0.450 acc:80.469 [ Epoch1: 1017/1407 ] loss:0.482 acc:73.438 [ Epoch1: 1043/1407 ] loss:0.424 acc:76.562 [ Epoch1: 1070/1407 ] loss:0.420 acc:79.688 [ Epoch1: 1096/1407 ] loss:0.440 acc:78.125 [ Epoch1: 1122/1407 ] loss:0.475 acc:75.000 [ Epoch1: 1148/1407 ] loss:0.403 acc:81.250 [ Epoch1: 1174/1407 ] loss:0.408 acc:82.031 [ Epoch1: 1200/1407 ] loss:0.465 acc:78.125 [ Epoch1: 1226/1407 ] loss:0.455 acc:78.125 [ Epoch1: 1253/1407 ] loss:0.430 acc:80.469 [ Epoch1: 1278/1407 ] loss:0.489 acc:73.438 [ Epoch1: 1304/1407 ] loss:0.434 acc:83.594 [ Epoch1: 1329/1407 ] loss:0.494 acc:78.906 [ Epoch1: 1355/1407 ] loss:0.608 acc:67.188 [ Epoch1: 1382/1407 ] loss:0.493 acc:72.656
  
Train | Loss:0.49571 Acc: 75.343  
Valid | Loss:0.45264 Acc: 78.652   
saving model with acc 78.652  
-----------------------------------------------
.........
-----------------------------------------------
[ Epoch5: 1407/1407 ] loss:0.427 acc:19.531 Epoch5: 24/1407 ] loss:0.396 acc:82.812 [ Epoch5: 50/1407 ] loss:0.371 acc:84.375 [ Epoch5: 76/1407 ] loss:0.488 acc:78.125 [ Epoch5: 102/1407 ] loss:0.323 acc:85.938 [ Epoch5: 128/1407 ] loss:0.409 acc:78.125 [ Epoch5: 154/1407 ] loss:0.451 acc:79.688 [ Epoch5: 180/1407 ] loss:0.476 acc:79.688 [ Epoch5: 207/1407 ] loss:0.477 acc:73.438 [ Epoch5: 232/1407 ] loss:0.426 acc:78.906 [ Epoch5: 259/1407 ] loss:0.383 acc:85.156 [ Epoch5: 285/1407 ] loss:0.381 acc:85.156 [ Epoch5: 311/1407 ] loss:0.413 acc:81.250 [ Epoch5: 337/1407 ] loss:0.446 acc:79.688 [ Epoch5: 363/1407 ] loss:0.445 acc:79.688 [ Epoch5: 389/1407 ] loss:0.330 acc:83.594 [ Epoch5: 415/1407 ] loss:0.405 acc:82.812 [ Epoch5: 441/1407 ] loss:0.429 acc:80.469 [ Epoch5: 466/1407 ] loss:0.355 acc:83.594 [ Epoch5: 493/1407 ] loss:0.371 acc:85.156 [ Epoch5: 519/1407 ] loss:0.439 acc:82.031 [ Epoch5: 545/1407 ] loss:0.393 acc:84.375 [ Epoch5: 571/1407 ] loss:0.449 acc:76.562 [ Epoch5: 597/1407 ] loss:0.352 acc:85.156 [ Epoch5: 623/1407 ] loss:0.331 acc:86.719 [ Epoch5: 649/1407 ] loss:0.404 acc:82.031 [ Epoch5: 674/1407 ] loss:0.378 acc:82.031 [ Epoch5: 700/1407 ] loss:0.461 acc:78.125 [ Epoch5: 726/1407 ] loss:0.423 acc:80.469 [ Epoch5: 752/1407 ] loss:0.354 acc:85.938 [ Epoch5: 778/1407 ] loss:0.348 acc:89.062 [ Epoch5: 803/1407 ] loss:0.372 acc:81.250 [ Epoch5: 829/1407 ] loss:0.391 acc:77.344 [ Epoch5: 854/1407 ] loss:0.423 acc:76.562 [ Epoch5: 880/1407 ] loss:0.381 acc:82.031 [ Epoch5: 906/1407 ] loss:0.342 acc:82.031 [ Epoch5: 931/1407 ] loss:0.410 acc:82.812 [ Epoch5: 957/1407 ] loss:0.318 acc:85.156 [ Epoch5: 982/1407 ] loss:0.511 acc:71.875 [ Epoch5: 1008/1407 ] loss:0.372 acc:85.156 [ Epoch5: 1033/1407 ] loss:0.363 acc:85.156 [ Epoch5: 1059/1407 ] loss:0.390 acc:77.344 [ Epoch5: 1083/1407 ] loss:0.418 acc:80.469 [ Epoch5: 1107/1407 ] loss:0.449 acc:81.250 [ Epoch5: 1133/1407 ] loss:0.396 acc:83.594 [ Epoch5: 1159/1407 ] loss:0.355 acc:82.812 [ Epoch5: 1184/1407 ] loss:0.381 acc:84.375 [ Epoch5: 1211/1407 ] loss:0.472 acc:78.906 [ Epoch5: 1237/1407 ] loss:0.352 acc:86.719 [ Epoch5: 1263/1407 ] loss:0.431 acc:82.031 [ Epoch5: 1289/1407 ] loss:0.433 acc:75.781 [ Epoch5: 1315/1407 ] loss:0.366 acc:82.031 [ Epoch5: 1340/1407 ] loss:0.469 acc:79.688 [ Epoch5: 1365/1407 ] loss:0.421 acc:76.562 [ Epoch5: 1391/1407 ] loss:0.290 acc:88.281    
Train | Loss:0.40156 Acc: 81.538  
Valid | Loss:0.42488 Acc: 79.951   
saving model with acc 79.951  
-----------------------------------------------



