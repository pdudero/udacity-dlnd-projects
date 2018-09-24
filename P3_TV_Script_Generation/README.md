These exercises were cloned from this [Udacity github location](https://github.com/udacity/deep-learning/tree/master/tv-script-generation). 

## Method

In this project I generated my own Simpsons TV script using RNNs.
I used the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).
I implemented functions that 
1. create vocab lookup tables,
1. create and build the model using Tensorflow, 
1. yield training batches of data,
1. set the hyperparameters
1. picks the next word in the generated text according to the probabilities output by the RNN from the previous word.

The RNN is built using Tensorflow's [BasicLSTMCells](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [MultiRNNcell](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
Tensorflow's [nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup) is used to create a word embedding layer.


## Results

The full project including results are found in the [dlnd_tv_script_generation.ipynb](dlnd_tv_script_generation.ipynb) notebook.

Here is the generated script:

```
moe_szyslak: everybody reach in and draw a pickled egg. whoever gets the black egg stays sober tonight.
barney_gumble: noooooooooo!
homer_simpson:(to barney) you got the black egg?
barney_gumble: i wish i was pretty nervous, my family's(calling out) but there's something you give a woman.
homer_simpson: but you can't leave! we're scammin' an old lady at the bar.
moe_szyslak:(pleased) whoa! look at me bein' polite!
maya: so, aren't you gonna invite me in?
homer_simpson: boy, i bid you--(choking sound)
homer_simpson:(spinning) who-o-oa! who-o-oa! who-o-oa! who-o-oa! who-o-oa! who-o-oa!
moe_szyslak: don't homer it's just.
moe_szyslak: so, uh, what are you pullin' the ripcord with?
lenny_leonard: uh-oh. maybe there's a ripcord app(loud) i can jump to see him again.
moe_szyslak:(pleased) really?
moe's_thoughts:(grimly) okay there, moe.
moe_szyslak:(calling out) uh, well, you don't get all the money to buy pants?(points to artie) this guy's the one what done the thing.
moe_szyslak: hey, your love boy, with the boy bags a deer.
barney_gumble: all right, i'm on the.
homer_simpson: how many gonna do you have?
homer_simpson: nothin' but don't need that name. i'm gonna help you reopen your bar no one ever foot or two.
homer_simpson: i wish mr. burns was back.
moe_szyslak:(sobs like teenage girl, then clearly says) if i didn't sell booze, they probably wouldn't even come here.(dawning) hey moe, gimme a bottle of--(he taps a glove slap to a great guy) good ol' stinkin', barney.
homer_simpson: barney, where's my car?!!


homer_simpson:(ringing bell) hear ye, hear ye, my daughter has something know that about a guy.
homer_simpson: i don't believe, use this one.
moe_szyslak: maybe i should just keep walking instead of going into your loved is, uh..
```

