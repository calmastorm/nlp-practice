# My note

## Data process

使用了pandas来读取csv文件，不过该数据集是tsv文件，一样可以用。使用`read_csv()`会返回一个叫dataframe类型的值，这个值是可以直接使用列名来取出数据的，非常实用。

遇到的一个问题是，总是找不到数据文件在哪，后来发现是working directory不正确。虽然脚本路径是当前文件的位置，但是运行代码时使用的是working directory，一般来说是项目的根目录。

## Feature extraction

复习了bag of word和n-grams模型，bow模型其实就是1-gram模型。这两个class都有`fit`和`transform`函数，作用分别是读取文档并识别出所有的词，以及计算出这些词的频率矩阵。

遇到的一个小问题是，把默认运行的代码放在class里了，导致一直说NGrams未定义，下次要注意。

## Softmax regression

batch更新方式就是一次性计算所有数据点，一次性更新所有权重。而stochastic就是每次随机抽一个index出来更新。整体的计算只要仔细看下来还是可以看懂的。要注意的就是维度的变化和计算，确实有点绕有点复杂，但不是困难。

注意此部分代码尚未测试运行。