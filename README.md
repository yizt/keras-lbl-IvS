# keras-lbl-IvS
keras实现人证比对论文Large-scale Bisample Learning on ID vs. Spot Face Recognition的核心思想; 当类别数非常大时(超过百万)，GPU显存可能无法装载权重参数；可以使用支配原型在每个step讲相关的类别一块训练，而不用训练所有的类别
