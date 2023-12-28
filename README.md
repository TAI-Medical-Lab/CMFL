## Fair Federated Cross-Modality Medical Image Segmentation via Hierarchical Contribution Rectification
# Project Overview
Cross-modality medical image offer significant value
and utility in medical applications by enhancing diagnostic
sensitivity and accuracy and broadening the scope of diagnosable
diseases. Meanwhile, federated learning (FL) provides a new
training paradigm that enables model sharing across hospitals
without sacrificing users’ privacy. However, it faces challenges
when training a fair federated model for different modal data,
encompassing unreliable collaboration fairness owing to data
sparsity, damaging performance fairness caused by heteroge-
neous data distribution, and fairness trade-offs arising from
simultaneously addressing different unfairness. In this paper,
we propose a method for optimizing collaboration and perfor-
mance fairness for Cross-Modality medical data in FL (CMFL).
CMFL aims to achieve collaboration fairness facilitation and
performance fairness enhancement that optimizes both fairness
trade-offs. By incorporating these insights, our method aims
to improve the training of federated models with cross-modal
medical data, achieving both collaboration and performance
fairness through the design of hierarchical contribution recti-
fication. CMFL rectifies client collaboration and performance
contributions based on the diverse spaces derived from cross-
modality data. Moreover, we incorporate an adaptive equilibrium
mechanism, ensuring clients’ contributions. The effectiveness
of CMFL has been demonstrated through exhibiting superior
performance, augmented collaboration equity, and enhanced
performance fairness in comparison to previous works.
# Dataset
HNSCC：https://wiki.cancerimagingarchive.net/display/Public/HNSCC


WB-FDG：https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287

# To train the model, you can run the following command.
sh headfl.sh
